//! Opt-in release measurements; see docs/architecture/query-profiling.md.
use std::{
    fmt::Write,
    hint::black_box,
    time::{Duration, Instant},
};

use zirium::{
    dialect::DialectRegistry,
    parser::ParsedFile,
    query::{Query, QueryOutput},
    semantic::{Document, LoweringMode, lower_with_dialect_registry},
};

struct Case {
    name: &'static str,
    source: &'static str,
    expected: usize,
}

fn cases(width: usize, depth: usize) -> Vec<Case> {
    vec![
        Case {
            name: "count",
            source: "count",
            expected: 1 + width * (depth + 3),
        },
        Case {
            name: "filter",
            source: r#"filter(op("arith.addi")) | count"#,
            expected: width * depth,
        },
        Case {
            name: "boolean",
            source: r#"filter((op("arith.addi") or op("arith.constant")) and not has_attr("profile.seed")) | count"#,
            expected: width * depth,
        },
        Case {
            name: "users",
            source: r#"filter(op("arith.constant")) | users | count"#,
            expected: width * (depth + 1),
        },
        Case {
            name: "union",
            source: r#"(filter(op("arith.constant")) | users union filter(has_attr("profile.seed")) | defs) | count"#,
            expected: width * (depth + 1),
        },
        Case {
            name: "subtree",
            source: r#"filter(op("func.func")) | subtree | filter(op("arith.addi")) | count"#,
            expected: width * depth,
        },
        Case {
            name: "closure",
            source: r#"filter(has_attr("profile.seed")) | closure | count"#,
            expected: width * 3,
        },
        Case {
            name: "fixpoint",
            source: r#"filter(has_attr("profile.seed")) | fixpoint(closure) | count"#,
            expected: width * (depth + 1),
        },
        Case {
            name: "emit",
            source: r#"filter(has_attr("profile.seed")) | emit | users | count"#,
            expected: width * 2,
        },
    ]
}

/// Independent functions, each containing a constant and a dependency chain.
/// Vary width for document size and depth for fixed-point iteration count.
fn fixture(width: usize, depth: usize) -> Document {
    let mut source = String::from("module {\n");
    for function in 0..width {
        writeln!(source, "func.func @f{function}() -> i32 {{").unwrap();
        source.push_str("%c = arith.constant 1 : i32\n");
        for step in 0..depth {
            let operand = if step == 0 {
                "%c".to_owned()
            } else {
                format!("%v{}", step - 1)
            };
            let attribute = if step + 1 == depth {
                r#" {profile.seed = "yes"}"#
            } else {
                ""
            };
            writeln!(
                source,
                "%v{step} = \"arith.addi\"({operand}, %c){attribute} : (i32, i32) -> i32"
            )
            .unwrap();
        }
        writeln!(source, "func.return %v{} : i32\n}}", depth - 1).unwrap();
    }
    source.push_str("}\n");
    let parsed =
        ParsedFile::parse_with_registry(source.into_bytes(), DialectRegistry::baseline()).unwrap();
    assert!(
        parsed.syntax().diagnostics().is_empty(),
        "{:?}",
        parsed.syntax().diagnostics()
    );
    let lowered =
        lower_with_dialect_registry(&parsed, LoweringMode::Strict, DialectRegistry::baseline());
    assert!(lowered.diagnostics.is_empty(), "{:?}", lowered.diagnostics);
    lowered.document.unwrap()
}

fn evaluate(query: &Query, document: &mut Document) -> usize {
    let mut checksum = 0;
    query
        .evaluate(document, DialectRegistry::baseline(), |_, output| {
            checksum += match output {
                QueryOutput::Count(count) => count,
                QueryOutput::Operations(selected) => selected.len(),
                QueryOutput::Values(values) => values.len(),
                QueryOutput::Json(json) | QueryOutput::Text(json) => json.len(),
                QueryOutput::Map(values) => values.len(),
            };
            Ok(())
        })
        .unwrap();
    checksum
}

/// Batch short operations so clock overhead does not dominate. Calibration is
/// untimed warm-up; each reported sample repeats the same number of calls.
fn measure(mut run: impl FnMut(), samples: usize) -> (usize, Vec<f64>) {
    let mut iterations = 1usize;
    loop {
        let started = Instant::now();
        for _ in 0..iterations {
            run();
        }
        if started.elapsed() >= Duration::from_millis(5) {
            break;
        }
        iterations = iterations.checked_mul(2).unwrap();
    }
    let mut times = Vec::with_capacity(samples);
    for _ in 0..samples {
        let started = Instant::now();
        for _ in 0..iterations {
            run();
        }
        times.push(started.elapsed().as_secs_f64() * 1e9 / iterations as f64);
    }
    times.sort_by(f64::total_cmp);
    (iterations, times)
}

fn report(
    phase: &str,
    case: &str,
    width: usize,
    depth: usize,
    operations: usize,
    samples: usize,
    run: impl FnMut(),
) {
    let (iterations, times) = measure(run, samples);
    println!(
        "query_profile phase={phase} case={case} width={width} depth={depth} operations={operations} batch={iterations} samples={samples} min_ns={:.0} median_ns={:.0} max_ns={:.0}",
        times[0],
        times[samples / 2],
        times[samples - 1]
    );
}

#[test]
#[ignore = "opt-in release profiling; see docs/architecture/query-profiling.md"]
// Deliberately reject misleading debug-profile measurements at runtime.
#[allow(clippy::assertions_on_constants)]
fn profile_query_language() {
    assert!(!cfg!(debug_assertions), "run this test with --release");
    let smoke = std::env::var_os("ZIRIUM_QUERY_PROFILE_SMOKE").is_some();
    let samples = if smoke { 3 } else { 7 };
    println!(
        "query_profile profile=release smoke={smoke} timing=in-process setup=excluded printing=excluded samples={samples}"
    );

    // Parsing costs depend on query text, not the input document's size.
    for case in cases(1, 4) {
        Query::parse(case.source).unwrap();
        report("parse", case.name, 0, 0, 0, samples, || {
            black_box(Query::parse(black_box(case.source)).unwrap());
        });
    }

    let widths: &[usize] = if smoke { &[4, 32] } else { &[16, 128, 1024] };
    for &width in widths {
        let depth = 4;
        let mut document = fixture(width, depth);
        let operations = document.operations().count();
        assert_eq!(operations, 1 + width * (depth + 3));
        // A direct semantic scan makes interpreter overhead visible without
        // involving MLIR parsing, process startup, or printing.
        let direct = |document: &Document| {
            document
                .operations()
                .filter(|&operation| document.operation_name(operation) == Some("arith.addi"))
                .count()
        };
        assert_eq!(direct(&document), width * depth);
        report(
            "evaluate",
            "direct_filter",
            width,
            depth,
            operations,
            samples,
            || {
                black_box(direct(black_box(&document)));
            },
        );
        for case in cases(width, depth) {
            let query = Query::parse(case.source).unwrap();
            assert_eq!(
                evaluate(&query, &mut document),
                case.expected,
                "{}",
                case.name
            );
            report(
                "evaluate",
                case.name,
                width,
                depth,
                operations,
                samples,
                || {
                    black_box(evaluate(black_box(&query), black_box(&mut document)));
                },
            );
        }
    }

    // Width alone hides repeated whole-selection work in deep fixed points.
    let depths: &[usize] = if smoke { &[16, 64] } else { &[16, 128, 1024] };
    for &depth in depths {
        let mut document = fixture(1, depth);
        let operations = document.operations().count();
        let case = cases(1, depth)
            .into_iter()
            .find(|case| case.name == "fixpoint")
            .unwrap();
        let query = Query::parse(case.source).unwrap();
        assert_eq!(evaluate(&query, &mut document), case.expected);
        report(
            "evaluate",
            "fixpoint_chain",
            1,
            depth,
            operations,
            samples,
            || {
                black_box(evaluate(black_box(&query), black_box(&mut document)));
            },
        );
    }
}
