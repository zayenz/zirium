use std::time::Instant;

use zirium::{
    dialect::DialectRegistry,
    diff::{DiffLimits, DiffOptions, compare},
    parser::ParsedFile,
    semantic::{Document, LoweringMode, lower_with_dialect_registry},
};

fn lower(source: &str) -> Document {
    let registry = DialectRegistry::baseline();
    let parsed = ParsedFile::parse_with_registry(source.as_bytes(), registry).unwrap();
    lower_with_dialect_registry(&parsed, LoweringMode::Strict, registry)
        .document
        .expect("generated workload must lower strictly")
}

fn run(name: &str, before: String, after: String) {
    let before_bytes = before.len();
    let after_bytes = after.len();
    let before = lower(&before);
    let after = lower(&after);
    let started = Instant::now();
    let diff = compare(
        &before,
        &after,
        DialectRegistry::baseline(),
        DiffOptions::default(),
        DiffLimits {
            max_work: 100_000_000,
            max_changes: 2_000_000,
        },
    )
    .expect("generated workload must compare");
    let elapsed = started.elapsed();
    let statistics = diff.statistics();
    println!(
        "{name}: before_bytes={before_bytes} after_bytes={after_bytes} changes={} \
         matched={} work={} ambiguous={} bounded_fallback={} elapsed_ms={:.3}",
        diff.len(),
        statistics.matched_operations,
        statistics.work_units,
        statistics.ambiguous_groups,
        statistics.bounded_fallback_groups,
        elapsed.as_secs_f64() * 1_000.0,
    );
}

fn operations(count: usize, repeated: bool, insertion: Option<usize>) -> String {
    let mut source = String::from("module {\n");
    for index in 0..=count {
        if insertion == Some(index) {
            source.push_str("  \"test.inserted\"() : () -> ()\n");
        }
        if index == count {
            break;
        }
        let name = if repeated {
            "test.repeated".to_owned()
        } else {
            format!("test.op{index}")
        };
        source.push_str(&format!("  \"{name}\"() : () -> ()\n"));
    }
    source.push_str("}\n");
    source
}

fn nested_functions(count: usize) -> String {
    let mut source = String::from("module {\n");
    for index in 0..count {
        source.push_str(&format!(
            "  func.func @function{index}() {{\n    \"test.body\"() : () -> ()\n    func.return\n  }}\n"
        ));
    }
    source.push_str("}\n");
    source
}

fn cfg_cycle(blocks: usize) -> String {
    let mut source = String::from("module { func.func @cycle() {\n^entry:\n  cf.br ^block0\n");
    for index in 0..blocks {
        let next = (index + 1) % blocks;
        source.push_str(&format!(
            "^block{index}:\n  \"test.body{index}\"() : () -> ()\n  cf.br ^block{next}\n"
        ));
    }
    source.push_str("} }\n");
    source
}

fn main() {
    let size = std::env::args()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(1_000);

    let unchanged = operations(size, false, None);
    run("unchanged", unchanged.clone(), unchanged);
    run(
        "one_insertion",
        operations(size, false, None),
        operations(size, false, Some(size / 2)),
    );
    run(
        "repeated",
        operations(size, true, None),
        operations(size, true, Some(size / 2)),
    );
    let nested = nested_functions(size / 10 + 1);
    run("nested_functions", nested.clone(), nested);
    let cfg = cfg_cycle(size / 20 + 2);
    run("cfg_cycle", cfg.clone(), cfg);
    let opaque = "x".repeat(size.saturating_mul(64));
    let opaque = format!(r#""test.opaque"() {{data = #vendor.data<{opaque}>}} : () -> ()"#);
    run("large_opaque", opaque.clone(), opaque);
}
