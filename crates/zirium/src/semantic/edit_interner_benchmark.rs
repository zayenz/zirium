use std::env;
use std::hint::black_box;
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::dialect::DialectRegistry;
use crate::parser::ParsedFile;

use super::{AttributeSpec, AttributeValue, Document, LoweringMode, TypeSpec, TypeValue};

const SEED: u64 = 0x5a49_5249_554d_0016;

#[derive(Clone, Copy)]
enum Kind {
    String,
    Type,
    Attribute,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Self::String => "string",
            Self::Type => "type",
            Self::Attribute => "attribute",
        }
    }
}

#[derive(Clone, Copy)]
enum Pattern {
    Distinct,
    Repeated,
}

impl Pattern {
    fn name(self) -> &'static str {
        match self {
            Self::Distinct => "distinct",
            Self::Repeated => "repeated",
        }
    }
}

#[derive(Clone, Copy)]
struct Sample {
    elapsed: Duration,
    peak_bytes: usize,
}

fn fixture() -> Document {
    let parsed = ParsedFile::parse(Arc::<[u8]>::from(
        b"\"bench.anchor\"() : () -> ()".as_slice(),
    ))
    .expect("benchmark fixture parses");
    let result = super::super::lower_with_dialect_registry(
        &parsed,
        LoweringMode::Strict,
        &DialectRegistry::EMPTY,
    );
    assert!(result.diagnostics.is_empty());
    result.document.expect("benchmark fixture lowers")
}

fn nested_type(index: usize, depth: usize) -> TypeValue {
    let mut value = TypeValue::Integer {
        width: 8 + index as u32,
        signedness: None,
    };
    for _ in 0..depth {
        value = TypeValue::Tuple(vec![value]);
    }
    value
}

fn nested_attribute(index: usize, depth: usize) -> AttributeValue {
    let mut value = AttributeValue::Integer(index.to_string());
    for level in 0..depth {
        value = AttributeValue::Dictionary(vec![(format!("level{level}"), value)]);
    }
    value
}

fn measure(kind: Kind, pattern: Pattern, count: usize, depth: usize) -> Sample {
    let mut document = fixture();
    let registry = DialectRegistry::EMPTY;
    let mut editor = document.edit(&registry).unwrap();
    let repeated_type = nested_type(0, depth);
    let repeated_attribute = nested_attribute(0, depth);
    let baseline = crate::benchmark_allocator::begin();
    let started = Instant::now();
    for index in 0..count {
        let distinct = matches!(pattern, Pattern::Distinct)
            .then_some(index)
            .unwrap_or(0);
        match kind {
            Kind::String => {
                let value = format!("bench.string.{distinct}");
                black_box(editor.intern_string(&value));
            }
            Kind::Type => {
                let value = if matches!(pattern, Pattern::Distinct) {
                    nested_type(distinct, depth)
                } else {
                    repeated_type.clone()
                };
                black_box(editor.intern_type_spec(&TypeSpec {
                    spelling: format!("bench.type.{distinct}"),
                    value,
                }));
            }
            Kind::Attribute => {
                let value = if matches!(pattern, Pattern::Distinct) {
                    nested_attribute(distinct, depth)
                } else {
                    repeated_attribute.clone()
                };
                black_box(editor.intern_attribute(&AttributeSpec {
                    name: "bench.attr".into(),
                    spelling: format!("bench.attr.{distinct}"),
                    value,
                }));
            }
        }
    }
    let elapsed = started.elapsed();
    let peak_bytes = crate::benchmark_allocator::finish(baseline);

    let expected = match pattern {
        Pattern::Distinct => count,
        Pattern::Repeated => 1,
    };
    match kind {
        Kind::String => assert_eq!(editor.working.strings.len(), expected + 1),
        Kind::Type => assert_eq!(editor.working.types.len(), expected + 1),
        Kind::Attribute => assert_eq!(editor.working.attributes.len(), expected),
    }
    black_box(editor);
    Sample {
        elapsed,
        peak_bytes,
    }
}

fn parse_list(name: &str, default: &[usize]) -> Vec<usize> {
    env::var(name).map_or_else(
        |_| default.to_vec(),
        |value| {
            value
                .split(',')
                .map(|part| part.parse().expect("benchmark dimensions are integers"))
                .collect()
        },
    )
}

fn command(program: &str, args: &[&str]) -> String {
    Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .unwrap_or_else(|| "unknown".into())
}

fn median(values: &mut [u128]) -> u128 {
    values.sort_unstable();
    values[values.len() / 2]
}

#[cfg(debug_assertions)]
fn require_release() {
    panic!("run this benchmark with --release");
}

#[cfg(not(debug_assertions))]
fn require_release() {}

#[test]
#[ignore = "release-only edit interner measurement; see processing-benchmarks.md"]
fn measure_edit_interner_scaling() {
    require_release();
    let smoke = env::var_os("ZIRIUM_EDIT_INTERNER_BENCH_SMOKE").is_some();
    let counts = parse_list(
        "ZIRIUM_EDIT_INTERNER_BENCH_COUNTS",
        if smoke { &[64] } else { &[1_000, 2_000, 4_000] },
    );
    let depths = parse_list(
        "ZIRIUM_EDIT_INTERNER_BENCH_DEPTHS",
        if smoke { &[2] } else { &[0, 4, 16] },
    );
    let warmups = env::var("ZIRIUM_EDIT_INTERNER_BENCH_WARMUPS")
        .ok()
        .map_or(if smoke { 0 } else { 1 }, |value| value.parse().unwrap());
    let runs = env::var("ZIRIUM_EDIT_INTERNER_BENCH_RUNS")
        .ok()
        .map_or(if smoke { 1 } else { 3 }, |value| value.parse().unwrap());
    assert!(runs > 0);

    println!(
        "benchmark=edit-interners profile=release seed=0x{SEED:016x} warmups={warmups} measured_runs={runs} rustc={} target={} os={}",
        command("rustc", &["-V"]),
        command("rustc", &["-vV"])
            .lines()
            .find_map(|line| line.strip_prefix("host: "))
            .unwrap_or("unknown"),
        command("uname", &["-srv"]),
    );
    println!("kind,pattern,count,depth,elapsed_ns,spread_ns,peak_allocated_bytes");

    for kind in [Kind::String, Kind::Type, Kind::Attribute] {
        let kind_depths: &[usize] = if matches!(kind, Kind::String) {
            &[0]
        } else {
            &depths
        };
        for &depth in kind_depths {
            for &count in &counts {
                assert!(count > 0);
                for pattern in [Pattern::Distinct, Pattern::Repeated] {
                    for _ in 0..warmups {
                        black_box(measure(kind, pattern, count, depth));
                    }
                    let samples = (0..runs)
                        .map(|_| measure(kind, pattern, count, depth))
                        .collect::<Vec<_>>();
                    let mut elapsed = samples
                        .iter()
                        .map(|sample| sample.elapsed.as_nanos())
                        .collect::<Vec<_>>();
                    let min = *elapsed.iter().min().unwrap();
                    let max = *elapsed.iter().max().unwrap();
                    let elapsed = median(&mut elapsed);
                    let mut peaks = samples
                        .iter()
                        .map(|sample| sample.peak_bytes as u128)
                        .collect::<Vec<_>>();
                    let peak = median(&mut peaks);
                    println!(
                        "{},{},{count},{depth},{elapsed},{},{peak}",
                        kind.name(),
                        pattern.name(),
                        max - min,
                    );
                }
            }
        }
    }
}
