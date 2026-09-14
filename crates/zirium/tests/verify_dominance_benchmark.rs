use std::alloc::{GlobalAlloc, Layout, System};
use std::env;
use std::hint::black_box;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicIsize, Ordering};
use std::time::Instant;

use zirium::dialect::DialectRegistry;
use zirium::parser::ParsedFile;
use zirium::semantic::{Document, LoweringMode, lower_with_dialect_registry};

struct CountingAllocator;
static LIVE_BYTES: AtomicIsize = AtomicIsize::new(0);
static PEAK_BYTES: AtomicIsize = AtomicIsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record_allocation(layout.size() as isize);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, old: Layout, size: usize) -> *mut u8 {
        let result = unsafe { System.realloc(pointer, old, size) };
        if !result.is_null() {
            record_allocation(size as isize - old.size() as isize);
        }
        result
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn record_allocation(delta: isize) {
    let live = LIVE_BYTES.fetch_add(delta, Ordering::Relaxed) + delta;
    PEAK_BYTES.fetch_max(live, Ordering::Relaxed);
}

#[derive(Clone, Copy)]
enum Shape {
    Chain,
    Diamond,
    Loop,
    Unreachable,
}

impl Shape {
    const ALL: [Self; 4] = [Self::Chain, Self::Diamond, Self::Loop, Self::Unreachable];

    fn name(self) -> &'static str {
        match self {
            Self::Chain => "chain",
            Self::Diamond => "diamond",
            Self::Loop => "loop",
            Self::Unreachable => "unreachable",
        }
    }
}

#[derive(Clone, Copy)]
struct Sample {
    elapsed_ns: u128,
    peak_bytes: usize,
}

#[test]
#[ignore = "release-only dominance scaling benchmark"]
fn measure_verifier_dominance_scaling() {
    require_release();
    let smoke = env::var_os("ZIRIUM_DOMINANCE_BENCH_SMOKE").is_some();
    let dimensions = if smoke {
        vec![16, 32]
    } else {
        parse_dimensions("ZIRIUM_DOMINANCE_BENCH_BLOCKS", &[64, 128, 256, 512])
    };
    let warmups = parse_usize("ZIRIUM_DOMINANCE_BENCH_WARMUPS", if smoke { 0 } else { 1 });
    let runs = parse_usize("ZIRIUM_DOMINANCE_BENCH_RUNS", if smoke { 1 } else { 3 });
    assert!(runs > 0, "benchmark needs at least one measured run");

    let rustc = command_output("rustc", &["-Vv"])
        .lines()
        .find(|line| line.starts_with("rustc "))
        .unwrap_or("unknown")
        .to_owned();
    let host = command_output("rustc", &["-Vv"])
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .unwrap_or("unknown")
        .to_owned();
    let os = command_output("uname", &["-sr"]);
    println!(
        "environment profile=release rustc={rustc:?} host={host} os={os:?} warmups={warmups} measured_runs={runs} dimensions={dimensions:?} boundary=semantic_verifier_only_above_lowered_document"
    );
    println!(
        "shape,blocks,verify_ns_min,verify_ns_median,verify_ns_max,verify_ns_spread,peak_bytes_min,peak_bytes_median,peak_bytes_max,peak_bytes_spread"
    );

    for shape in Shape::ALL {
        let mut endpoint_peaks = Vec::new();
        for &blocks in &dimensions {
            for _ in 0..warmups {
                let _ = measure(shape, blocks);
            }
            let mut samples = (0..runs)
                .map(|_| measure(shape, blocks))
                .collect::<Vec<_>>();
            samples.sort_by_key(|sample| sample.elapsed_ns);
            let elapsed_min = samples[0].elapsed_ns;
            let elapsed_median = samples[samples.len() / 2].elapsed_ns;
            let elapsed_max = samples[samples.len() - 1].elapsed_ns;
            samples.sort_by_key(|sample| sample.peak_bytes);
            let peak_min = samples[0].peak_bytes;
            let peak_median = samples[samples.len() / 2].peak_bytes;
            let peak_max = samples[samples.len() - 1].peak_bytes;
            endpoint_peaks.push((blocks, peak_median));
            println!(
                "{},{blocks},{elapsed_min},{elapsed_median},{elapsed_max},{},{peak_min},{peak_median},{peak_max},{}",
                shape.name(),
                elapsed_max - elapsed_min,
                peak_max - peak_min
            );
        }
        if let (Some(&(first_blocks, first_peak)), Some(&(last_blocks, last_peak))) =
            (endpoint_peaks.first(), endpoint_peaks.last())
        {
            println!(
                "growth shape={} block_ratio={:.3} peak_ratio={:.3} first_blocks={first_blocks} last_blocks={last_blocks} first_peak_bytes={first_peak} last_peak_bytes={last_peak}",
                shape.name(),
                last_blocks as f64 / first_blocks as f64,
                last_peak as f64 / first_peak as f64
            );
        }
    }
}

fn measure(shape: Shape, blocks: usize) -> Sample {
    let document = lower(&fixture(shape, blocks));
    document.validate_structure().unwrap();
    let function = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("func.func"))
        .unwrap();
    let region = document.operation_regions(function).unwrap()[0];
    assert_eq!(
        document
            .region(region)
            .unwrap()
            .blocks(&document)
            .unwrap()
            .len(),
        blocks
    );
    assert_eq!(document.statistics().dominance_index_entries, 0);

    let baseline = LIVE_BYTES.load(Ordering::Relaxed);
    PEAK_BYTES.store(baseline, Ordering::Relaxed);
    let start = Instant::now();
    black_box(&document)
        .verify_semantics(DialectRegistry::baseline())
        .unwrap();
    let elapsed_ns = start.elapsed().as_nanos();
    let peak_bytes = PEAK_BYTES.load(Ordering::Relaxed).saturating_sub(baseline) as usize;

    // Verification uses a fresh local analysis; only the public query may
    // populate the revision- and registry-bound cache.
    assert_eq!(document.statistics().dominance_index_entries, 0);
    let definition = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("arith.constant"))
        .unwrap();
    let value = document
        .operation(definition)
        .unwrap()
        .result(definition, 0)
        .unwrap();
    let use_operation = document
        .operations()
        .filter(|operation| document.operation_name(*operation) == Some("use"))
        .last()
        .unwrap();
    assert!(document.dominates(value, use_operation, DialectRegistry::baseline()));
    assert!(document.statistics().dominance_index_entries > 0);

    Sample {
        elapsed_ns,
        peak_bytes,
    }
}

fn require_release() {
    #[cfg(debug_assertions)]
    panic!("run this ignored benchmark with --release");
}

fn lower(source: &str) -> Document {
    let parsed = ParsedFile::parse_with_registry(
        Arc::<[u8]>::from(source.as_bytes()),
        DialectRegistry::baseline(),
    )
    .unwrap();
    lower_with_dialect_registry(
        &parsed,
        LoweringMode::BestEffort,
        DialectRegistry::baseline(),
    )
    .document
    .unwrap()
}

fn fixture(shape: Shape, requested_blocks: usize) -> String {
    let mut source = String::from(
        "builtin.module {\n  func.func @dominance() {\n  ^entry:\n    %value = arith.constant 1 : i32\n    %condition = arith.constant 1 : i1\n",
    );
    match shape {
        Shape::Chain => {
            source.push_str("    cf.br ^b0\n");
            for block in 0..requested_blocks.saturating_sub(2) {
                source.push_str(&format!(
                    "  ^b{block}:\n    \"use\"(%value) : (i32) -> ()\n    cf.br ^b{}\n",
                    block + 1
                ));
            }
            let last = requested_blocks.saturating_sub(2);
            source.push_str(&format!(
                "  ^b{last}:\n    \"use\"(%value) : (i32) -> ()\n    func.return\n"
            ));
        }
        Shape::Diamond => {
            let diamonds = requested_blocks.saturating_sub(1) / 3;
            let tail_blocks = requested_blocks.saturating_sub(1 + diamonds * 3);
            source.push_str("    cf.cond_br %condition, ^left0, ^right0\n");
            for diamond in 0..diamonds {
                source.push_str(&format!(
                    "  ^left{diamond}:\n    \"use\"(%value) : (i32) -> ()\n    cf.br ^merge{diamond}\n  ^right{diamond}:\n    \"use\"(%value) : (i32) -> ()\n    cf.br ^merge{diamond}\n  ^merge{diamond}:\n    \"use\"(%value) : (i32) -> ()\n"
                ));
                if diamond + 1 == diamonds {
                    if tail_blocks == 0 {
                        source.push_str("    func.return\n");
                    } else {
                        source.push_str("    cf.br ^tail0\n");
                    }
                } else {
                    source.push_str(&format!(
                        "    cf.cond_br %condition, ^left{}, ^right{}\n",
                        diamond + 1,
                        diamond + 1
                    ));
                }
            }
            for tail in 0..tail_blocks {
                source.push_str(&format!("  ^tail{tail}:\n"));
                if tail + 1 == tail_blocks {
                    source.push_str("    func.return\n");
                } else {
                    source.push_str(&format!("    cf.br ^tail{}\n", tail + 1));
                }
            }
        }
        Shape::Loop => {
            let loop_blocks = requested_blocks.saturating_sub(2);
            source.push_str("    cf.br ^loop0\n");
            for block in 0..loop_blocks.saturating_sub(1) {
                source.push_str(&format!(
                    "  ^loop{block}:\n    \"use\"(%value) : (i32) -> ()\n    cf.br ^loop{}\n",
                    block + 1
                ));
            }
            let last = loop_blocks.saturating_sub(1);
            source.push_str(&format!(
                "  ^loop{last}:\n    \"use\"(%value) : (i32) -> ()\n    cf.cond_br %condition, ^loop0, ^exit\n  ^exit:\n    func.return\n"
            ));
        }
        Shape::Unreachable => {
            source.push_str("    cf.br ^exit\n");
            for block in 0..requested_blocks.saturating_sub(2) {
                let successor = if block + 1 == requested_blocks.saturating_sub(2) {
                    0
                } else {
                    block + 1
                };
                source.push_str(&format!(
                    "  ^dead{block}:\n    \"use\"(%value) : (i32) -> ()\n    cf.br ^dead{successor}\n"
                ));
            }
            source.push_str("  ^exit:\n    func.return\n");
        }
    }
    source.push_str("  }\n}\n");
    source
}

fn parse_dimensions(name: &str, default: &[usize]) -> Vec<usize> {
    let Some(value) = env::var_os(name) else {
        return default.to_vec();
    };
    let dimensions = value
        .to_string_lossy()
        .split(',')
        .map(|part| {
            part.trim()
                .parse()
                .expect("block dimensions must be integers")
        })
        .collect::<Vec<_>>();
    assert!(
        dimensions.iter().all(|blocks| *blocks >= 4),
        "block dimensions must be at least four"
    );
    dimensions
}

fn parse_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .map(|value| value.parse().expect("benchmark option must be an integer"))
        .unwrap_or(default)
}

fn command_output(program: &str, arguments: &[&str]) -> String {
    Command::new(program)
        .args(arguments)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .unwrap_or_else(|| "unknown".to_owned())
}
