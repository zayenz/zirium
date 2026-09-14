use std::{
    alloc::{GlobalAlloc, Layout, System},
    hint::black_box,
    process::Command,
    sync::atomic::{AtomicIsize, Ordering},
    time::Instant,
};

use zirium::{
    dialect::DialectRegistry,
    parser::ParsedFile,
    semantic::{
        AttributeSpec, DocumentStatistics, LoweringMode, TypeSpec, TypeValue,
        lower_with_dialect_registry,
    },
};

struct CountingAllocator;
static LIVE_BYTES: AtomicIsize = AtomicIsize::new(0);
static PEAK_BYTES: AtomicIsize = AtomicIsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record(layout.size() as isize);
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
            record(size as isize - old.size() as isize);
        }
        result
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn record(delta: isize) {
    let live = LIVE_BYTES.fetch_add(delta, Ordering::Relaxed) + delta;
    PEAK_BYTES.fetch_max(live, Ordering::Relaxed);
}

#[derive(Clone, Copy)]
struct Measurement {
    edit_ns: u128,
    verify_ns: u128,
    peak_allocated_bytes: usize,
    baseline_owned_bytes: usize,
    stats: DocumentStatistics,
    reclaimed_list_entries: usize,
}

fn main() {
    let (iterations, warmups, runs) = arguments();
    println!(
        "environment rustc={} target={} os={} profile={}",
        command_output("rustc", &["--version"]),
        rustc_host(),
        command_output("uname", &["-sr"]),
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
    );
    for _ in 0..warmups {
        black_box(measure(iterations));
    }
    let measurements = (0..runs).map(|_| measure(iterations)).collect::<Vec<_>>();
    let representative = measurements[runs / 2];
    println!(
        "workload iterations={iterations} warmups={warmups} runs={runs} live_strings={} retained_strings={} live_types={} retained_types={} live_attributes={} retained_attributes={} baseline_owned_bytes={} retained_owned_bytes={} pooled_entries={} reclaimed_list_entries={}",
        representative.stats.live_strings,
        representative.stats.retained_strings,
        representative.stats.live_types,
        representative.stats.retained_types,
        representative.stats.live_attributes,
        representative.stats.retained_attributes,
        representative.baseline_owned_bytes,
        representative.stats.direct_owned_bytes,
        representative.stats.pooled_list_entries,
        representative.reclaimed_list_entries,
    );
    print_summary("edit_ns", measurements.iter().map(|run| run.edit_ns));
    print_summary("verify_ns", measurements.iter().map(|run| run.verify_ns));
    print_summary(
        "peak_allocated_bytes",
        measurements
            .iter()
            .map(|run| run.peak_allocated_bytes as u128),
    );
}

fn arguments() -> (u32, usize, usize) {
    let (mut iterations, mut warmups, mut runs) = (10_000, 1, 3);
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        let value = args
            .next()
            .unwrap_or_else(|| panic!("{argument} requires a value"));
        match argument.as_str() {
            "--iterations" => iterations = value.parse().expect("iterations must be an integer"),
            "--warmups" => warmups = value.parse().expect("warmups must be an integer"),
            "--runs" => runs = value.parse().expect("runs must be an integer"),
            _ => panic!("unknown argument: {argument}"),
        }
    }
    assert!(iterations > 0, "iterations must be positive");
    assert!(runs > 0, "runs must be positive");
    (iterations, warmups, runs)
}

fn measure(iterations: u32) -> Measurement {
    let registry = DialectRegistry::EMPTY;
    let parsed = ParsedFile::parse(br#"%0 = "anchor"() : () -> i32"#.as_slice()).unwrap();
    let mut document = lower_with_dialect_registry(&parsed, LoweringMode::Strict, &registry)
        .document
        .unwrap();
    let operation = document.root_operations()[0];
    let baseline = document.statistics();
    let allocator_baseline = LIVE_BYTES.load(Ordering::Relaxed);
    PEAK_BYTES.store(allocator_baseline, Ordering::Relaxed);
    let edit_start = Instant::now();
    let mut editor = document.edit(&registry).unwrap();
    for index in 0..iterations {
        let width = 33 + index;
        editor
            .replace_result_types(
                operation,
                &[TypeSpec {
                    spelling: format!("i{width}"),
                    value: TypeValue::Integer {
                        width,
                        signedness: None,
                    },
                }],
            )
            .unwrap();
        let name = format!("temporary_{index}");
        editor
            .set_attribute(
                operation,
                AttributeSpec::string(name.clone(), format!("value_{index}")),
            )
            .unwrap();
        editor.remove_attribute(operation, &name).unwrap();
    }
    editor
        .replace_result_types(
            operation,
            &[TypeSpec {
                spelling: "i32".into(),
                value: TypeValue::Integer {
                    width: 32,
                    signedness: None,
                },
            }],
        )
        .unwrap();
    let pooled_before = editor.document().statistics().pooled_list_entries;
    let reclaimed_list_entries = editor.compact_pools();
    editor.commit().unwrap();
    let edit_ns = edit_start.elapsed().as_nanos();
    let verify_start = Instant::now();
    black_box(document.verify_semantics(&registry)).unwrap();
    let verify_ns = verify_start.elapsed().as_nanos();
    let stats = document.statistics();
    assert_eq!(stats.live_attributes, 0);
    assert!(stats.retained_types > stats.live_types);
    assert!(stats.retained_attributes > stats.live_attributes);
    assert!(reclaimed_list_entries > 0 && stats.pooled_list_entries < pooled_before);
    Measurement {
        edit_ns,
        verify_ns,
        peak_allocated_bytes: PEAK_BYTES
            .load(Ordering::Relaxed)
            .saturating_sub(allocator_baseline) as usize,
        baseline_owned_bytes: baseline.direct_owned_bytes,
        stats,
        reclaimed_list_entries,
    }
}

fn print_summary(name: &str, values: impl Iterator<Item = u128>) {
    let mut values = values.collect::<Vec<_>>();
    values.sort_unstable();
    println!(
        "summary metric={name} min={} median={} max={} spread={}",
        values[0],
        values[values.len() / 2],
        values[values.len() - 1],
        values[values.len() - 1] - values[0]
    );
}

fn command_output(command: &str, args: &[&str]) -> String {
    Command::new(command)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .unwrap_or_else(|| "unavailable".into())
}

fn rustc_host() -> String {
    command_output("rustc", &["-vV"])
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .unwrap_or("unavailable")
        .to_owned()
}
