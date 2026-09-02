use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::process::Command;
use std::sync::atomic::{AtomicIsize, Ordering};
use std::time::{Duration, Instant};
use zirium::lexer::{Token, TokenKind};
use zirium::source::TextRange;
use zirium::{EventBuilder, SyntaxKind};

const SEED: u64 = 0x5a49_5249_554d_0001;
const RUNS: usize = 10;

struct CountingAllocator;
static LIVE: AtomicIsize = AtomicIsize::new(0);
static PEAK: AtomicIsize = AtomicIsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record_allocation(layout.size() as isize);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, old: Layout, new_size: usize) -> *mut u8 {
        let new_pointer = unsafe { System.realloc(pointer, old, new_size) };
        if !new_pointer.is_null() {
            record_allocation(new_size as isize - old.size() as isize);
        }
        new_pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn record_allocation(delta: isize) {
    let live = LIVE.fetch_add(delta, Ordering::Relaxed) + delta;
    PEAK.fetch_max(live, Ordering::Relaxed);
}

fn reset_peak() -> isize {
    let baseline = LIVE.load(Ordering::Relaxed);
    PEAK.store(baseline, Ordering::Relaxed);
    baseline
}

#[derive(Clone, Copy)]
enum Generator {
    Dense(usize),
    Trivia(usize),
    Nested(usize),
}

impl Generator {
    fn name(self) -> String {
        match self {
            Self::Dense(bytes) => format!("dense-{bytes}"),
            Self::Trivia(bytes) => format!("trivia-{bytes}"),
            Self::Nested(depth) => format!("nested-depth-{depth}"),
        }
    }

    fn events(self) -> Generated {
        match self {
            Self::Dense(bytes) => flat_events(bytes, false),
            Self::Trivia(bytes) => flat_events(bytes, true),
            Self::Nested(depth) => nested_events(depth),
        }
    }
}

struct Generated {
    builder: EventBuilder,
    tokens: Vec<Token>,
}

struct Sample {
    construction: Duration,
    compaction: Duration,
    traversal: Duration,
    parents: Duration,
    peak_bytes: usize,
    retained_bytes: usize,
    event_count: usize,
    node_count: usize,
    token_count: usize,
}

fn main() {
    print_environment();
    println!("seed=0x{SEED:016x} warmups=1 measured_runs={RUNS}");
    let cases = [
        Generator::Dense(4 * 1024),
        Generator::Dense(256 * 1024),
        Generator::Dense(4 * 1024 * 1024),
        Generator::Trivia(4 * 1024),
        Generator::Trivia(256 * 1024),
        Generator::Trivia(4 * 1024 * 1024),
        Generator::Nested(8),
        Generator::Nested(64),
        Generator::Nested(256),
    ];

    for generator in cases {
        let _ = measure(generator);
        let mut samples: Vec<_> = (0..RUNS).map(|_| measure(generator)).collect();
        samples.sort_by_key(|sample| sample.construction);
        let construction = samples[RUNS / 2].construction;
        samples.sort_by_key(|sample| sample.compaction);
        let compaction = samples[RUNS / 2].compaction;
        samples.sort_by_key(|sample| sample.traversal);
        let traversal = samples[RUNS / 2].traversal;
        samples.sort_by_key(|sample| sample.parents);
        let parents = samples[RUNS / 2].parents;
        samples.sort_by_key(|sample| sample.peak_bytes);
        let representative = &samples[RUNS / 2];
        println!(
            "case={} events={} nodes={} tokens={} construction_ns={} compaction_ns={} traversal_ns={} parent_index_ns={} peak_live_bytes={} exact_retained_bytes={}",
            generator.name(),
            representative.event_count,
            representative.node_count,
            representative.token_count,
            construction.as_nanos(),
            compaction.as_nanos(),
            traversal.as_nanos(),
            parents.as_nanos(),
            representative.peak_bytes,
            representative.retained_bytes
        );
    }
}

fn measure(generator: Generator) -> Sample {
    let baseline = reset_peak();
    let start = Instant::now();
    let generated = generator.events();
    let construction = start.elapsed();
    let event_count = generated.builder.events().len();

    let start = Instant::now();
    let tree = generated
        .builder
        .finish(generated.tokens)
        .expect("generator produces valid events");
    let compaction = start.elapsed();

    let start = Instant::now();
    let mut checksum = 0_u64;
    let mut visited = 0_usize;
    for node in tree.subtree(tree.root()).unwrap() {
        visited += 1;
        checksum = checksum
            .wrapping_mul(0x100_0000_01b3)
            .wrapping_add(tree.kind(node).unwrap() as u64)
            .wrapping_add(tree.has_error(node).unwrap() as u64)
            .wrapping_add(tree.token_indices(node).unwrap().len() as u64);
    }
    black_box((visited, checksum));
    let traversal = start.elapsed();

    let start = Instant::now();
    tree.build_parent_index();
    let parents = start.elapsed();
    let peak_bytes = PEAK.load(Ordering::Relaxed).saturating_sub(baseline) as usize;

    Sample {
        construction,
        compaction,
        traversal,
        parents,
        peak_bytes,
        retained_bytes: tree.exact_retained_bytes(),
        event_count,
        node_count: tree.node_count(),
        token_count: tree.token_count(),
    }
}

fn flat_events(target_bytes: usize, trivia_heavy: bool) -> Generated {
    let operation_bytes = if trivia_heavy { 64 } else { 32 };
    let operations = target_bytes.div_ceil(operation_bytes);
    let mut builder = EventBuilder::new();
    let mut tokens = Vec::with_capacity(operations * 2);
    let mut offset = 0_u32;
    let file = builder.start();
    for index in 0..operations {
        let operation = builder.start();
        let word = 12 + ((index as u64 ^ SEED) & 7) as u32;
        push_token(
            &mut builder,
            &mut tokens,
            TokenKind::BareIdentifier,
            offset,
            offset + word,
        );
        offset += word;
        let remaining = operation_bytes as u32 - word;
        let kind = if trivia_heavy && index % 2 == 0 {
            TokenKind::LineComment
        } else if trivia_heavy {
            TokenKind::Whitespace
        } else {
            TokenKind::Colon
        };
        push_token(&mut builder, &mut tokens, kind, offset, offset + remaining);
        offset += remaining;
        builder.complete(operation, SyntaxKind::Operation).unwrap();
    }
    builder.complete(file, SyntaxKind::File).unwrap();
    Generated { builder, tokens }
}

fn nested_events(depth: usize) -> Generated {
    let mut builder = EventBuilder::new();
    let mut tokens = Vec::with_capacity(depth);
    let file = builder.start();
    let mut regions = Vec::with_capacity(depth);
    for offset in 0..depth {
        let offset = offset as u32;
        regions.push(builder.start());
        push_token(
            &mut builder,
            &mut tokens,
            TokenKind::Colon,
            offset,
            offset + 1,
        );
    }
    for region in regions.into_iter().rev() {
        builder.complete(region, SyntaxKind::Region).unwrap();
    }
    builder.complete(file, SyntaxKind::File).unwrap();
    Generated { builder, tokens }
}

fn push_token(
    builder: &mut EventBuilder,
    tokens: &mut Vec<Token>,
    kind: TokenKind,
    start: u32,
    end: u32,
) {
    let index = tokens.len();
    tokens.push(Token::new(kind, TextRange::new(start, end).unwrap()));
    builder.token(index).unwrap();
}

fn print_environment() {
    let rustc = command_output("rustc", &["-Vv"]);
    for line in rustc.lines() {
        println!("rustc_{line}");
    }
    println!("target={}", target_from_rustc(&rustc));
    println!("profile=release");
    println!("os={}", command_output("uname", &["-srv"]));
    println!("cpu={}", cpu_name());
}

fn target_from_rustc(output: &str) -> &str {
    output
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .unwrap_or("unknown")
}

fn cpu_name() -> String {
    if cfg!(target_os = "macos") {
        command_output("sysctl", &["-n", "machdep.cpu.brand_string"])
    } else {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|contents| {
                contents
                    .lines()
                    .find_map(|line| line.strip_prefix("model name\t: ").map(str::to_owned))
            })
            .unwrap_or_else(|| "unknown".to_owned())
    }
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
