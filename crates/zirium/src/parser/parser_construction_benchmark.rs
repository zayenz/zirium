use super::*;
use crate::{
    lexer::{Token, lex},
    representation::SyntaxElement,
    source::{Source, TextRange},
};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    collections::{HashMap, HashSet},
    env,
    hint::black_box,
    mem::size_of,
    sync::{
        Arc,
        atomic::{AtomicIsize, Ordering},
    },
    time::{Duration, Instant},
};

struct CountingAllocator;
static LIVE: AtomicIsize = AtomicIsize::new(0);
static PEAK: AtomicIsize = AtomicIsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record(layout.size() as isize);
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
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
    let live = LIVE.fetch_add(delta, Ordering::Relaxed) + delta;
    PEAK.fetch_max(live, Ordering::Relaxed);
}

fn begin_measurement() -> (Instant, isize) {
    let baseline = LIVE.load(Ordering::Relaxed);
    PEAK.store(baseline, Ordering::Relaxed);
    (Instant::now(), baseline)
}

fn finish_measurement(started: Instant, baseline: isize) -> (Duration, usize) {
    (
        started.elapsed(),
        PEAK.load(Ordering::Relaxed).saturating_sub(baseline) as usize,
    )
}

#[derive(Clone, Copy)]
enum Shape {
    Primary,
    BlockRich,
}

impl Shape {
    fn from_env() -> Self {
        match env::var("ZIRIUM_PARSER_BENCH_SHAPE").as_deref() {
            Ok("block-rich") => Self::BlockRich,
            Ok("primary") | Err(_) => Self::Primary,
            Ok(value) => panic!("unknown ZIRIUM_PARSER_BENCH_SHAPE={value}"),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Primary => "primary",
            Self::BlockRich => "block-rich",
        }
    }
}

#[test]
#[ignore = "release-only parser construction measurement; see processing-benchmarks.md"]
fn measure_parser_construction_and_string_identity() {
    let smoke = env::var_os("ZIRIUM_PARSER_BENCH_SMOKE").is_some();
    let bytes = if smoke {
        64 * 1024
    } else {
        env::var("ZIRIUM_PARSER_BENCH_SIZE_MIB")
            .unwrap_or_else(|_| "10".into())
            .parse::<usize>()
            .expect("integer ZIRIUM_PARSER_BENCH_SIZE_MIB")
            * 1024
            * 1024
    };
    let runs = if smoke {
        1
    } else {
        env::var("ZIRIUM_PARSER_BENCH_RUNS")
            .unwrap_or_else(|_| "3".into())
            .parse::<usize>()
            .expect("integer ZIRIUM_PARSER_BENCH_RUNS")
    };
    let warmups = if smoke {
        1
    } else {
        env::var("ZIRIUM_PARSER_BENCH_WARMUPS")
            .unwrap_or_else(|_| "1".into())
            .parse::<usize>()
            .expect("integer ZIRIUM_PARSER_BENCH_WARMUPS")
    };
    assert!(runs > 0);
    let shape = Shape::from_env();
    let source_bytes = Arc::<[u8]>::from(fixture(shape, bytes));
    println!(
        "parser_construction fixture={} input_bytes={} warmups={} runs={} smoke={smoke}",
        shape.name(),
        bytes,
        warmups,
        runs
    );

    let source = Source::new(source_bytes.clone()).unwrap();
    let registry = DialectRegistry::proving();
    report_phase("lex", warmups, runs, || {
        let (started, baseline) = begin_measurement();
        let output = lex(&source);
        let result = finish_measurement(started, baseline);
        black_box(output.tokens().len());
        (result, output.tokens().len())
    });

    let lexed = lex(&source);
    report_phase("events", warmups, runs, || {
        let (started, baseline) = begin_measurement();
        let (builder, diagnostics) =
            produce_operation_events(&lexed, source.bytes(), registry, ParserLimits::default())
                .unwrap();
        let events = builder.into_events();
        let result = finish_measurement(started, baseline);
        black_box((&events, &diagnostics));
        (result, events.len())
    });

    let (builder, _) =
        produce_operation_events(&lexed, source.bytes(), registry, ParserLimits::default())
            .unwrap();
    let events = builder.into_events();
    report_phase("compact", warmups, runs, || {
        let (started, baseline) = begin_measurement();
        let tree =
            SyntaxTree::from_events_unverified(events.clone(), lexed.tokens().to_vec()).unwrap();
        let result = finish_measurement(started, baseline);
        black_box(&tree);
        (result, tree.node_count())
    });

    let tree = SyntaxTree::from_events_unverified(events.clone(), lexed.tokens().to_vec()).unwrap();
    report_string_identity(source.bytes(), &tree);
    drop(events);
    drop(lexed);
    report_phase("verify", warmups, runs, || {
        let (started, baseline) = begin_measurement();
        tree.verify().unwrap();
        let result = finish_measurement(started, baseline);
        black_box(&tree);
        (result, tree.node_count())
    });
}

fn report_phase(
    phase: &str,
    warmups: usize,
    runs: usize,
    mut measure: impl FnMut() -> ((Duration, usize), usize),
) {
    for _ in 0..warmups {
        black_box(measure());
    }
    let mut samples = (0..runs).map(|_| measure()).collect::<Vec<_>>();
    samples.sort_by_key(|sample| sample.0.0);
    let ((elapsed, peak), output_items) = samples[samples.len() / 2];
    println!(
        "parser_phase phase={phase} median_ns={} incremental_peak_live_bytes={peak} output_items={output_items}",
        elapsed.as_nanos()
    );
}

fn report_string_identity(source: &[u8], tree: &SyntaxTree) {
    let operation_names = tree
        .subtree(tree.root())
        .unwrap()
        .filter(|id| {
            matches!(
                tree.kind(*id),
                Some(SyntaxKind::Operation | SyntaxKind::DialectOperation)
            )
        })
        .filter_map(|id| {
            tree.elements(id)?
                .into_iter()
                .find_map(|element| match element {
                    SyntaxElement::Token { index, token } if token.kind() == TokenKind::String => {
                        Some(index)
                    }
                    _ => None,
                })
        })
        .collect::<HashSet<_>>();
    let classes = [
        (TokenKind::BareIdentifier, "bare"),
        (TokenKind::AtIdentifier, "at"),
        (TokenKind::PercentIdentifier, "percent"),
        (TokenKind::CaretIdentifier, "caret"),
        (TokenKind::ExclamationIdentifier, "exclamation"),
        (TokenKind::HashIdentifier, "hash"),
        (TokenKind::String, "operation-string"),
    ];
    let mut all_spellings = Vec::new();
    for (kind, name) in classes {
        let spellings = tree
            .tokens(tree.root())
            .unwrap()
            .iter()
            .enumerate()
            .filter(|(index, token)| {
                token.kind() == kind
                    && (kind != TokenKind::String || operation_names.contains(index))
            })
            .map(|(_, token)| token_text(source, token))
            .collect::<Vec<_>>();
        let unique = spellings.iter().copied().collect::<HashSet<_>>();
        let unique_bytes = unique.iter().map(|text| text.len()).sum::<usize>();
        let current = spellings.len() * size_of::<TextRange>();
        let candidate = current
            + spellings.len() * size_of::<u32>()
            + unique.len() * (size_of::<u32>() + size_of::<u64>() + 2 * size_of::<usize>())
            + unique_bytes;
        println!(
            "string_identity class={name} frequency={} unique={} current_range_record_bytes={current} candidate_range_plus_id_interner_bytes={candidate} unique_spelling_bytes={unique_bytes}",
            spellings.len(),
            unique.len()
        );
        all_spellings.extend(spellings);
    }
    let mut interner = HashMap::<&[u8], u32>::new();
    for spelling in &all_spellings {
        let next = interner.len() as u32;
        interner.entry(spelling).or_insert(next);
    }
    let started = Instant::now();
    let checksum = all_spellings
        .iter()
        .map(|spelling| *interner.get(spelling).unwrap() as u64)
        .sum::<u64>();
    println!(
        "string_identity_lookup lookups={} elapsed_ns={} checksum={}",
        all_spellings.len(),
        started.elapsed().as_nanos(),
        black_box(checksum)
    );
}

fn token_text<'a>(source: &'a [u8], token: &Token) -> &'a [u8] {
    &source[token.range().start() as usize..token.range().end() as usize]
}

fn fixture(shape: Shape, bytes: usize) -> Vec<u8> {
    match shape {
        Shape::Primary => primary_fixture(bytes),
        Shape::BlockRich => block_rich_fixture(bytes),
    }
}

fn primary_fixture(bytes: usize) -> Vec<u8> {
    let prefix = b"\"bench.container\"() ({\n^bb:\n%seed = \"bench.source\"() : () -> i32\n";
    let suffix = b"}) : () -> ()\n";
    let line = b"\"bench.op\"() {tag = \"zirium\"} : () -> () // deterministic parser benchmark padding xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n";
    exact_fixture(bytes, prefix, line, suffix)
}

fn block_rich_fixture(bytes: usize) -> Vec<u8> {
    let mut output = b"builtin.module {\n".to_vec();
    let suffix = b"}\n";
    let mut index = 0;
    loop {
        let function = format!(
            "func.func @f{index}() {{\n^entry:\n%value = arith.constant 1 : i32\ncf.br ^middle\n^middle:\n\"bench.use\"(%value) : (i32) -> ()\ncf.br ^exit\n^exit:\nfunc.return\n}}\n"
        );
        if output.len() + function.len() + suffix.len() > bytes {
            break;
        }
        output.extend_from_slice(function.as_bytes());
        index += 1;
    }
    output.resize(bytes - suffix.len(), b' ');
    output.extend_from_slice(suffix);
    output
}

fn exact_fixture(bytes: usize, prefix: &[u8], line: &[u8], suffix: &[u8]) -> Vec<u8> {
    assert!(bytes >= prefix.len() + suffix.len());
    let mut output = Vec::with_capacity(bytes);
    output.extend_from_slice(prefix);
    while output.len() + line.len() + suffix.len() <= bytes {
        output.extend_from_slice(line);
    }
    output.resize(bytes - suffix.len(), b' ');
    output.extend_from_slice(suffix);
    output
}
