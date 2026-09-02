use std::alloc::{GlobalAlloc, Layout, System};
use std::env;
use std::fs::{self, File};
use std::hint::black_box;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicIsize, Ordering};
use std::time::Instant;

use zirium::dialect::DialectRegistry;
use zirium::parser::{ParseLimits, ParsedFile};
use zirium::printer::PrintLayout;
use zirium::semantic::{
    AttributeSpec, AttributeValue, LoweringMode, RetentionProfile,
    lower_with_dialect_registry_and_retention,
};

const MIB: usize = 1024 * 1024;
const SEED: u64 = 0x5a49_5249_554d_0028;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    Primary,
    BlockRich,
    Trivia,
    Nested,
    Payload,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Parse,
    Traverse,
    SyntaxPayload,
    SyntaxOperationPayload,
    SyntaxOperationTraverse,
    OperationPayload,
    OperationTraverse,
    Lower,
    Verify,
    Canonical,
    UseIndex,
    SymbolIndex,
    DominanceIndex,
    Editor,
    Preserving,
    DirtyPreserving,
}

struct Args {
    bytes: usize,
    shape: Shape,
    depth: Option<usize>,
    stages: Vec<Stage>,
    warmups: usize,
    runs: usize,
    smoke: bool,
}

#[derive(Clone)]
struct Measurement {
    ns: u128,
    peak: usize,
    output: usize,
    checksum: u64,
    stats: String,
}

#[derive(Default)]
struct ByteCounter(usize);

impl Write for ByteCounter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0 += bytes.len();
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn main() {
    if env::args().nth(1).as_deref() == Some("--report") {
        report();
        return;
    }
    let args = parse_args();
    print_environment(&args);
    let path = temp_path();
    let source = generate(args.shape, args.bytes, args.depth);
    assert_eq!(source.len(), args.bytes);
    fs::write(&path, &source).expect("write temporary fixture");
    drop(source);
    println!(
        "fixture_storage=temporary generated_bytes={} seed=0x{SEED:016x}",
        args.bytes
    );
    for stage in args.stages {
        for _ in 0..args.warmups {
            black_box(measure(stage, args.shape, args.depth, &path));
        }
        let mut samples = (0..args.runs)
            .map(|_| measure(stage, args.shape, args.depth, &path))
            .collect::<Vec<_>>();
        samples.sort_by_key(|sample| sample.ns);
        let sample = samples[samples.len() / 2].clone();
        let mib_s = args.bytes as f64 / MIB as f64 / (sample.ns as f64 / 1e9);
        println!(
            "measurement stage={} shape={} depth={} input_bytes={} median_ns={} throughput_mib_s={mib_s:.3} peak_live_bytes={} output_bytes={} checksum={} {}",
            stage.name(),
            args.shape.name(),
            args.depth
                .map_or_else(|| "none".into(), |depth| depth.to_string()),
            args.bytes,
            sample.ns,
            sample.peak,
            sample.output,
            sample.checksum,
            sample.stats
        );
    }
    fs::remove_file(path).expect("remove temporary fixture");
}

fn parse_args() -> Args {
    let mut bytes = 10 * MIB;
    let mut shape = Shape::Primary;
    let mut depth = None;
    let mut stages = vec![Stage::Parse];
    let mut warmups = 1;
    let mut runs = 3;
    let mut smoke = false;
    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--smoke" => {
                smoke = true;
                bytes = 64 * 1024;
                warmups = 1;
                runs = 1;
            }
            "--size-mib" => {
                bytes = it
                    .next()
                    .expect("size")
                    .parse::<usize>()
                    .expect("integer size")
                    * MIB
            }
            "--size-kib" => {
                bytes = it
                    .next()
                    .expect("size")
                    .parse::<usize>()
                    .expect("integer size")
                    * 1024
            }
            "--shape" => shape = Shape::parse(&it.next().expect("shape")),
            "--depth" => depth = Some(it.next().expect("depth").parse().expect("integer depth")),
            "--stage" => {
                stages = it
                    .next()
                    .expect("stage")
                    .split(',')
                    .map(Stage::parse)
                    .collect()
            }
            "--runs" => runs = it.next().expect("runs").parse().expect("integer runs"),
            "--warmups" => {
                warmups = it
                    .next()
                    .expect("warmups")
                    .parse()
                    .expect("integer warmups")
            }
            _ => panic!("unknown argument {arg}"),
        }
    }
    assert!(runs > 0);
    assert_eq!(
        shape == Shape::Nested,
        depth.is_some(),
        "--depth is required exactly when --shape nested is selected"
    );
    assert!(depth != Some(0), "depth must be positive");
    Args {
        bytes,
        shape,
        depth,
        stages,
        warmups,
        runs,
        smoke,
    }
}

fn measure(stage: Stage, shape: Shape, depth: Option<usize>, path: &PathBuf) -> Measurement {
    let registry = DialectRegistry::proving();
    let mut limits = ParseLimits::default();
    if let Some(depth) = depth {
        limits.max_delimiter_depth = limits.max_delimiter_depth.max(depth.saturating_add(8));
    }
    let mut bytes = Vec::new();
    File::open(path).unwrap().read_to_end(&mut bytes).unwrap();
    let mut parsed = if stage == Stage::Parse {
        None
    } else {
        Some(
            ParsedFile::parse_with_limits_and_registry(
                Arc::<[u8]>::from(bytes.clone()),
                limits,
                registry,
            )
            .unwrap(),
        )
    };
    let retention = if shape == Shape::Primary
        && matches!(
            stage,
            Stage::Lower | Stage::Preserving | Stage::DirtyPreserving
        ) {
        RetentionProfile::Hybrid
    } else {
        RetentionProfile::SemanticOnly
    };
    let mut document = if matches!(
        stage,
        Stage::Lower
            | Stage::Parse
            | Stage::Traverse
            | Stage::SyntaxPayload
            | Stage::SyntaxOperationPayload
            | Stage::SyntaxOperationTraverse
    ) {
        None
    } else {
        Some(
            lower_with_dialect_registry_and_retention(
                parsed.as_ref().unwrap(),
                LoweringMode::Strict,
                retention,
                registry,
            )
            .document
            .expect("strict lowering"),
        )
    };
    if stage == Stage::DirtyPreserving {
        let document = document.as_mut().unwrap();
        let operations = document
            .operations()
            .filter(|operation| {
                document
                    .operation_regions(*operation)
                    .unwrap_or(&[])
                    .is_empty()
            })
            .collect::<Vec<_>>();
        let mut editor = document.edit(registry).unwrap();
        for operation in operations {
            editor
                .set_attribute(
                    operation,
                    AttributeSpec {
                        name: "bench_dirty".into(),
                        spelling: "\"yes\"".into(),
                        value: AttributeValue::String("\"yes\"".into()),
                    },
                )
                .unwrap();
        }
        editor.commit().unwrap();
    }
    let baseline = LIVE.load(Ordering::Relaxed);
    PEAK.store(baseline, Ordering::Relaxed);
    let started = Instant::now();
    let mut output = 0;
    let mut checksum = 0_u64;
    match stage {
        Stage::Parse => {
            parsed = Some(
                ParsedFile::parse_with_limits_and_registry(
                    Arc::<[u8]>::from(bytes),
                    limits,
                    registry,
                )
                .unwrap(),
            )
        }
        Stage::Traverse => {
            for node in parsed
                .as_ref()
                .unwrap()
                .syntax()
                .tree()
                .subtree(parsed.as_ref().unwrap().syntax().tree().root())
                .unwrap()
            {
                let tree = parsed.as_ref().unwrap().syntax().tree();
                checksum = checksum
                    .wrapping_mul(0x100000001b3)
                    .wrapping_add(tree.kind(node).unwrap() as u64)
                    .wrapping_add(tree.token_indices(node).unwrap().len() as u64);
            }
        }
        Stage::SyntaxPayload => {
            let tree = parsed.as_ref().unwrap().syntax().tree();
            let mut node_kind = Vec::with_capacity(tree.node_count() * 2);
            let mut node_start = Vec::with_capacity(tree.node_count() * 4);
            let mut node_end = Vec::with_capacity(tree.node_count() * 4);
            let mut node_subtree_end = Vec::with_capacity(tree.node_count() * 4);
            let mut node_flags = Vec::with_capacity(tree.node_count());
            for index in 0..tree.node_count() {
                let id = tree.node(index).unwrap();
                node_kind.extend_from_slice(&(tree.kind(id).unwrap() as u16).to_ne_bytes());
                node_start.extend_from_slice(
                    &tree
                        .text_range(id)
                        .map_or(u32::MAX, |r| r.start())
                        .to_ne_bytes(),
                );
                node_end.extend_from_slice(
                    &tree
                        .text_range(id)
                        .map_or(u32::MAX, |r| r.end())
                        .to_ne_bytes(),
                );
                node_subtree_end
                    .extend_from_slice(&(tree.subtree_end(id).unwrap() as u32).to_ne_bytes());
                node_flags.push(u8::from(tree.has_error(id).unwrap()));
            }
            let mut token_kind = Vec::with_capacity(tree.token_count() * 2);
            let mut token_start = Vec::with_capacity(tree.token_count() * 4);
            let mut token_end = Vec::with_capacity(tree.token_count() * 4);
            for index in 0..tree.token_count() {
                let token = tree.token(index).unwrap();
                token_kind.extend_from_slice(&(token.kind() as u16).to_ne_bytes());
                token_start.extend_from_slice(&token.range().start().to_ne_bytes());
                token_end.extend_from_slice(&token.range().end().to_ne_bytes());
            }
            output = node_kind.len()
                + node_start.len()
                + node_end.len()
                + node_subtree_end.len()
                + node_flags.len()
                + token_kind.len()
                + token_start.len()
                + token_end.len();
            checksum = node_kind
                .iter()
                .chain(&token_kind)
                .fold(0_u64, |sum, byte| sum.wrapping_add(*byte as u64));
            black_box((
                node_kind,
                node_start,
                node_end,
                node_subtree_end,
                node_flags,
                token_kind,
                token_start,
                token_end,
            ));
        }
        Stage::SyntaxOperationTraverse => {
            for operation in parsed.as_ref().unwrap().syntax().file().operations() {
                checksum = checksum
                    .wrapping_add(operation.id().index() as u64)
                    .wrapping_add(operation.results().count() as u64)
                    .wrapping_add(operation.operands().count() as u64)
                    .wrapping_add(operation.successors().count() as u64)
                    .wrapping_add(operation.regions().count() as u64);
            }
        }
        Stage::SyntaxOperationPayload => {
            let operations = parsed
                .as_ref()
                .unwrap()
                .syntax()
                .file()
                .operations()
                .collect::<Vec<_>>();
            let result_count = operations.iter().map(|op| op.results().count()).sum();
            let operand_count = operations.iter().map(|op| op.operands().count()).sum();
            let successor_count = operations.iter().map(|op| op.successors().count()).sum();
            let region_count = operations.iter().map(|op| op.regions().count()).sum();
            let mut columns = [
                Vec::with_capacity(operations.len()),
                Vec::with_capacity(operations.len() + 1),
                Vec::with_capacity(result_count),
                Vec::with_capacity(operations.len() + 1),
                Vec::with_capacity(operand_count),
                Vec::with_capacity(operations.len() + 1),
                Vec::with_capacity(successor_count),
                Vec::with_capacity(operations.len() + 1),
                Vec::with_capacity(region_count),
            ];
            for index in [1, 3, 5, 7] {
                columns[index].push(0);
            }
            for operation in operations {
                columns[0].push(operation.id().index() as u32);
                columns[2].extend(operation.results().map(|node| node.id().index() as u32));
                let end = columns[2].len() as u32;
                columns[1].push(end);
                columns[4].extend(operation.operands().map(|node| node.id().index() as u32));
                let end = columns[4].len() as u32;
                columns[3].push(end);
                columns[6].extend(operation.successors().map(|node| node.id().index() as u32));
                let end = columns[6].len() as u32;
                columns[5].push(end);
                columns[8].extend(operation.regions().map(|node| node.id().index() as u32));
                let end = columns[8].len() as u32;
                columns[7].push(end);
            }
            checksum = columns
                .iter()
                .flatten()
                .fold(0_u64, |sum, value| sum.wrapping_add(*value as u64));
            output = columns
                .iter()
                .map(|column| column.len() * size_of::<u32>())
                .sum();
            black_box(columns);
        }
        Stage::Lower => {
            document = Some(
                lower_with_dialect_registry_and_retention(
                    parsed.as_ref().unwrap(),
                    LoweringMode::Strict,
                    retention,
                    registry,
                )
                .document
                .expect("strict lowering"),
            )
        }
        Stage::OperationPayload => {
            let doc = document.as_ref().unwrap();
            let mut names = std::collections::HashMap::<u32, u32>::new();
            let mut distinct = Vec::new();
            let mut columns = Vec::<u32>::new();
            for id in doc.operations() {
                let stored = doc.operation_name_index(id).unwrap();
                let next = distinct.len() as u32;
                let code = *names.entry(stored).or_insert_with(|| {
                    distinct.push(stored);
                    next
                });
                let range = doc.operation_source_range(id).unwrap();
                columns.extend([code, range.start(), range.end()]);
                checksum = checksum.wrapping_add(code as u64);
            }
            let name_bytes: usize = distinct
                .iter()
                .map(|&index| doc.string_at(index).unwrap().len())
                .sum();
            output = columns.len() * std::mem::size_of::<u32>()
                + doc.operations().count()
                + (distinct.len() + 1) * std::mem::size_of::<u32>()
                + name_bytes;
            black_box(columns);
        }
        Stage::OperationTraverse => {
            let doc = document.as_ref().unwrap();
            for id in doc.operations() {
                checksum = checksum
                    .wrapping_add(doc.operation_name_index(id).unwrap() as u64)
                    .wrapping_add(doc.operation_source_range(id).unwrap().start() as u64);
            }
        }
        Stage::Verify => document
            .as_ref()
            .unwrap()
            .verify_semantics(registry)
            .unwrap(),
        Stage::Canonical => {
            let mut sink = ByteCounter::default();
            document
                .as_ref()
                .unwrap()
                .write_canonical(&mut sink, PrintLayout::Compact)
                .unwrap();
            output = sink.0;
        }
        Stage::UseIndex => {
            let doc = document.as_ref().unwrap();
            if let Some((_op, value)) = doc.operations().find_map(|op| {
                doc.operation(op)
                    .unwrap()
                    .result(op, 0)
                    .map(|value| (op, value))
            }) {
                black_box(doc.uses(value));
            }
        }
        Stage::SymbolIndex => {
            black_box(
                document
                    .as_ref()
                    .unwrap()
                    .symbol_index_diagnostics(registry),
            );
        }
        Stage::DominanceIndex => {
            let doc = document.as_ref().unwrap();
            let ops = doc.operations().collect::<Vec<_>>();
            if let (Some((_, value)), Some(last)) = (
                ops.iter().find_map(|op| {
                    doc.operation(*op)
                        .unwrap()
                        .result(*op, 0)
                        .map(|value| (*op, value))
                }),
                ops.last(),
            ) {
                black_box(doc.dominates(value, *last, registry));
            }
        }
        Stage::Editor => {
            let doc = document.as_mut().unwrap();
            let mut editor = doc.edit(registry).unwrap();
            if shape == Shape::BlockRich {
                black_box(editor.compact_pools());
            } else {
                let leaf = editor
                    .document()
                    .operations()
                    .find(|op| editor.document().operation_name(*op) == Some("bench.op"))
                    .expect("editable leaf");
                editor.erase(leaf).unwrap();
            }
            editor.commit().unwrap();
        }
        Stage::Preserving | Stage::DirtyPreserving => {
            let doc = document.as_ref().unwrap();
            let mut sink = ByteCounter::default();
            doc.write_preserving(&mut sink, PrintLayout::Compact)
                .unwrap();
            output = sink.0;
        }
    }
    let elapsed = started.elapsed();
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(baseline) as usize;
    let stats = document.as_ref().map(|doc| { let s=doc.statistics(); format!("operations={} regions={} blocks={} pooled_entries={} direct_owned_bytes={} document_index_bytes={} retained_source_bytes={} retained_cst_nodes={} retained_cst_bytes={} source_storage_shared={} cst_storage_shared={} retained_mapping_entries={} payload_blob_bytes={} use_index_entries={} symbol_index_entries={} dominance_index_entries={}",s.operations,s.regions,s.blocks,s.pooled_list_entries,s.direct_owned_bytes,s.document_index_bytes,s.retained_source_bytes,s.retained_cst_nodes,s.retained_cst_bytes,s.source_storage_shared,s.cst_storage_shared,s.retained_mapping_entries,s.payload_blob_bytes,s.use_index_entries,s.symbol_index_entries,s.dominance_index_entries) }).unwrap_or_else(|| { let tree=parsed.as_ref().map(|p|p.syntax().tree()); format!("cst_nodes={} cst_tokens={} exact_cst_retained_bytes={}",tree.map_or(0,|t|t.node_count()),tree.map_or(0,|t|t.token_count()),tree.map_or(0,|t|t.exact_retained_bytes())) });
    black_box(checksum);
    Measurement {
        ns: elapsed.as_nanos(),
        peak,
        output,
        checksum,
        stats,
    }
}

fn generate(shape: Shape, bytes: usize, depth: Option<usize>) -> Vec<u8> {
    if shape == Shape::Primary {
        return primary_fixture(bytes);
    }
    if shape == Shape::BlockRich {
        return block_rich_fixture(bytes);
    }
    if shape == Shape::Nested {
        return nested_fixture(bytes, depth.expect("validated nested depth"));
    }
    if shape == Shape::Payload {
        return payload_fixture(bytes);
    }
    let mut out = match shape {
        Shape::Primary => b"\"bench.container\"() ({\n^bb:\n%seed = \"bench.source\"() : () -> i32\n\"bench.use\"(%seed) : (i32) -> ()\n".to_vec(),
        Shape::BlockRich => b"builtin.module { func.func @bench() {\n^entry:\n".to_vec(),
        Shape::Trivia => b"builtin.module {\n".to_vec(),
        Shape::Nested => b"builtin.module {\n".to_vec(),
        Shape::Payload => b"builtin.module {\n".to_vec(),
    };
    let close: &[u8] = match shape {
        Shape::Primary => b"}) : () -> ()\n",
        Shape::BlockRich => b"func.return\n} }\n",
        Shape::Trivia | Shape::Nested | Shape::Payload => b"}\n",
    };
    let line: &[u8] = match shape {
        Shape::Primary | Shape::BlockRich => b"\"bench.op\"() {tag = \"zirium\"} : () -> () // deterministic primary operation padding xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n",
        Shape::Trivia => b"// deterministic trivia xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n\n",
        Shape::Nested => b"\"bench.region\"() ({ \"bench.op\"() : () -> () }) : () -> ()\n",
        Shape::Payload => b"\"bench.payload\"() {value = #vendor.attr<\"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\">} : () -> ()\n",
    };
    assert!(bytes >= out.len() + close.len());
    while out.len() + close.len() + line.len() <= bytes {
        out.extend_from_slice(line);
    }
    let padding = bytes - out.len() - close.len();
    if padding > 0 {
        out.extend(std::iter::repeat_n(b' ', padding));
    }
    out.extend_from_slice(close);
    out
}

fn primary_fixture(bytes: usize) -> Vec<u8> {
    let prefix = b"\"bench.container\"() ({\n^bb:\n%seed = \"bench.source\"() : () -> i32\n\"bench.use\"(%seed) : (i32) -> ()\n";
    let suffix = b"}) : () -> ()\n";
    let mut line =
        b"\"bench.op\"() {tag = \"zirium\"} : () -> () // deterministic operation ".to_vec();
    line.extend(std::iter::repeat_n(b'x', 440));
    line.push(b'\n');
    let mut out = Vec::with_capacity(bytes);
    out.extend_from_slice(prefix);
    while out.len() + line.len() + suffix.len() <= bytes {
        out.extend_from_slice(&line);
    }
    out.extend(std::iter::repeat_n(b' ', bytes - out.len() - suffix.len()));
    out.extend_from_slice(suffix);
    out
}

fn nested_fixture(bytes: usize, depth: usize) -> Vec<u8> {
    let open = b"\"bench.region\"() ({\n";
    let close = b"}) : () -> ()\n";
    let mut out = Vec::with_capacity(bytes);
    for _ in 0..depth {
        out.extend_from_slice(open);
    }
    out.extend_from_slice(b"\"bench.op\"() : () -> ()\n");
    let closing = close.len() * depth;
    assert!(
        out.len() + closing <= bytes,
        "requested depth does not fit fixture size"
    );
    out.extend(std::iter::repeat_n(b' ', bytes - out.len() - closing));
    for _ in 0..depth {
        out.extend_from_slice(close);
    }
    out
}

fn block_rich_fixture(bytes: usize) -> Vec<u8> {
    let mut out = b"builtin.module {\n".to_vec();
    let close = b"}\n";
    let mut index = 0;
    loop {
        let function = format!(
            "func.func @f{index}() {{\n^entry:\n%value = arith.constant 1 : i32\ncf.br ^middle\n^middle:\n\"bench.use\"(%value) : (i32) -> ()\ncf.br ^exit\n^exit:\nfunc.return\n}}\n"
        );
        if out.len() + function.len() + close.len() > bytes {
            break;
        }
        out.extend_from_slice(function.as_bytes());
        index += 1;
    }
    out.extend(std::iter::repeat_n(b' ', bytes - out.len() - close.len()));
    out.extend_from_slice(close);
    out
}

fn payload_fixture(bytes: usize) -> Vec<u8> {
    let prefix = b"\"bench.payload\"() {value = #vendor.attr<\"";
    let suffix = b"\">} : () -> ()\n";
    assert!(bytes >= prefix.len() + suffix.len());
    let mut out = Vec::with_capacity(bytes);
    out.extend_from_slice(prefix);
    out.extend(std::iter::repeat_n(
        b'x',
        bytes - prefix.len() - suffix.len(),
    ));
    out.extend_from_slice(suffix);
    out
}

fn temp_path() -> PathBuf {
    env::temp_dir().join(format!(
        "zirium-processing-{}-{}.mlir",
        std::process::id(),
        SEED
    ))
}

fn print_environment(args: &Args) {
    println!(
        "benchmark=processing profile=release warmups={} measured_runs={} smoke={} shape={} depth={} requested_bytes={}",
        args.warmups,
        args.runs,
        args.smoke,
        args.shape.name(),
        args.depth
            .map_or_else(|| "none".into(), |depth| depth.to_string()),
        args.bytes
    );
    println!(
        "rustc={} target={} os={} cpu={}",
        command("rustc", &["-V"]),
        command("rustc", &["-vV"])
            .lines()
            .find_map(|l| l.strip_prefix("host: "))
            .unwrap_or("unknown"),
        command("uname", &["-srv"]),
        if cfg!(target_os = "macos") {
            command("sysctl", &["-n", "machdep.cpu.brand_string"])
        } else {
            "unknown".into()
        }
    );
}

fn command(program: &str, args: &[&str]) -> String {
    Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_else(|| "unknown".into())
}

impl Shape {
    fn parse(s: &str) -> Self {
        match s {
            "primary" => Self::Primary,
            "block-rich" => Self::BlockRich,
            "trivia" => Self::Trivia,
            "nested" => Self::Nested,
            "payload" => Self::Payload,
            _ => panic!("unknown shape {s}"),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Primary => "primary",
            Self::BlockRich => "block-rich",
            Self::Trivia => "trivia",
            Self::Nested => "nested",
            Self::Payload => "payload",
        }
    }
}
impl Stage {
    fn parse(s: &str) -> Self {
        match s {
            "parse" => Self::Parse,
            "traverse" => Self::Traverse,
            "syntax-payload" => Self::SyntaxPayload,
            "syntax-operation-payload" => Self::SyntaxOperationPayload,
            "syntax-operation-traverse" => Self::SyntaxOperationTraverse,
            "operation-payload" => Self::OperationPayload,
            "operation-traverse" => Self::OperationTraverse,
            "lower" => Self::Lower,
            "verify" => Self::Verify,
            "canonical" => Self::Canonical,
            "use-index" => Self::UseIndex,
            "symbol-index" => Self::SymbolIndex,
            "dominance-index" => Self::DominanceIndex,
            "editor" => Self::Editor,
            "preserving" => Self::Preserving,
            "dirty-preserving" => Self::DirtyPreserving,
            "all" => panic!("use comma-separated stage names"),
            _ => panic!("unknown stage {s}"),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Parse => "parse",
            Self::Traverse => "traverse",
            Self::SyntaxPayload => "syntax-payload",
            Self::SyntaxOperationPayload => "syntax-operation-payload",
            Self::SyntaxOperationTraverse => "syntax-operation-traverse",
            Self::OperationPayload => "operation-payload",
            Self::OperationTraverse => "operation-traverse",
            Self::Lower => "lower",
            Self::Verify => "verify",
            Self::Canonical => "canonical",
            Self::UseIndex => "use-index",
            Self::SymbolIndex => "symbol-index",
            Self::DominanceIndex => "dominance-index",
            Self::Editor => "editor",
            Self::Preserving => "preserving",
            Self::DirtyPreserving => "dirty-preserving",
        }
    }
}

fn report() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    println!(
        "projection_target_mib=500 input_policy=direct_measurement_permitted_for_bounded_stages fit_points_mib=1,10,25,50 held_out_points_mib=75,100 held_out_error_limit=10_percent slope_intervals_mib=25-50,50-75,75-100 slope_deviation_limit=10_percent_of_median signal_floor=10_mib_latency>=0.005_seconds"
    );
    for stage in [
        "parse",
        "traverse",
        "syntax-payload",
        "operation-payload",
        "operation-traverse",
        "lower",
        "verify",
        "canonical",
        "use-index",
        "symbol-index",
        "dominance-index",
        "editor",
        "preserving",
        "dirty-preserving",
    ] {
        let mut points = Vec::new();
        for line in input.lines().filter(|l| {
            l.starts_with("measurement ")
                && l.contains(&format!("stage={stage} "))
                && l.contains("shape=primary ")
        }) {
            let get = |key: &str| {
                line.split_whitespace()
                    .find_map(|p| p.strip_prefix(&format!("{key}=")))
                    .unwrap()
                    .parse::<f64>()
                    .unwrap()
            };
            points.push((
                get("input_bytes") / MIB as f64,
                get("median_ns") / 1e9,
                get("peak_live_bytes") / MIB as f64,
            ));
        }
        if points.len() != 6 {
            println!(
                "projection stage={stage} label=projected status=not-credible reason=requires_exactly_six_primary_points observed_points={}",
                points.len()
            );
            continue;
        }
        points.sort_by(|a, b| a.0.total_cmp(&b.0));
        if points.iter().map(|p| p.0 as usize).collect::<Vec<_>>() != [1, 10, 25, 50, 75, 100] {
            println!(
                "projection stage={stage} label=projected status=not-credible reason=requires_1_10_25_50_75_100_mib_points"
            );
            continue;
        }
        let fitted = &points[..4];
        let (ta, tb) = fit(fitted, |p| p.1);
        let (ma, mb) = fit(fitted, |p| p.2);
        let time_errors = held_out_errors(&points[4..], ta, tb, |p| p.1);
        let peak_errors = held_out_errors(&points[4..], ma, mb, |p| p.2);
        let time_slopes = interval_slopes(&points[2..], |p| p.1);
        let peak_slopes = interval_slopes(&points[2..], |p| p.2);
        let time_stable = slopes_stable(time_slopes);
        let peak_stable = slopes_stable(peak_slopes);
        println!(
            "fit stage={stage} trained_on_mib=1,10,25,50 time_formula_seconds={ta:.9}+{tb:.9}*MiB peak_formula_mib={ma:.6}+{mb:.6}*MiB held_out_time_error_percent={:.3},{:.3} held_out_peak_error_percent={:.3},{:.3} time_slopes_per_mib={:.9},{:.9},{:.9} peak_slopes_per_mib={:.6},{:.6},{:.6} time_slope_stable={} peak_slope_stable={}",
            time_errors[0] * 100.0,
            time_errors[1] * 100.0,
            peak_errors[0] * 100.0,
            peak_errors[1] * 100.0,
            time_slopes[0],
            time_slopes[1],
            time_slopes[2],
            peak_slopes[0],
            peak_slopes[1],
            peak_slopes[2],
            time_stable,
            peak_stable
        );
        let held_out_credible = time_errors
            .iter()
            .chain(&peak_errors)
            .all(|error| *error <= 0.10);
        // Sub-millisecond stages are dominated by timer and allocator noise at
        // this scale. Retain the established 10 MiB signal floor
        // before allowing a claim about a 500 MiB continuation.
        let credible = held_out_credible && time_stable && peak_stable && points[1].1 >= 0.005;
        if credible {
            println!(
                "projection stage={stage} label=projected status=credible target_mib=500 estimated_seconds={:.3} estimated_peak_mib={:.1} assumptions=same_fixture_mix_allocator_hardware_release_build_and_no_paging_cliff_beyond_100_mib",
                ta + tb * 500.0,
                ma + mb * 500.0
            );
        } else {
            let reason = if points[1].1 < 0.005 {
                "10_mib_latency_below_5ms_signal_floor"
            } else if !held_out_credible {
                "held_out_error_exceeds_10_percent"
            } else {
                "slope_deviation_exceeds_10_percent_of_median"
            };
            println!(
                "projection stage={stage} label=projected status=not-credible reason={reason} assumptions=same_fixture_mix_allocator_hardware_release_build_and_no_paging_cliff_beyond_100_mib"
            );
        }
    }
}

fn fit<F: Fn(&(f64, f64, f64)) -> f64>(points: &[(f64, f64, f64)], y: F) -> (f64, f64) {
    let count = points.len() as f64;
    let xm = points.iter().map(|p| p.0).sum::<f64>() / count;
    let ym = points.iter().map(&y).sum::<f64>() / count;
    let b = points.iter().map(|p| (p.0 - xm) * (y(p) - ym)).sum::<f64>()
        / points.iter().map(|p| (p.0 - xm).powi(2)).sum::<f64>();
    let a = ym - b * xm;
    (a, b)
}

fn held_out_errors<F: Fn(&(f64, f64, f64)) -> f64>(
    points: &[(f64, f64, f64)],
    a: f64,
    b: f64,
    y: F,
) -> [f64; 2] {
    std::array::from_fn(|i| {
        ((a + b * points[i].0) - y(&points[i])).abs() / y(&points[i]).abs().max(1e-12)
    })
}

fn interval_slopes<F: Fn(&(f64, f64, f64)) -> f64>(points: &[(f64, f64, f64)], y: F) -> [f64; 3] {
    std::array::from_fn(|i| (y(&points[i + 1]) - y(&points[i])) / (points[i + 1].0 - points[i].0))
}

fn slopes_stable(mut slopes: [f64; 3]) -> bool {
    slopes.sort_by(f64::total_cmp);
    let median = slopes[1];
    slopes
        .into_iter()
        .all(|slope| (slope - median).abs() <= median.abs().max(1e-12) * 0.10)
}
