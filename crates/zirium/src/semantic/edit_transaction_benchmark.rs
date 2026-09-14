use std::env;
use std::hint::black_box;
use std::process::Command;
use std::time::{Duration, Instant};

use crate::dialect::DialectRegistry;
use crate::parser::ParsedFile;

use super::{AnalysisCaches, AttributeSpec, Document, DocumentEditor, LoweringMode};

const SEED: u64 = 0x5a49_5249_554d_0015;

#[derive(Clone, Copy, Default)]
struct Stages {
    opening_validation: Duration,
    copy: Duration,
    edit: Duration,
    commit_validation: Duration,
    verifier: Duration,
    total: Duration,
    peak_bytes: usize,
}

impl std::ops::AddAssign for Stages {
    fn add_assign(&mut self, other: Self) {
        self.opening_validation += other.opening_validation;
        self.copy += other.copy;
        self.edit += other.edit;
        self.commit_validation += other.commit_validation;
        self.verifier += other.verifier;
        self.total += other.total;
        self.peak_bytes = self.peak_bytes.max(other.peak_bytes);
    }
}

fn fixture(operation_count: usize) -> Document {
    let mut source = String::with_capacity(operation_count * 32);
    for _ in 0..operation_count {
        source.push_str("\"bench.op\"() : () -> ()\n");
    }
    let parsed = ParsedFile::parse(source.as_bytes()).expect("benchmark fixture parses");
    let result =
        super::lower_with_dialect_registry(&parsed, LoweringMode::Strict, &DialectRegistry::EMPTY);
    assert!(result.diagnostics.is_empty());
    let document = result.document.expect("benchmark fixture lowers");
    document
        .validate_structure()
        .expect("fixture is structurally valid");
    document
        .verify_semantics(&DialectRegistry::EMPTY)
        .expect("fixture verifies");
    document
}

fn transaction(document: &mut Document, targets: &[super::OperationId]) -> Stages {
    let total_started = Instant::now();
    let started = Instant::now();
    document
        .validate_structure()
        .expect("transaction input stays structurally valid");
    let opening_validation = started.elapsed();

    let started = Instant::now();
    let working = document.edit_snapshot();
    let copy = started.elapsed();
    let mut editor = DocumentEditor {
        working,
        original: document,
        registry: &DialectRegistry::EMPTY,
    };

    let started = Instant::now();
    for operation in targets {
        editor
            .set_attribute(*operation, AttributeSpec::string("bench.tag", "edited"))
            .expect("benchmark edit succeeds");
    }
    let edit = started.elapsed();

    let started = Instant::now();
    editor
        .working
        .validate_structure()
        .expect("edited document stays structurally valid");
    let commit_validation = started.elapsed();

    let started = Instant::now();
    editor
        .working
        .verify_semantics_only(editor.registry)
        .expect("edited document verifies");
    let verifier = started.elapsed();

    editor.working.revision = editor.original.revision.wrapping_add(1);
    *editor
        .working
        .analyses
        .0
        .write()
        .expect("analysis cache lock is not poisoned") = AnalysisCaches::default();
    *editor.original = editor.working;

    Stages {
        opening_validation,
        copy,
        edit,
        commit_validation,
        verifier,
        total: total_started.elapsed(),
        peak_bytes: 0,
    }
}

fn measure(operation_count: usize, edit_count: usize, batched: bool) -> Stages {
    let mut document = fixture(operation_count);
    let targets = document.root_operations()[..edit_count].to_vec();
    let baseline = crate::benchmark_allocator::begin();

    let mut stages = Stages::default();
    if batched {
        stages += transaction(&mut document, &targets);
    } else {
        for target in &targets {
            stages += transaction(&mut document, std::slice::from_ref(target));
        }
    }
    stages.peak_bytes = crate::benchmark_allocator::finish(baseline);

    document
        .validate_structure()
        .expect("measured result validates");
    document
        .verify_semantics(&DialectRegistry::EMPTY)
        .expect("measured result verifies");
    assert!(
        targets
            .iter()
            .all(|target| document.attribute_id(*target, "bench.tag").is_some())
    );
    assert_eq!(
        document.revision(),
        if batched { 1 } else { edit_count as u64 }
    );
    black_box(document);
    stages
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

fn field(samples: &[Stages], select: impl Fn(Stages) -> Duration) -> (u128, u128) {
    let mut values = samples
        .iter()
        .copied()
        .map(|sample| select(sample).as_nanos())
        .collect::<Vec<_>>();
    let min = *values.iter().min().unwrap();
    let max = *values.iter().max().unwrap();
    (median(&mut values), max - min)
}

#[cfg(debug_assertions)]
fn require_release() {
    panic!("run this benchmark with --release");
}

#[cfg(not(debug_assertions))]
fn require_release() {}

#[test]
#[ignore = "release-only edit transaction measurement; see processing-benchmarks.md"]
fn measure_edit_transaction_costs() {
    require_release();
    let smoke = env::var_os("ZIRIUM_EDIT_BENCH_SMOKE").is_some();
    let operations = parse_list(
        "ZIRIUM_EDIT_BENCH_OPERATIONS",
        if smoke { &[64] } else { &[1_000, 10_000] },
    );
    let edits = parse_list(
        "ZIRIUM_EDIT_BENCH_EDITS",
        if smoke { &[4] } else { &[1, 10, 100] },
    );
    let warmups = env::var("ZIRIUM_EDIT_BENCH_WARMUPS")
        .ok()
        .map_or(if smoke { 0 } else { 1 }, |value| value.parse().unwrap());
    let runs = env::var("ZIRIUM_EDIT_BENCH_RUNS")
        .ok()
        .map_or(if smoke { 1 } else { 3 }, |value| value.parse().unwrap());
    assert!(runs > 0);

    println!(
        "benchmark=edit-transactions profile=release seed=0x{SEED:016x} warmups={warmups} measured_runs={runs} rustc={} target={} os={}",
        command("rustc", &["-V"]),
        command("rustc", &["-vV"])
            .lines()
            .find_map(|line| line.strip_prefix("host: "))
            .unwrap_or("unknown"),
        command("uname", &["-srv"]),
    );
    println!(
        "operations,edits,mode,opening_validation_ns,opening_validation_spread_ns,copy_ns,copy_spread_ns,edit_ns,edit_spread_ns,commit_validation_ns,commit_validation_spread_ns,verifier_ns,verifier_spread_ns,total_ns,total_spread_ns,peak_allocated_bytes"
    );

    for operation_count in operations {
        for &edit_count in &edits {
            assert!(edit_count > 0 && edit_count <= operation_count);
            for (mode, batched) in [("batch", true), ("many", false)] {
                for _ in 0..warmups {
                    black_box(measure(operation_count, edit_count, batched));
                }
                let samples = (0..runs)
                    .map(|_| measure(operation_count, edit_count, batched))
                    .collect::<Vec<_>>();
                let opening = field(&samples, |sample| sample.opening_validation);
                let copy = field(&samples, |sample| sample.copy);
                let edit = field(&samples, |sample| sample.edit);
                let validation = field(&samples, |sample| sample.commit_validation);
                let verifier = field(&samples, |sample| sample.verifier);
                let total = field(&samples, |sample| sample.total);
                let mut peaks = samples
                    .iter()
                    .map(|sample| sample.peak_bytes as u128)
                    .collect::<Vec<_>>();
                let peak = median(&mut peaks);
                println!(
                    "{operation_count},{edit_count},{mode},{},{},{},{},{},{},{},{},{},{},{},{},{peak}",
                    opening.0,
                    opening.1,
                    copy.0,
                    copy.1,
                    edit.0,
                    edit.1,
                    validation.0,
                    validation.1,
                    verifier.0,
                    verifier.1,
                    total.0,
                    total.1,
                );
            }
        }
    }
}
