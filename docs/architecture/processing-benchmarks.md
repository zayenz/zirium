# Processing benchmarks

Use the Rust and Python harnesses to measure parsing, semantic processing,
printing, and bulk API access. Fixtures are deterministic; compare runs using
the same fixture, build profile, machine, and compiler. Generate results from
the checkout being measured.

## Full pipeline

The Rust harness generates exact-size fixtures in temporary storage with seed
`0x5a495249554d0028`. Select stages to isolate parsing, traversal, payload
construction, lowering, printing, indexes, or editing:

```sh
cargo build --release -p zirium --example processing_benchmark
for size in 1 10 25 50 75 100; do
  target/release/examples/processing_benchmark --size-mib "$size" --shape primary --stage parse,traverse,syntax-payload,lower,canonical,use-index,symbol-index,dominance-index,editor,preserving --warmups 1 --runs 3
done > /tmp/zirium-primary.txt
target/release/examples/processing_benchmark --report < /tmp/zirium-primary.txt
```

Use different shapes to distinguish byte scanning from syntax and control-flow
costs. `--depth` is required for nested fixtures and rejected for other shapes.
The generator adjusts delimiter limits for the requested depth and pads the
fixture to its requested size.

```sh
for depth in 8 64 256; do
  target/release/examples/processing_benchmark --size-mib 10 --shape nested --depth "$depth" --stage parse,traverse,lower --warmups 1 --runs 3
done
target/release/examples/processing_benchmark --size-mib 10 --shape block-rich --stage parse,lower,verify,dominance-index --warmups 1 --runs 3
```

Peak live allocation is measured above each stage's live inputs. Peaks from
different stages cannot be added. Exact retained CST bytes exclude the source.
Document-owned statistics exclude Python wrappers, interpreter memory, and RSS.

The report fits 1, 10, 25, and 50 MiB samples and checks projections against
75 and 100 MiB. A projection requires held-out errors and per-MiB slope variation
within 10%, plus at least 5 ms of latency at 10 MiB. Treat passing projections
as estimates for the same fixture mix and environment, not measured results.

## Edit transactions

The ignored Rust benchmark compares one transaction containing many attribute
edits with the same edits committed as separate transactions. It varies both
the number of operations in the document and the number of edits, checks the
result after every measured run, and separates the transaction-opening
structural validation, document copy, edit calls, commit structural validation,
semantic verifier, total time, and peak allocated bytes above the live input.

```sh
ZIRIUM_EDIT_BENCH_SMOKE=1 cargo test --release -p zirium semantic::edit::edit_transaction_benchmark::measure_edit_transaction_costs -- --ignored --nocapture --test-threads=1

ZIRIUM_EDIT_BENCH_OPERATIONS=1000,10000 \
ZIRIUM_EDIT_BENCH_EDITS=1,10,100 \
ZIRIUM_EDIT_BENCH_WARMUPS=1 \
ZIRIUM_EDIT_BENCH_RUNS=3 \
cargo test --release -p zirium semantic::edit::edit_transaction_benchmark::measure_edit_transaction_costs -- --ignored --nocapture --test-threads=1
```

Use a release build and one test thread because the harness measures process-wide
allocations. Output records the compiler, target, operating system, deterministic
seed, workload dimensions, repetitions, medians, and min-to-max spread. Peak
bytes are allocator high-water growth during the transaction workload; they do
not include fixture parsing and lowering, and allocator retention can affect the
baseline. The benchmark has no timing or memory pass/fail threshold.

`Document::edit` checks the whole starting document and copies its semantic
storage. `DocumentEditor::commit` checks the whole working document again and
runs semantic verification. Consequently, separate transactions repeat
document-wide work even when each transaction changes only one operation.
Batch related edits in one transaction when they should succeed or roll back as
one unit. Use separate transactions when an intermediate committed state must be
observable or when edits need independent rollback; expect their validation,
copying, and verification costs to scale with the transaction count. These
measurements establish the current cost boundary and do not justify journaling,
copy-on-write storage, or targeted validation on their own.

For orientation, a 14 September 2026 release run on an Apple M1 Max with Rust
1.98.1 measured 1,000 operations and ten edits as follows (three measured runs
after one warm-up):

| Mode | Opening validation | Copy | Edit | Commit validation | Verifier | Total | Peak allocation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| One batch | 10.0 us | 9.6 us | 23.3 us | 10.3 us | 20.1 us | 75.5 us | 327 KB |
| Ten transactions | 98.1 us | 98.2 us | 24.5 us | 101 us | 200 us | 547 us | 326 KB |

This single recorded run illustrates the repeated document-wide work; it is not
a performance guarantee. Re-run the harness on the target machine and workload
before making design or capacity decisions.

## Parser construction

The ignored internal test separates lexing, grammar events, compaction, and
structural verification. It also reports a complete `parser_whole` measurement
and estimates the cost of adding identifier interning.

```sh
ZIRIUM_PARSER_BENCH_SMOKE=1 cargo test --release -p zirium parser_construction_benchmark::measure_parser_construction_and_string_identity -- --ignored --nocapture --test-threads=1

ZIRIUM_PARSER_BENCH_SHAPE=primary ZIRIUM_PARSER_BENCH_SIZE_MIB=10 ZIRIUM_PARSER_BENCH_RUNS=3 cargo test --release -p zirium parser_construction_benchmark::measure_parser_construction_and_string_identity -- --ignored --nocapture --test-threads=1
```

Smoke mode uses 64 KiB, one warm-up, and one measured run. Defaults are a 10 MiB
primary fixture, one warm-up, and three runs. Override them with
`ZIRIUM_PARSER_BENCH_SHAPE`, `ZIRIUM_PARSER_BENCH_SIZE_MIB`,
`ZIRIUM_PARSER_BENCH_WARMUPS`, and `ZIRIUM_PARSER_BENCH_RUNS`.

Lexing retains the source; grammar production also retains tokens; compaction
adds events and working storage; verification retains the unverified CST.
Keep those different baselines in mind when comparing phase peaks.

## Python access and output

Build the release extension before measuring Python access:

```sh
uv run --locked maturin develop --release
.venv/bin/python python/benchmarks/processing_benchmark.py --smoke
.venv/bin/python python/benchmarks/processing_benchmark.py --size-mib 10 --runs 3
.venv/bin/python python/benchmarks/processing_benchmark.py --size-mib 1 --shape block-rich --runs 3
```

The harness measures wrappers, packed snapshots, and file output. Compare packed
Python tables with native payload construction as well as native traversal:
payload construction includes allocation and encoding that a bare walk omits.
Python fills final `bytes` columns directly, without temporary column copies.
`tracemalloc` measures Python allocations, not the Rust heap.

Attribute traversal has a separate release benchmark for ordinary arrays,
dictionaries, and dense arrays. It checks every child value and dictionary key,
includes the first traversal that builds the wrapper's spelling index, and
reports medians, spread, per-element cost, and adjacent size ratios from 1,000
through 8,000 elements. Ratios near 2 when the element count doubles are the
expected linear shape; they are measurement evidence, not CI timing gates.

```sh
uv run --locked maturin develop --release
.venv/bin/python python/benchmarks/attribute_traversal_benchmark.py --runs 5
```

Canonical and custom file output validate once and stream through a Rust
`BufWriter`. Original output copies the parsed source. Preserving output depends
on hybrid retention and valid source mappings; see [output contracts](../getting-started.md#write-output).

## Process RSS

RSS mode reads an existing file in separate children for source loading, parsing,
and hybrid best-effort lowering. This isolates measurements from earlier
high-water marks and fixture generation.

```sh
.venv/bin/python python/benchmarks/processing_benchmark.py --size-mib 10 --shape primary --write-fixture /tmp/zirium-primary-10.mlir
.venv/bin/python python/benchmarks/processing_benchmark.py --rss-input /tmp/zirium-primary-10.mlir
```

Use `--rss-max-delimiter-depth 512` for a nested fixture requiring that limit.
On macOS, current RSS comes from `PROC_PIDTASKINFO`, and `ru_maxrss` is in bytes.
On Linux, current RSS comes from `/proc/self/statm`; `ru_maxrss` is converted from
KiB to bytes.

Each row includes input bytes, RSS before source loading, source-resident RSS,
and peak RSS after the selected stage. Interpret both ratios:

- `(parse peak RSS - imported-process RSS) / input bytes` includes source loading.
- `(parse peak RSS - source-resident RSS) / input bytes` measures growth beyond
  the loaded source.

Source loading and allocator retention affect the baselines differently. Keep
the formula with each result; ratios from different harnesses are not interchangeable.

## CLI selection output

```sh
cargo build --release -p zirium --bin zirium
python3 python/benchmarks/selection_printing_benchmark.py
```

This measures startup, parsing, lowering, query evaluation, printing, and pipe
I/O for modules with 128, 512, and 1,024 independent constants. It checks output
counts and reports three-run medians after one warm-up. An optional positional
argument selects another binary.

The opt-in output-amplification mode measures the CLI's peak RSS while many
emissions are staged before stdout delivery:

```sh
python3 python/benchmarks/selection_printing_benchmark.py --output-rss
```

It accepts only a binary in a `release` directory. Normal and JSONL runs execute
in separate child processes, redirect stdout to a file, and compare every byte
with a one-emission reference. The default workload emits one 128-constant
selection 256, 1,024, and 4,096 times. Each row records the platform, binary,
release profile, workload dimensions, emitted bytes, run count, elapsed-time
median and spread, and peak-RSS median and spread. Use `--rss-constants`,
`--rss-emissions`, and `--rss-runs` to change those dimensions. Large RSS runs
remain manual measurements; there is no CI memory or timing threshold.

For evaluator-only and generated-graph checks, see [query profiling](query-profiling.md).
For production-shaped inputs, see the [compiler-input stress benchmark](stress-instance-benchmark.md).
