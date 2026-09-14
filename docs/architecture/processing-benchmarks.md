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

## Edit interning

The ignored Rust benchmark isolates the private string, type, and attribute
interners used by one open edit. It varies distinct and repeated values, value
count, and nesting depth, checks the resulting interner cardinality, and records
peak allocated bytes above the open editor:

```sh
ZIRIUM_EDIT_INTERNER_BENCH_SMOKE=1 \
cargo test --release -p zirium semantic::edit::edit_interner_benchmark::measure_edit_interner_scaling -- --ignored --nocapture --test-threads=1

ZIRIUM_EDIT_INTERNER_BENCH_COUNTS=1000,2000,4000 \
ZIRIUM_EDIT_INTERNER_BENCH_DEPTHS=0,4,16 \
ZIRIUM_EDIT_INTERNER_BENCH_WARMUPS=1 \
ZIRIUM_EDIT_INTERNER_BENCH_RUNS=3 \
cargo test --release -p zirium semantic::edit::edit_interner_benchmark::measure_edit_interner_scaling -- --ignored --nocapture --test-threads=1
```

Strings use depth zero. Types and attributes run every selected depth. Full
defaults are the dimensions in the second command; override them with the same
environment variables. Use one release test thread because allocation tracking
is process-wide. There is no timing or memory threshold in CI.

On 14 September 2026, an Apple M1 Max with Rust 1.98.1 ran the matrix above with
three measured runs after one warm-up. These representative medians compare the
original linear scan with first-value indexes; durations are milliseconds:

| Kind | Depth | Pattern | Count | Linear scan | Indexed | Indexed peak allocation |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| String | 0 | Distinct | 1,000 | 1.319 | 0.187 | 154 KB |
| String | 0 | Distinct | 4,000 | 14.095 | 0.757 | 623 KB |
| String | 0 | Repeated | 4,000 | 0.246 | 0.239 | 126 B |
| Type | 16 | Distinct | 1,000 | 27.874 | 2.170 | 3.46 MB |
| Type | 16 | Distinct | 4,000 | 426.025 | 8.908 | 13.82 MB |
| Type | 16 | Repeated | 4,000 | 2.703 | 2.703 | 5.0 KB |
| Attribute | 16 | Distinct | 1,000 | 79.958 | 4.546 | 4.69 MB |
| Attribute | 16 | Distinct | 4,000 | 1,241.027 | 18.625 | 18.77 MB |
| Attribute | 16 | Repeated | 4,000 | 5.584 | 5.530 | 7.5 KB |

The retained string, type, and attribute indexes reduced the distinct 4,000-value
cases by 18.6x, 47.8x, and 66.6x respectively. Their 1,000-to-4,000 distinct
ratios were 4.1x after indexing, while the measured repeated cases did not
regress. Index keys increase allocation for distinct values: the corresponding
linear-scan peaks at 4,000 values were 165 KB, 6.75 MB, and 9.22 MB. The existing
vectors remain the storage and output-order authority; hash maps are used only
to find the first equal vector index and are never iterated for output.

## Semantic compaction lifetimes

`Editor::compact_pools` reclaims only fragmented list-pool entries. It does not
reclaim or remap interned strings, types, attributes, locations, affine
expressions, affine maps, or integer sets. Those IDs may have escaped through a
public accessor or a dialect callback. They remain valid for their original
value for the lifetime of the document, and a later edit never reuses an ID for
a different value. Erased operation and value handles remain stale under the
document's generation checks.

Registered type and attribute verification callbacks run only for values
reachable from live operation result and function types, live block argument
types, live attributes, live properties, locations, and their nested values.
Callbacks must not retain borrowed value references beyond the call. They may
copy a public ID or owned spelling; copied IDs follow the document-lifetime
rule above. Nested type/attribute values, locations, aliases after lowering,
and affine references are followed by structural validation and registered
value traversal before callbacks run.

This deliberately trades bounded identity semantics for retained arena growth.
Long-lived processes with unbounded replacement workloads should periodically
rebuild a document from canonical output when memory matters. Rebuilding
creates a new document identity, so old handles become foreign rather than
being silently remapped.

The release benchmark performs unique result-type replacement plus temporary
attribute insertion/removal, compacts list pools, and checks the final document.
It reports live and retained string/type/attribute counts, direct owned bytes,
and reclaimed list entries. Timed fields cover the complete edit transaction
(`edit_ns`), final semantic verification (`verify_ns`), and peak allocation
above each run's live input (`peak_allocated_bytes`). Each timed field reports
minimum, median, maximum, and absolute spread across measured runs. The header
records the Rust compiler, host target, OS kernel, and debug/release profile.
Warm-ups are discarded, measurements rebuild the same input for every run, and
there are no timing or memory thresholds.

```sh
cargo run --release -p zirium --example semantic_compaction_benchmark -- \
  --iterations 10000 --warmups 1 --runs 3
```

On 14 September 2026, an Apple M1 Max ran the command above using Rust 1.98.1,
target `aarch64-apple-darwin`, Darwin 25.6.0, and a release build. It retained
10,001 strings, 20,002 types, 10,000 attributes, and 6,904,361 direct owned
bytes while only 1 string, 2 types, and no attributes remained live. List
compaction reclaimed 20,001 entries. The three-run edit median was 707.80 ms
with 9.44 ms spread; verification was 127.83 us with 2.33 us spread; peak
allocation was 16,977,428 bytes with zero spread. These values are descriptive
evidence for this machine, not acceptance limits.

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
