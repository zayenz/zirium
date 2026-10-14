# Query-language profiling

The opt-in `query_profile` integration test measures query parsing and
execution separately on a pre-lowered document with a warmed release evaluator.
It reports timings without a pass/fail deadline.

```sh
# Quick check of the benchmark and its expected results.
ZIRIUM_QUERY_PROFILE_SMOKE=1 cargo test --release -p zirium --test query_profile -- --ignored --nocapture --test-threads=1

# Full width and dependency-depth measurements.
cargo test --release -p zirium --test query_profile -- --ignored --nocapture --test-threads=1
```

The test is ignored by normal `cargo test` runs and rejects debug builds.
It needs no benchmark dependency or external profiling tool. Output rows use
`key=value` fields, so a run can be saved and compared with a later revision.

## What is measured

- Parsing constructs and drops each query, independently of document size.
- Evaluation runs a pre-parsed query on a pre-lowered document. Cases
  cover count, exact-name and boolean filters, users, union, subtree, one closure
  step, a fixed point, and an intermediate emission.
- A direct filter scans operation names through the semantic API, without
  the query interpreter, as a reference for filtering overhead.
- Deep fixed-point cases increase dependency depth in one function, separately
  from increasing the number of independent functions.

Fixtures contain a module and independent functions. Each function has one
constant, a chain of additions, and a return. The last addition is tagged as
its dependency-slice seed. Width runs use 16, 128, and 1,024 functions, each
with four additions: 113, 897, and 7,169 operations in total. Depth runs use
one function with 16, 128, or 1,024 additions. The smoke run uses smaller sizes.
Expected operation and emitted-item counts are checked before timing.

Calibration doubles the batch size until it takes at least 5 ms; these calls
also warm the code and any lazy document indexes. Each measured sample uses
that batch size. The full run reports the minimum, median, and maximum of
seven samples as nanoseconds per query; smoke mode uses three samples.
`black_box` keeps the query inputs and observed outputs in the measured work.

MLIR parsing, lowering, query parsing during evaluation, correctness checks,
and fixture construction are outside evaluation timing. Emission callbacks
consume selection sizes or scalar counts: these measurements include query
selection allocation and callback overhead, but exclude MLIR formatting and
I/O. Editing is covered by the existing processing benchmark's editor stages.
This test does not measure peak memory or cold-start latency.

Compare revisions on the same machine and compiler. Use the direct-filter
baseline and width/depth scaling to interpret costs.

## CLI queries

The [query harness](../../python/benchmarks/query_benchmark.py) checks generated
StableHLO graphs, query composition, diagnostics, and reports. Its timings include
process startup, MLIR parsing, lowering, evaluation, and output. Compare them
separately from the evaluator-only measurements above.

```sh
cargo build --release -p zirium --bin zirium
mkdir -p target/query-benchmark
python3 python/benchmarks/query_benchmark.py > target/query-benchmark/results.json
```

The harness uses only the Python standard library and creates temporary fixtures
and query files. Inspect its JSON results for failed checks and timeouts.

Pure `fixpoint(closure)` uses a worklist. Other fixed points run each iteration
to preserve query and emission semantics, so deep chains can still be costly.
See the [language reference](../query-language.md#closure-and-fixed-points) for
evaluation limits.

## Bounded shared report comparison

A small report comparison uses the 333-byte
[function fixture](../../examples/cli/function-report.mlir) and its
[report program](../../examples/cli/function-report.zirql). The program projects
native functions, filters its constructed rows with a comprehension, and reports
`functions`, `total_operations` and `total_matches`. It selects `double` and
`mix`, with totals 5 and 2. Three separate controls report one field each and
retain only that field's required bindings and projections. Native constructed
fields and rendered JSON agree; every command exits 0 with empty stderr.

The source is Zirium `c467e7dc6cfe9f8f5a1afe2eeb15b46ab2788d69`, with Zirql
pinned at `cef521687d81a1f23ab909cf2ea2e7d54a52784c`. The release command was
rebuilt from that source with `cargo build --release --locked -p zirium --bin zirium`.
The host is an Apple M1 Max, 10 cores and 64 GB memory, running arm64 macOS
26.6.2 (25G83) and Rust 1.98.1 (`48a229cea`). Limits retain the CLI defaults:
10,000,000 work units, 1,000,000 aggregate results, 1,000,000 items per stream
and nesting 64.

Seven fresh-process pairs ran serially in one quiet window. Odd pairs ran the
program first; even pairs ran the controls first. Controls stay in field order.
All 14 samples and 28 children completed, with no errors, timeouts, incomplete
results or unfinished samples. Each child had a ten-second deadline. Complete
stdout, stderr and statuses remain with every sample.

| Variant | First sampled wall ms | Median wall ms (min..max) | Median peak child RSS MiB (min..max) |
| --- | ---: | ---: | ---: |
| One report program | 6.46 | 5.80 (5.57..6.46) | 4.02 (3.98..4.05) |
| Three separate requests | 16.46 | 17.36 (16.18..17.75) | 3.98 (3.97..4.02) |

Wall time includes process launch, input/query reads and parsing, lowering,
evaluation, rendering and output capture. A control group's wall time also
includes capture bookkeeping between its sequential children. Exact-child
`os.wait4` records peak RSS in bytes on macOS, converted to MiB here; the group
value is the largest child peak, not a sum. The unchanged Zincite
`bench-save.py::file_invoke` helper and its existing `waitid` bridge keep the
child reserved for that reaper. These are warm-filesystem observations after
build and correctness probes. They describe this small workload and startup
cost, not controlled cold I/O or evaluator-only throughput. The existing
`query_profile` measurements above keep their separate timing boundary.

### Native retention observations

A separate untimed Rust probe parses each query once and lowers one document
with the CLI's `Hybrid` retention profile. Its first and ten repeated native
reports have equal constructed fields and JSON. The retained source and CST
identities stay unchanged. Each repeated output and native result is dropped
before the next request; the final result and output are then dropped explicitly.

`Document::statistics` reports the following byte boundaries for this fixture:

| Scope | Source bytes | CST bytes | Document index bytes | Use-index entries |
| --- | ---: | ---: | ---: | ---: |
| Lowered document, before reports | 333 | 2,808 | 480 | 0 |
| After first report and ten repeated result/output drops | 333 | 2,808 | 480 | 0 |
| After first `users` query and ten repeated result/output drops | 333 | 2,808 | 1,824 | 9 |
| After `ParsedFile` is dropped, document still live | 333 | 2,808 | 1,824 | 9 |

The report does not demand the use index. The separate
`filter(op("arith.muli")) | users | names` query demands it once; subsequent
requests retain the same entry count and reported storage. Direct document-owned
storage is 8,920 bytes throughout. Source and CST storage are shared with
`ParsedFile` while it lives; after its drop, the document still retains both and
the shared-storage flags become false.

Two independently parsed and lowered copies produce equal report fields but
have distinct source storage and operation handles. Each document refuses the
other's handle. Dropping the first document and all its results leaves the second
usable, with its original storage statistics. The probe then drops the second
result/output, `ParsedFile`, document and parsed queries explicitly. No allocation
observer was installed: these statistics do not measure total live heap, allocator
metadata or memory returned to the OS. RSS is a process high-water value and
cannot establish release after a drop. An initial observer incorrectly expected
source retention from `SemanticOnly`; its failure is retained separately, and
only the explicit `Hybrid` observation above supplies these storage claims.

Raw samples, complete streams, source/binary/input/program hashes, environment,
native observations and analysis remain under
`/private/tmp/zirql010-author-zirium`. The private capture runner has named
`run`/`analyze` commands; its final smoke checks cover preview, resume of only
unfinished work, unchanged completed records, configuration mismatch, failure,
timeout and analysis from saved records. This comparison changes no product
source, installed command, benchmark driver, dependency pin or Python extension.
