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
  step, a fixed point, an intermediate emission, and an ordinary report with no
  function declarations.
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

## Explicit native worklists

A separate release comparison measures the native worklist adapter, retained
aliases and growing scan results. The compiled host source is
`204d01024afc7c2f40e3cf61928bfccc29f33148`, with Zirql pinned to the actual Git
revision `ea30035e010cea2e12047a1aa4e5bf39a3ac983e`. The unchanged-query baseline is
host `7d5f56d5da990952de2fb992ff58a0de7fd3272f`, pinned to Zirql
`4eaaafb7cfc112dbc0673fd67007a16f1f8e8ee2`. Both use their locked dependencies,
Rust 1.98.1 (`48a229cea`, LLVM 22.1.8), and the same Apple M1 Max with 10 cores,
64 GiB RAM and arm64 macOS 26.6.2. The final compiled-file identity is SHA-256
`6a6dac05f84afae1fd351b1d6fd32e2736d59d5f94837d3c1bbc14a09a778cd4`.

The private probe reuses the existing pre-lowered query profiling boundary.
Generated valid SSA contains one start, `n` same-named nodes each consuming the
start twice, one end consuming every node, and a disconnected target. There
are `n + 3` operations and `3n` uses. The
[complete reachability program](../../examples/cli/worklist-reachability.zirql)
provides the traversal function; the measured request asks whether the start
reaches the disconnected target, so it exhausts the reachable graph. Correctness
warmup checks that the answer is false. The alias case drains a retained queue
and returns its count with the unchanged original queue. The scan case appends
one native operation per step and retains every queue snapshot; its first and
last sizes are checked. These are native results, without JSON conversion.

Each row has five fresh-process batches. Baseline and current small-query
batches alternate order. A small-query batch has 500 requests; a warm worklist,
alias or scan batch has three. A cold-index batch has one request. Parsing
constructs and drops the query. Evaluation includes request preparation,
evaluation, the native callback and drops between requests; the final result
stays live for the retention checkpoint and is dropped after timing. Fixture
construction, MLIR parsing/lowering, query parsing and correctness checks are
outside evaluation timing. Only the small report includes JSON rendering.
A separate document provides the cold case's correctness warmup: “cold” means
this document's use index is unbuilt, not cold process startup or filesystem I/O.

The probe explicitly uses 1,000,000,000 work units, 100,000,000 aggregate results,
100,000 items and nesting 64. Product defaults are unchanged. The main campaign
completed all 165 success samples and two separate budget-refusal controls,
with no failed, timed-out or unfinished captures.

### Unchanged small queries

The 19-operation fixture uses `count`, `filter(op("graph.node")) | count` and
`report {"sum": 2 + 3, "items": [1, 2, 3]} | json`. Times below are microseconds
per request: median, with minimum–maximum in parentheses. Evaluation uses a
pre-parsed query; the report includes its JSON emitter.

| Query | Phase | Baseline µs | Current µs | Median change |
| --- | --- | ---: | ---: | ---: |
| count | parse | 0.351 (0.347–0.362) | 0.351 (0.347–0.359) | +0.2% |
| count | evaluate | 1.392 (1.358–1.431) | 1.365 (1.345–1.399) | -1.9% |
| filter | parse | 1.249 (1.234–1.262) | 1.281 (1.260–1.310) | +2.5% |
| filter | evaluate | 2.428 (2.419–2.460) | 2.415 (2.404–2.423) | -0.5% |
| report | parse | 3.358 (3.329–3.362) | 3.677 (3.666–3.764) | +9.5% |
| report | evaluate | 8.191 (8.179–8.263) | 8.248 (8.225–8.447) | +0.7% |

The report's 9.5% parsing increase costs 0.319 µs on this input. Inspection
found that the host now scans ordinary report suffixes to preserve the `| json`
boundary with the newer common reader. The comparison also includes added
common-reader recognition, so it does not isolate either cost. The pre-parsed
report's evaluation median differs by 0.7% and remains within the baseline's
observed range.

### Increasing native workloads

Times are milliseconds per request, median (minimum–maximum). Cold and warm
adjacency cases remain separate; their overlapping ranges do not establish a
reliable isolated use-index cost.

| Nodes `n` | Reachability, cold index | Reachability, warm index | Retained alias | Retained scan |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 3.315 (3.294–3.339) | 3.271 (3.239–4.359) | 0.557 (0.551–0.593) | 0.427 (0.423–0.435) |
| 512 | 13.040 (12.907–13.532) | 12.819 (12.729–13.866) | 2.095 (2.079–2.213) | 1.661 (1.630–1.673) |
| 2,048 | 51.727 (51.532–52.587) | 51.752 (51.331–53.316) | 8.294 (8.188–8.385) | 6.568 (6.514–7.054) |

Across a 16-fold increase in node count, warm reachability takes 15.8 times as long,
retained aliases 14.9 times and retained scans 15.4 times. These three sizes
support roughly proportional growth for this wide, duplicate-route graph;
they do not characterize every graph shape.

### Allocation and result drops

A separate release binary observes requested `System` allocation sizes. These
observations do not supply the clean timing rows above. Each sample runs one
warm request. “Retained” and “peak” are extra live KiB above the document/query
checkpoint. Every result drop restores that checkpoint exactly. RSS is the
exact child's process peak from macOS `os.wait4`, in MiB, and includes setup and
warmup. It cannot show memory release after a drop.

| Case | `n` | Retained KiB | Peak extra KiB | After-result excess bytes | Peak child RSS MiB, median (min–max) |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph | 128 | 0.29 | 372.34 | 0 | 5.55 (5.55–5.58) |
| graph | 512 | 0.29 | 1,375.10 | 0 | 8.00 (7.95–8.38) |
| graph | 2,048 | 0.29 | 5,276.92 | 0 | 17.45 (16.80–19.69) |
| alias | 128 | 18.58 | 58.14 | 0 | 5.02 (5.00–5.03) |
| alias | 512 | 72.58 | 196.14 | 0 | 6.50 (6.48–6.59) |
| alias | 2,048 | 288.58 | 748.14 | 0 | 11.80 (11.22–12.91) |
| scan | 128 | 36.16 | 105.31 | 0 | 5.05 (5.05–5.05) |
| scan | 512 | 144.16 | 393.31 | 0 | 6.67 (6.64–7.11) |
| scan | 2,048 | 576.16 | 1,545.31 | 0 | 12.75 (11.70–13.86) |

Retained and after-result-drop counts match across all five samples. Alias and
scan peak allocations also match. Graph peak-extra ranges are 371.60–372.64 KiB
at 128 nodes, 1,373.47–1,375.40 KiB at 512 and 5,274.69–5,280.63 KiB at 2,048.
The table reports their medians; full live checkpoints remain in the raw data.

Keeping every prefix scan state therefore retains about 288 bytes per item in
this case, rather than the sum of all queued prefix lengths. After dropping the
query, document, fixture and result, the original samples retain a constant
268 bytes above the initial process checkpoint. A separate control lowers and
drops one document before that checkpoint; its final excess is zero. This is
consistent with the existing document-identity table retaining initial capacity,
and is separate from request results.

With work reduced to 50,000, the 512-node graph refuses with a query range at
`474..495`. The allocator-observed refusal reaches 1,095,317 live bytes and,
after all request/document data are dropped, leaves 375 bytes above its initial
checkpoint: the same 268-byte startup difference plus the retained 107-byte
diagnostic string capacity. Refusals are not successful throughput samples.

Complete inputs and hashes, commands, streams, individual samples, binary and
source identities, compiler details, allocation checkpoints and the focused
controls remain under `/private/tmp/zirql045-measure`. The named main run is
`final-ea30035-204d010-v2`; `collect-v2.py analyze --name final-ea30035-204d010-v2`
regenerates its analysis without invoking a target. The private collector has
no product dependency or installation effect.

## Prepared report execution

Every textual and typed request owns one prepared function unit, including an
empty declaration list. Zirium prepares shared report entries, projection fields
and native argument bodies before statement execution. Those entries run with
the current lexical bindings, relative input and request budget. Type discovery
uses the same function-aware kind rules and Zirium's scalar projection facts.
A report-only stage requires constructed data; typed value entries may retain
native handles.

The ordinary textual report frontend normalizes saved names to scalar query
projections. Function-capable bodies retain native value queries so their calls
can pass checked handles. These are admission policies on shared value nodes;
both use the same prepared runner.

Zirium keeps statement sequencing, bindings, native stage dispatch and emission.
Native selections and counts retain their existing work and stream bounds.
Overloaded collection stages retain a lowered constructed body in their stage
payload. Preparation checks that body only for constructed input, so borrowed
native argument execution can use the same body without rebuilding syntax.

### Small-query comparison

The smoke command above ran before and after this consolidation on arm64 macOS
with Rust 1.98.1 (`48a229cea`), the same release profile, fixtures and arguments.
The baseline is Zirium `16fee69fdd93411b87564c9a95e065853d913cb4`; both builds
use Zirql `967eb05c80fd122fb5a91cd9056cce124f844150`. The added harness case is
`report {"sum": 2 + 3, "items": [1, 2, 3]} | json`, with no declarations.
All cases passed their output checks in both runs.

Times are microseconds per query: median, with the three-sample minimum–maximum
in parentheses. Evaluation includes request preparation and result handling;
the report also includes JSON rendering.

| Case | Phase | Operations | Before µs | Prepared path µs |
| --- | --- | ---: | ---: | ---: |
| report | parse | — | 6.095 (6.093–6.112) | 6.357 (6.326–6.358) |
| count | parse | — | 0.390 (0.389–0.393) | 0.628 (0.623–0.633) |
| report | evaluate | 29 | 8.631 (8.627–8.638) | 9.744 (9.720–9.745) |
| report | evaluate | 225 | 10.127 (10.119–10.220) | 11.116 (11.100–11.376) |
| count | evaluate | 29 | 1.293 (1.292–1.295) | 1.458 (1.445–1.470) |
| filter | evaluate | 29 | 4.554 (4.554–4.562) | 4.835 (4.756–4.838) |
| count | evaluate | 225 | 3.142 (3.133–3.156) | 3.357 (3.340–3.360) |
| filter | evaluate | 225 | 24.749 (24.411–24.880) | 24.455 (24.453–24.542) |
| direct filter | evaluate | 29 | 0.110 (0.110–0.110) | 0.112 (0.111–0.112) |

The first prepared-path run took 11.253 µs (11.222–11.336) for the 29-operation
report. Inspection found repeated kind discovery immediately after preparing
that entry. Reusing its prepared kind for stage type flow reduced the median
to 9.744 µs; final whole-expression validation still checks composition rules.
The remaining report increase is about 1.1 µs on this small input. Native count
evaluation adds about 0.17 µs. Count parsing retains its constructed overload
body and costs about 0.24 µs more. The larger filter's ranges overlap; these
three-sample smoke observations do not establish a general throughput bound.

Complete command output remains in `profile-before.log`, `profile-after-initial.log`
and `profile-after.log` under `/private/tmp/zirql-parallel-20261021/base-116`.
The unchanged correlated report controls also retain native fields and rendered
results for `double` and `mix`, with operation and match totals 5 and 2.
