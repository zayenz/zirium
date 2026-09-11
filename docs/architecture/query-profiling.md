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
