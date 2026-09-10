# Query-language profiling

The opt-in `query_profile` integration test measures query parsing and
execution separately. It uses the release profile, a pre-lowered document,
and a warmed evaluator. It reports timings without enforcing a machine-dependent time limit.

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

- **Parsing:** construct and drop each query, independently of document size.
- **Evaluation:** execute a pre-parsed query on a pre-lowered document. Cases
  cover count, exact-name and boolean filters, users, union, subtree, one closure
  step, a fixed point, and an intermediate emission.
- **Direct filter:** scan operation names through the semantic API, without
  the query interpreter, as a reference for filtering overhead.
- **Deep fixed points:** increase dependency depth in one function, separately
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

Use runs on the same machine and compiler to compare revisions. There are no
portable timing thresholds: the direct-filter comparison and scaling across
width and depth are more useful than a pass/fail deadline.

## Recorded measurements

Measured on Apple M1 Max, macOS arm64, rustc 1.98.1, on 2026-09-10 after
the query-review fixes to `d0c9705`. No concurrent build or test was running.
Values below are median microseconds per evaluation.

| Case | 113 operations | 897 operations | 7,169 operations |
| --- | ---: | ---: | ---: |
| Direct name scan | 0.42 | 3.31 | 31.40 |
| Count | 0.60 | 2.58 | 20.34 |
| Name filter | 1.48 | 9.67 | 74.95 |
| Boolean filter | 4.04 | 29.62 | 235.00 |
| Users | 4.11 | 30.04 | 233.74 |
| Union of navigations | 16.30 | 119.04 | 942.29 |
| Subtree and filter | 6.28 | 45.48 | 356.16 |
| One closure step | 11.02 | 86.26 | 680.16 |
| Fixed-point closure | 20.78 | 160.73 | 1,297.54 |
| Emit, users, count | 3.36 | 24.39 | 189.95 |

The name-filter pipeline cost about 2.4 times the direct scan at the largest
width, including selection construction and scalar emission. Width scaling
was approximately linear for these cases. `users` counts use sites, including
the first addition's two uses of its constant; its expected count is
`width * (depth + 1)`.

Deep dependency slices now use a worklist:

| Additions in one chain | Total operations | Fixed-point closure, median ms |
| ---: | ---: | ---: |
| 16 | 20 | 0.0046 |
| 128 | 132 | 0.0316 |
| 1,024 | 1,028 | 0.2440 |

Before the worklist change, the same 1,024-addition case took 117.29 ms in the
review run: repeated whole-selection evaluation produced approximately quadratic
growth. Pure `fixpoint(closure)` now visits each dependency once and forms the
source-ordered result once. Scope retention also avoids revisiting subtrees
already expanded during that evaluation. One closure step has a small additional
bookkeeping cost; in the 7,169-operation case it increased from 614 to 680 µs.

Arbitrary fixed points, including `fixpoint(closure | emit)`, still execute each
iteration to preserve query and emission semantics. Those queries can remain
quadratic on a deep chain. Evaluation work and stream-size limits prevent
unbounded duplicate growth; see the [language reference](../query-language.md).

The [CLI stress harness](../../python/benchmarks/query_language_review.py) includes
process startup, parsing, lowering, and printing/counting. With one warmup and
three measured subprocess runs, its 8,192-addition closure fell from 7.39 seconds
to 50.24 ms. Counting the same input took 47.83 ms. These end-to-end numbers
should not be compared directly with evaluator-only timings.
