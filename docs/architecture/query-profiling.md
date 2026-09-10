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

Measured on Apple M1 Max, macOS arm64, rustc 1.98.1, using the release command
above and the query implementation in `ad1d3c7`. No concurrent build was
running. Values below are median microseconds per evaluation.

| Case | 113 operations | 897 operations | 7,169 operations |
| --- | ---: | ---: | ---: |
| Direct name scan | 0.44 | 3.52 | 31.57 |
| Count | 0.62 | 2.82 | 21.61 |
| Name filter | 1.39 | 9.41 | 71.06 |
| Boolean filter | 3.78 | 29.05 | 227.09 |
| Users | 14.11 | 109.80 | 880.43 |
| Union of navigations | 32.84 | 247.10 | 1,985.23 |
| Root and filter | 17.71 | 135.82 | 1,102.11 |
| One closure step | 9.37 | 73.39 | 584.65 |
| Fixed-point closure | 50.42 | 390.47 | 3,121.13 |
| Emit, users, count | 8.33 | 64.18 | 496.60 |

Query parsing medians ranged from 0.14 to 1.52 microseconds. The name-filter
pipeline cost about 2.3 times the direct scan at the largest width, including
selection construction and scalar emission. Width scaling was approximately
linear for these cases; navigation and set composition cost more than a scan.

Deep dependency slices show a different limit:

| Additions in one chain | Total operations | Fixed-point closure, median ms |
| ---: | ---: | ---: |
| 16 | 20 | 0.035 |
| 128 | 132 | 1.725 |
| 1,024 | 1,028 | 108.059 |

Each fixed-point iteration reapplies the query to the entire current selection.
Closure also scans document order when forming its result. On a chain, the
selection grows one dependency step at a time, so this repeats increasing work
and produces approximately quadratic growth. Deep slices were therefore
substantially more expensive than shallow ones in this run. Use these depth
cases when evaluating optimizations, and check that fixed-point semantics, error
handling, and per-iteration emissions still hold.
