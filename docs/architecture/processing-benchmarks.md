# Processing benchmarks

The benchmark harness measures parsing, semantic processing, output, and Python
access on deterministic MLIR fixtures. The results below come from several
recorded runs and implementation revisions. They are useful for comparisons on
the same hardware and compiler; they are not performance guarantees for the
current checkout.

## Parser design

The parser stores transient events in four bytes, compacts them directly into a
flat CST, and reuses the lexer's token vector. It releases the event buffer
before trimming the completed node vector. This internal path checks that parser events reference every
token once in source order. The public event constructor also accepts
arbitrary token-event order, so it keeps a token copy and a bitmap to check
indices, duplicates, omissions, source order, and root coverage.

The recorded measurements support keeping the event parser and flat CST.
Syntax-dense inputs account for much of the cost. Adding parser-level string
IDs increased estimated memory use, and operation-name lookup did not dominate
event production. Source ranges remain the token-text representation.

Python packed tables fill their final `bytes` allocations directly. Their
construction time is close to equivalent native payload builders, though
several cases exceed three times the cost of a native walk that produces no
payload. The sections below distinguish these comparisons.

## Run the full pipeline benchmark

`processing_benchmark` generates deterministic, exact-size MLIR fixtures in
temporary storage with seed `0x5a495249554d0028`. Unless a section states
otherwise, measurements used Apple M1 Max, Darwin 25.5.0,
`aarch64-apple-darwin`, a release build, and rustc 1.97.1. Results are medians
of three runs after one warm-up. Peak allocation is the increase in
process-wide live allocation above the stage baseline.

```sh
cargo build --release -p zirium --example processing_benchmark
for size in 1 10 25 50 75 100; do
  target/release/examples/processing_benchmark --size-mib "$size" --shape primary --stage parse,traverse,syntax-payload,lower,canonical,use-index,symbol-index,dominance-index,editor,preserving --warmups 1 --runs 3
done > /tmp/zirium-primary.txt
target/release/examples/processing_benchmark --report < /tmp/zirium-primary.txt

for depth in 8 64 256; do
  target/release/examples/processing_benchmark --size-mib 10 --shape nested --depth "$depth" --stage parse,traverse,lower --warmups 1 --runs 3
done

target/release/examples/processing_benchmark --size-mib 10 --shape block-rich --stage parse,lower,verify,dominance-index --warmups 1 --runs 3
target/release/examples/processing_benchmark --size-mib 500 --shape primary --stage parse,lower --warmups 1 --runs 3

PYTHONPATH=python python3 python/benchmarks/processing_benchmark.py --smoke
PYTHONPATH=python python3 python/benchmarks/processing_benchmark.py --size-mib 10 --runs 3
```

## Parser construction and string storage

The ignored crate-internal test separates lexing, grammar event production, CST
compaction, and structural verification without adding a public profiling API.
Run it in release mode with one test thread:

```sh
ZIRIUM_PARSER_BENCH_SMOKE=1 cargo test --release -p zirium parser_construction_benchmark::measure_parser_construction_and_string_identity -- --ignored --nocapture --test-threads=1

ZIRIUM_PARSER_BENCH_SHAPE=primary ZIRIUM_PARSER_BENCH_SIZE_MIB=10 ZIRIUM_PARSER_BENCH_RUNS=3 cargo test --release -p zirium parser_construction_benchmark::measure_parser_construction_and_string_identity -- --ignored --nocapture --test-threads=1
ZIRIUM_PARSER_BENCH_SHAPE=primary ZIRIUM_PARSER_BENCH_SIZE_MIB=100 ZIRIUM_PARSER_BENCH_RUNS=3 cargo test --release -p zirium parser_construction_benchmark::measure_parser_construction_and_string_identity -- --ignored --nocapture --test-threads=1
ZIRIUM_PARSER_BENCH_SHAPE=block-rich ZIRIUM_PARSER_BENCH_SIZE_MIB=10 ZIRIUM_PARSER_BENCH_RUNS=3 cargo test --release -p zirium parser_construction_benchmark::measure_parser_construction_and_string_identity -- --ignored --nocapture --test-threads=1
```

The smoke fixture is 64 KiB with one warm-up and one run. The other controls
default to a 10 MiB primary fixture, one warm-up, and three runs;
`ZIRIUM_PARSER_BENCH_WARMUPS` and `ZIRIUM_PARSER_BENCH_RUNS` override those
counts. These measurements used the same machine described above: Apple M1 Max,
Darwin 25.5.0, `aarch64-apple-darwin`, release profile, rustc 1.97.1. Fixtures
are deterministic and generated in memory; no generated fixture is retained in
the repository.

Each run first reports an integrated `parser_whole` row with the input size,
token and node counts, exact retained CST bytes, and peak allocation for the
complete parse. It then reports the individual construction phases. Values are
medians of three runs. Peak allocation is incremental above a baseline taken
while the input and prerequisite outputs are live. Lexing retains the source;
event production retains the source and token tape; compaction also retains the
event tape and measures the input clones and working allocations required by
destructive compaction; verification retains the completed unverified CST. The
measured output remains live through a black-box observation. The phase peaks
describe separate lifetimes and are not additive.

The integrated rows from this run were:

| fixture | input MiB | median ms | peak live MiB | tokens | nodes | retained CST MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| primary 10 MiB | 10 | 17.862 | 30.62 | 445,433 | 92,801 | 7.10 |
| block-rich 10 MiB | 10 | 125.355 | 269.85 | 3,936,313 | 1,864,570 | 77.05 |

The base-041 revision measured construction phases separately:

| fixture | phase | median ms | incremental peak live MiB | output items |
| --- | --- | ---: | ---: | ---: |
| primary 10 MiB | lex | 8.805 | 6.00 | 445,433 tokens |
| primary 10 MiB | events | 3.049 | 12.00 | 631,035 events |
| primary 10 MiB | compact | 8.224 | 27.06 | 92,801 nodes |
| primary 10 MiB | verify | 0.501 | 0 | 92,801 nodes |
| primary 100 MiB | lex | 83.079 | 96.00 | 4,454,153 tokens |
| primary 100 MiB | events | 39.493 | 96.00 | 6,310,055 events |
| primary 100 MiB | compact | 84.479 | 266.62 | 927,951 nodes |
| primary 100 MiB | verify | 5.094 | 0 | 927,951 nodes |
| block-rich 10 MiB | lex | 25.487 | 48.00 | 3,936,313 tokens |
| block-rich 10 MiB | events | 34.143 | 96.00 | 7,665,453 events |
| block-rich 10 MiB | compact | 142.525 | 301.30 | 1,864,570 nodes |
| block-rich 10 MiB | verify | 9.943 | 0 | 1,864,570 nodes |

In base-042, direct compaction removed the second complete event vector.
Forward-parent chains use one reusable buffer, and each normalized event is
compacted immediately. The same release commands on the same machine produced:

| fixture | compact median ms | base-041 / base-042 throughput MiB/s | throughput change | incremental peak live MiB | change from base-041 peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| primary 10 MiB | 4.503 | 1,215.95 / 2,220.64 | +82.6% | 19.84 | -26.7% |
| primary 100 MiB | 46.467 | 1,183.73 / 2,152.08 | +81.8% | 194.41 | -27.1% |
| block-rich 10 MiB | 68.244 | 70.16 / 146.53 | +108.8% | 213.57 | -29.1% |

Direct compaction roughly halved compaction time and reduced peak allocation
across both fixture shapes.

The base-043 comparison measured the internal path that reuses parser tokens.
Parser events reference every lexer token once in source order, so this path
moves the lexer token vector into the CST and checks the sequential-index
invariant without allocating a second token vector or a `seen` bitmap. The
public `SyntaxTree::from_events` constructor is unchanged. It still accepts
arbitrarily ordered token events and checks invalid indices, duplicates, source
order, omissions, and root coverage.

The table compares whole parsing immediately before and after the change. Both
sides used the same generated fixture, release build, warm-up count, and
three-run median. Shrinking the moved token vector keeps retained CST size
unchanged.

| fixture | before ms | after ms | time change | before peak MiB | after peak MiB | peak change | retained CST MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| primary 10 MiB | 17.367 | 13.582 | -21.8% | 30.62 | 19.10 | -37.6% | 7.10 |
| block-rich 10 MiB | 127.976 | 118.075 | -7.7% | 269.85 | 173.05 | -35.9% | 77.05 |

Reusing parser tokens reduced peak allocation and parse time for both shapes.
The CST still owns the token vector.

The string-storage experiment counted identifier tokens and quoted operation
names. Quoted names counted only when they were direct children of an operation
node, excluding strings in attributes or nested regions.

Each occurrence already needs an eight-byte source range. The candidate added
a four-byte ID per occurrence. Each unique spelling also required a four-byte
ID, an eight-byte hash, two machine-word range or pointer fields, and the
spelling bytes. The original source still had to be retained. The estimate
excluded hash-table control bytes and spare capacity, so an implementation
would use somewhat more memory.

| fixture | class | frequency / unique | current range KiB | candidate range + ID/interner KiB |
| --- | --- | ---: | ---: | ---: |
| primary 10 MiB | bare | 18,558 / 1 | 145.0 | 217.5 |
| primary 10 MiB | operation string | 18,560 / 3 | 145.0 | 217.6 |
| primary 10 MiB | percent / caret | 1 / 1 each | less than 0.1 | less than 0.1 |
| primary 100 MiB | bare | 185,588 / 1 | 1,449.9 | 2,174.9 |
| primary 100 MiB | operation string | 185,590 / 3 | 1,450.0 | 2,175.0 |
| primary 100 MiB | percent / caret | 1 / 1 each | less than 0.1 | less than 0.1 |
| block-rich 10 MiB | bare | 345,291 / 5 | 2,697.6 | 4,046.6 |
| block-rich 10 MiB | at | 69,058 / 69,058 | 539.5 | 3,158.8 |
| block-rich 10 MiB | percent | 138,116 / 1 | 1,079.0 | 1,618.6 |
| block-rich 10 MiB | caret | 345,290 / 3 | 2,697.6 | 4,046.5 |
| block-rich 10 MiB | operation string | 69,058 / 1 | 539.5 | 809.3 |

The exclamation and hash classes had no occurrences in these fixtures. The
representative prebuilt hash-table lookup pass took 0.934 ms for 37,120
primary-10 lookups, 14.781 ms for 371,180 primary-100 lookups, and 21.881 ms for
966,813 block-rich lookups. This experiment measured lookup cost only.

The storage estimate keeps the eight-byte source range required for lossless
syntax, then adds IDs and an interner. It adds about 1.42 MiB on the repetitive
100 MiB primary fixture before hash-table overhead, and about 5.98 MiB on the
block-rich fixture, which has 69,058 unique symbol names.

Operation dispatch compares bare names with registered descriptors. The other
parser comparisons use a small vocabulary of grammar keywords; typed syntax
views compare node kinds. The event-production measurements did not identify
name lookup as a dominant cost, so the study did not justify charging every
identifier for an ID. Semantic lowering has separate document-local interning.
Consumers that need identity tables can measure that tradeoff for their own
access patterns.

## Packed Python syntax tables

The base-045 measurement used the same deterministic 10 MiB primary fixture,
seed, release profile, one warm-up, and three measured runs as the processing
baseline above. The native traversal took 0.171 ms. Constructing the same eight
packed payload columns in the Rust counting-allocator harness took 2.102 ms,
produced 6,489,810 payload bytes, and added 6,489,810 bytes at peak. This stage
retains its eight final Rust vectors only for the measurement.

The release Python `SyntaxTable` construction took 1.984 ms and returned the
same 6,489,810 payload bytes. Tracemalloc reported 6,490,218 bytes of retained
growth, or 1.00006 times the payload, below the study's 1.5-times
retained-memory threshold. Each final Python `bytes` allocation is filled in
place with `PyBytes::new_with`; no Rust temporary column buffer is allocated, so
there is no material transient column duplication.

Python construction took 11.6 times as long as native traversal, exceeding the
study's three-times comparison threshold. Most of that comparison is
the necessary encoding and allocation of 6.49 MB rather than traversal alone:
Python construction was 0.94 times the matching native payload-construction
stage. These figures are observations on the recorded machine, not performance
guarantees.

`--depth` is required for nested fixtures and rejected for other shapes. The
generator raises the parser delimiter limit to cover the selected depth; padding
still makes every fixture exactly the requested byte count. Release smoke checks
covered primary, block-rich, trivia, payload, and nested depths 8, 64, and 256.

## Recorded parser scaling

This run used direct compaction. It predates the token-reuse comparison above.
The measurement machine was
an Apple M1 Max with 64 GiB RAM, Darwin 25.5.0, `aarch64-apple-darwin`, release
profile, and rustc 1.97.1. Each row reports the median of three measured runs
after one untimed warm-up.
Peak allocation is incremental process-scoped live allocation above the stage
baseline. Retained bytes are `SyntaxTree::exact_retained_bytes()` and exclude
the separately owned source. Token and node rates are derived from the exact
reported counts and median latency. Rates are observations for planning, not
test thresholds or public performance guarantees.

| fixture | median ms | MiB/s | peak MiB | retained MiB | million tokens/s | million nodes/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| primary 1 MiB | 1.522 | 656.832 | 3.68 | 0.82 | 32.495 | 6.773 |
| primary 10 MiB | 16.623 | 601.587 | 31.79 | 7.66 | 29.746 | 6.197 |
| primary 25 MiB | 43.508 | 574.601 | 81.47 | 18.15 | 28.410 | 5.919 |
| primary 50 MiB | 90.572 | 552.048 | 162.94 | 36.29 | 27.295 | 5.686 |
| primary 75 MiB | 140.784 | 532.729 | 248.41 | 58.44 | 26.339 | 5.487 |
| primary 100 MiB | 181.661 | 550.476 | 325.88 | 72.58 | 27.217 | 5.670 |
| block-rich 10 MiB | 142.671 | 70.091 | 269.85 | 77.05 | 27.590 | 13.069 |
| nested 10 MiB, depth 8 | 6.690 | 1,494.861 | 10.00 | 0.003 | 0.026 | 0.005 |
| nested 10 MiB, depth 64 | 6.610 | 1,512.764 | 10.00 | 0.023 | 0.196 | 0.039 |
| nested 10 MiB, depth 256 | 6.999 | 1,428.733 | 10.00 | 0.090 | 0.734 | 0.147 |
| payload 100 MiB | 88.757 | 1,126.677 | 100.00 | 0.0004 | 0.0003 | 0.00008 |
| trivia 100 MiB | 98.427 | 1,015.981 | 162.67 | 32.00 | 28.409 | 0.00004 |

### Comparison with base-039

The 100 MiB primary result is the direct comparison with base-039. Median
latency fell from 234.975 ms to 181.661 ms (22.7%), and peak live allocation
fell from 406.04 MiB to 325.88 MiB (19.7%). The resulting CST contains
4,944,201 tokens and 1,030,046 nodes in 76,107,628 retained bytes. These counts belong to this run.

The direct 500 MiB primary parse ran with 64 GiB RAM and 295 GiB free temporary
storage. It took 941.563 ms at 531.032 MiB/s, with 1,960,199,833 bytes (1,869.39 MiB) peak
live allocation and 430,868,492 bytes (410.91 MiB) retained for 24,720,897
tokens and 5,150,191 nodes. That is 26.255 million tokens/s and 5.470 million
nodes/s. Compared with the base-039 direct 500 MiB result of 1.235528 s and
2,380,455,133 bytes peak, latency fell 23.8% and peak allocation fell 17.7%.
It also remains consistent with the earlier base-039 projection of 1.184 s and
2,030.3 MiB peak.

### What the scaling shows

Syntax density accounts for the largest differences in these measurements.
At 10 MiB the block-rich fixture is about 8.6 times slower than primary and
retains about ten times as many CST bytes, while its token rate remains close
to primary. Payload, trivia, and exact-depth nested fixtures stay
scanning-dominated. Primary throughput and normalized token/node rates stayed stable through the
500 MiB run. These results point to per-token grammar and event overhead on
syntax-dense input as the next useful profiling target.

Run the same fixture matrix with:

```sh
cargo build --release -p zirium --example processing_benchmark
for size in 1 10 25 50 75 100; do
  target/release/examples/processing_benchmark --size-mib "$size" --shape primary --stage parse --warmups 1 --runs 3
done
target/release/examples/processing_benchmark --size-mib 10 --shape block-rich --stage parse --warmups 1 --runs 3
for depth in 8 64 256; do
  target/release/examples/processing_benchmark --size-mib 10 --shape nested --depth "$depth" --stage parse --warmups 1 --runs 3
done
target/release/examples/processing_benchmark --size-mib 100 --shape payload --stage parse --warmups 1 --runs 3
target/release/examples/processing_benchmark --size-mib 100 --shape trivia --stage parse --warmups 1 --runs 3
target/release/examples/processing_benchmark --size-mib 500 --shape primary --stage parse --warmups 1 --runs 3
```

Every generated fixture is exact-size, deterministic with seed
`0x5a495249554d0028`, stored at a process-specific path in the system temporary
directory, and removed after the run.

## Recorded full-pipeline results

All fixture byte counts were exact: 1,048,576; 10,485,760; 26,214,400;
52,428,800; 78,643,200; and 104,857,600. Cells contain median milliseconds /
peak live MiB.

| stage | 1 MiB | 10 MiB | 25 MiB | 50 MiB | 75 MiB | 100 MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| parse | 2.115 / 4.48 | 22.785 / 39.81 | 57.659 / 101.51 | 118.168 / 203.02 | 177.070 / 308.53 | 234.975 / 406.04 |
| traversal | 0.017 / 0 | 0.171 / 0 | 0.429 / 0 | 0.856 / 0 | 1.280 / 0 | 1.716 / 0 |
| lower (hybrid) | 1.683 / 2.07 | 18.877 / 17.22 | 51.834 / 36.13 | 125.559 / 72.25 | 182.048 / 136.00 | 245.462 / 144.50 |
| canonical | 0.168 / 0.09 | 1.679 / 0.94 | 4.854 / 2.36 | 10.954 / 4.72 | 19.525 / 7.07 | 23.371 / 9.43 |
| first use index | 0.011 / 0 | 0.183 / 0 | 0.665 / 0 | 1.749 / 0 | 3.614 / 0 | 3.781 / 0 |
| first symbol index | 0.025 / 0 | 0.284 / 0 | 1.171 / 0 | 4.198 / 0 | 6.833 / 0 | 7.982 / 0 |
| first dominance index | 0.230 / 0.41 | 2.270 / 3.30 | 5.545 / 6.59 | 12.847 / 13.19 | 22.718 / 26.38 | 28.093 / 26.38 |
| editor erase/commit | 0.423 / 0.96 | 4.072 / 9.00 | 12.050 / 21.34 | 24.245 / 42.69 | 52.695 / 68.62 | 64.005 / 85.37 |
| preserving output | 0.0004 / 0 | 0.0005 / 0 | 0.0006 / 0 | 0.0005 / 0 | 0.0008 / 0 | 0.0006 / 0 |

The operation counts were 2,062; 20,603; 51,504; 103,006; 154,508; and 206,009.
Canonical output sizes were 84,525; 844,706; 2,111,647; 4,223,229; 6,334,811;
and 8,446,352 bytes. Preserving output matched each input size exactly.

These canonical and editor measurements use linear structural validation.
Earlier 100 MiB timings of 11.635 s and 46.875 s predate that implementation
and should not be used to project its performance.

## Projection checks and the first 500 MiB run

The report fits only 1, 10, 25, and 50 MiB. It checks predictions against
held-out 75 and 100 MiB measurements, requiring at most 10% error for both
latency and peak allocation. It separately checks the 25–50, 50–75, and 75–100
per-MiB slopes, requiring every slope to remain within 10% of their median.
Stages must take at least 5 ms at 10 MiB to support a projection. A projection
must pass every check. Assumptions are the same fixture mix, allocator,
hardware, release build, and no paging cliff beyond 100 MiB.

Parse fit was `seconds = -0.000785087 + 0.002370543 × MiB` and `peak MiB =
-0.102681 + 4.060791 × MiB`. Held-out latency errors were 0.037% and 0.551%;
peak errors were 1.320% and 0.015%. Latency slopes were 0.002420388,
0.002356072, and 0.002316182 seconds/MiB; peak slopes were 4.060363, 4.220363,
and 3.900324 MiB/MiB. Both slope checks passed. The projected parse estimate was
1.184 seconds and 2,030.3 MiB peak at 500 MiB.

Lowering failed the projection checks. Its held-out latency errors were 2.057%
and 1.640%, but peak errors were 20.962% and 1.170%; both latency and peak
slope-stability checks failed. Other stages either failed held-out/stability
checks or the 5 ms signal floor. The report prints the fit, held-out errors,
slopes, stability decisions, assumptions, and `label=projected` for every stage.

A direct 500 MiB run produced these measurements:

| stage | input | median | peak live allocation | retained/result detail |
| --- | ---: | ---: | ---: | --- |
| parse | 524,288,000 bytes | 1.235528 s | 2,380,455,133 bytes | 5,150,191 CST nodes |
| lower (hybrid) | 524,288,000 bytes | 1.375655 s | 744,434,528 bytes | 1,030,038 operations; 356,287,558 direct-owned bytes |

The 500 MiB command measured parsing and lowering only.

## Nested regions and dense control flow

Each nested fixture was exactly 10,485,760 bytes. Padding sits inside the
deepest region and does not change the requested nesting depth.

| depth | parse ms / peak MiB | traversal ms / peak MiB | lower ms / peak MiB | semantic operations / regions / blocks |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 6.616 / 10.00 | 0.000125 / 0 | 4.100 / 0.015 | 9 / 8 / 8 |
| 64 | 6.785 / 10.00 | 0.000458 / 0 | 29.147 / 0.106 | 65 / 64 / 64 |
| 256 | 7.041 / 10.00 | 0.001791 / 0 | 115.684 / 0.420 | 257 / 256 / 256 |

The bounded 10 MiB block-rich run produced 414,349 operations, 69,059 regions,
207,175 blocks, and 2,486,096 dominance-index entries. Parse was 232.784 ms,
lower 1,018.031 ms, full verification 354.030 ms, and dominance-index
construction 144.366 ms. Their peak live allocations were 357.57, 483.78, 87.08,
and 134.13 MiB respectively. This run measures dense control flow at a larger
size than the earlier 1 MiB case.

## Python API measurements

Recorded with CPython 3.14.6 on macOS arm64 at exactly 10,485,760 input bytes,
one warm-up, and three measured runs:

| operation | median |
| --- | ---: |
| parse file | 272.005 ms |
| syntax operation access | 9.287 ms |
| operation component access | 71.903 ms |
| strict hybrid lowering | 292.910 ms |
| bulk operation snapshot | 18.645 ms |
| canonical file output | 94.233 s |
| preserving file output | 2.160 ms |

The result contained 20,603 semantic operations, wrote 844,706 canonical bytes
and 10,485,760 preserving bytes, and reported 10,336,767 direct-owned bytes,
988,944 document-index bytes, 10,485,760 retained-source bytes, and 8,030,636
retained-CST bytes. These are document-owned statistics only. They do not
measure Python interpreter memory, wrapper-object memory, or total process RSS.

### Buffered Python file-output rerun

Canonical and custom file output validate once before streaming through Rust's
default `BufWriter`. A benchmark of this implementation ran on
Apple M1 Max, macOS 26.5.2 arm64, CPython 3.14.5, and rustc 1.97.1. The fixture
was the same deterministic 10,485,760-byte primary input with seed
`0x5a495249554d0028`, one untimed warm-up, and three measured runs.

| output path | before median | after median |
| --- | ---: | ---: |
| canonical bytes plus buffered Python file write | not recorded | 14.351 ms |
| canonical file output | 94.233 s | 14.035 ms |
| custom file output | not recorded | 13.627 ms |
| original file output | not recorded | 2.520 ms |
| preserving file output | 2.160 ms | 8.811 ms |

The canonical file measurement was 0.978 times the separately measured
canonical-bytes-plus-buffered-write baseline. The benchmark reports all five output paths separately. Timings are observations,
not correctness thresholds; the preserving difference is not a like-for-like
regression claim because the rerun used a different OS and Python patch
release.

## Packed semantic operation snapshots

The release harness has an `operation-payload` Rust stage and matching Python
primary and `--shape block-rich` cases. They report traversal/payload time,
packed output bytes, counting-allocator peak, construction and stored-name
filter time, distinct-name count and bytes, and `tracemalloc` retained growth.
All final Python columns are filled directly with `PyBytes::new_with`; no
temporary byte-column copy is made. A filter first performs one non-mutating
document-string lookup and then compares stored u32 indices.

The study used three comparison limits: construction within 3x native
traversal, retained Python growth within 1.5x packed payload, and no material
transient duplication of columns.

The Apple M1 Max measurements used three runs after one warm-up:

| Measurement | Primary, 10 MiB | Block-rich, 1 MiB |
| --- | ---: | ---: |
| Native operation traversal | 0.247 ms | 0.733 ms |
| Native payload construction | 0.508 ms | 1.342 ms |
| Native peak allocation | 262,240 B | 524,400 B |
| Python construction | 0.594 ms | 1.383 ms |
| Python filtering | 0.552 ms | 0.468 ms |
| Operation rows | 20,603 | 41,707 |
| Distinct names / name bytes | 4 / 44 B | 6 / 62 B |
| Packed output, Rust and Python | 267,903 B | 542,281 B |
| Retained Python growth | 268,269 B | 542,647 B |

Both native payload builders took less than 3x the native traversal time.
Retained Python growth stayed below 1.5x payload, and direct filling avoided
transient column copies.

A separate warmed run measured higher Python construction costs:

| Fixture | Python construction | Native traversal | Ratio |
| --- | ---: | ---: | ---: |
| Primary | 1.198 ms | 0.273 ms | 4.39x |
| Block-rich | 2.871 ms | 0.682 ms | 4.21x |

These exceed the study's 3x traversal threshold. Further profiling could
separate allocation, column filling, and traversal costs. The native payload
builder provides the closest comparison because it also allocates and fills
columns. Cold import and setup were excluded from timing.

## Packed syntax operation relationships

`File.operation_table()` provides packed Python access to parsed
operation results, operands, successors, and regions. It returns nine frozen,
native-endian u32 byte columns: the operation node index and one offsets/value
pair for each relationship family. Callers use
`memoryview(column).cast("I")`; `File.operation_count` and bounds-checked
`File.operation(index)` retain lazy typed handles. Specialized semantic
snapshots remain available. Attributes, types, uses, blocks, dominance,
symbols, and semantic relationships require the semantic API.

Release measurements used Apple M1 Max, macOS 26.5.2 arm64, CPython 3.14.5,
rustc 1.97.1, one untimed warm-up, and three measured runs. Timing excludes Python startup.

| fixture | native walk | native packed / peak / payload | Python packed / traversal | retained / payload | Python/native packed | Python/native walk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| primary, 10 MiB | 0.890 ms | 2.153 ms / 936,376 B / 412,088 B | 2.830 ms / 0.001 ms | 412,504 B / 412,088 B | 1.31x | 3.18x |
| block-rich, 1 MiB | 1.869 ms | 4.809 ms / 2,021,756 B / 973,180 B | 9.107 ms / 0.001 ms | 973,629 B / 973,180 B | 1.89x | 4.87x |

Python fills its `bytes` directly with no temporary byte-column copy. Retained
growth is 1.001x payload for both fixtures, below 1.5x, and construction stays
below 3x the equivalent native packed builder. Both comparisons to a bare
native walk exceed 3x, at 3.18x and 4.87x, which suggests profiling repeated syntax walks.
