# Processing benchmark evidence

## Current baseline and decision

The current parser uses direct event-to-CST compaction. These figures are the
baseline for parser planning. The older full-pipeline tables below remain only
as historical evidence.

| fixture | median | throughput | peak live allocation | retained CST |
| --- | ---: | ---: | ---: | ---: |
| primary 10 MiB | 16.623 ms | 601.587 MiB/s | 31.79 MiB | 7.66 MiB |
| primary 100 MiB | 181.661 ms | 550.476 MiB/s | 325.88 MiB | 72.58 MiB |
| block-rich 10 MiB | 142.671 ms | 70.091 MiB/s | 269.85 MiB | 77.05 MiB |
| primary 500 MiB | 941.563 ms | 531.032 MiB/s | 1,869.39 MiB | 410.91 MiB |

Keep the event parser and flat CST. Syntax density drives the expensive case;
raw byte scanning and nesting do not. Direct compaction removes a full event
vector without narrowing the public event contract. The token copy and
token-index bitmap stay because they enforce arbitrary token-event ordering,
duplicate and omission checks, source order, and root coverage.

Keep source ranges in parser tokens and typed syntax. Parser-level string IDs
cost more memory than they save for the measured workloads, and registered
operation lookup does not dominate event production. Purpose-specific identity
tables belong in consumers that can show a benefit from their own access
patterns.

Python packed tables fill their final `bytes` allocations directly and remain
close to the equivalent native payload builders. Comparisons with a bare native
walk still exceed the three-times follow-up threshold for some shapes, so
reducing repeated syntax walks remains useful follow-up work.

## Historical parser construction and string-identity study

The ignored crate-internal test separates lexing, grammar event production, CST compaction, and structural verification without adding a public profiling API. Run it in release mode with one test thread:

```sh
ZIRIUM_PARSER_BENCH_SMOKE=1 cargo test --release -p zirium parser_construction_benchmark::measure_parser_construction_and_string_identity -- --ignored --nocapture --test-threads=1

ZIRIUM_PARSER_BENCH_SHAPE=primary ZIRIUM_PARSER_BENCH_SIZE_MIB=10 ZIRIUM_PARSER_BENCH_RUNS=3 cargo test --release -p zirium parser_construction_benchmark::measure_parser_construction_and_string_identity -- --ignored --nocapture --test-threads=1
ZIRIUM_PARSER_BENCH_SHAPE=primary ZIRIUM_PARSER_BENCH_SIZE_MIB=100 ZIRIUM_PARSER_BENCH_RUNS=3 cargo test --release -p zirium parser_construction_benchmark::measure_parser_construction_and_string_identity -- --ignored --nocapture --test-threads=1
ZIRIUM_PARSER_BENCH_SHAPE=block-rich ZIRIUM_PARSER_BENCH_SIZE_MIB=10 ZIRIUM_PARSER_BENCH_RUNS=3 cargo test --release -p zirium parser_construction_benchmark::measure_parser_construction_and_string_identity -- --ignored --nocapture --test-threads=1
```

The smoke fixture is 64 KiB with one warm-up and one run. The other controls default to a 10 MiB primary fixture, one warm-up, and three runs; `ZIRIUM_PARSER_BENCH_WARMUPS` and `ZIRIUM_PARSER_BENCH_RUNS` override those counts. These measurements used the same machine described below: Apple M1 Max, Darwin 25.5.0, `aarch64-apple-darwin`, release profile, rustc 1.97.1. Fixtures are deterministic and generated in memory; no generated fixture is retained in the repository.

Each row is the median of three runs. Peak allocation is incremental above a baseline taken while the input and prerequisite outputs are live. Lexing retains the source; event production retains the source and token tape; compaction also retains the event tape and measures the input clones and working allocations required by destructive compaction; verification retains the completed unverified CST. The measured output remains live through a black-box observation. These peaks describe separate lifetimes and are not additive.

The following table is the base-041 baseline, before changing compaction:

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

Base-042 removes the second complete event vector. Forward-parent chains are resolved into one reused chain buffer and each normalized event is compacted immediately. The same release commands on the same machine produced:

| fixture | compact median ms | base-041 / base-042 throughput MiB/s | throughput change | incremental peak live MiB | change from base-041 peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| primary 10 MiB | 4.503 | 1,215.95 / 2,220.64 | +82.6% | 19.84 | -26.7% |
| primary 100 MiB | 46.467 | 1,183.73 / 2,152.08 | +81.8% | 194.41 | -27.1% |
| block-rich 10 MiB | 68.244 | 70.16 / 146.53 | +108.8% | 213.57 | -29.1% |

Decision: retain the direct compaction pass. Its peak reduction scales across both fixture shapes, and it also roughly halves compaction time without changing the event-based CST architecture. Retain the token copy into CST storage and the token-index `seen` bitmap. Together they preserve the public `from_events` contract for arbitrarily ordered token events, including invalid-index, duplicate, source-order, omission, and root-coverage checks. The bitmap is one byte per input token and the token copy is the final tree's required ordered storage; removing either safely would require constraining the public event contract or adding another construction path. After the event-vector removal, neither is a sufficiently isolated measured temporary to justify that complexity. Retain source-range token text and do not add parser-level interning for the reasons measured below.

String accounting includes `BareIdentifier`, `AtIdentifier`, `PercentIdentifier`, `CaretIdentifier`, `ExclamationIdentifier`, and `HashIdentifier`. Quoted `String` tokens count only when they are direct token elements of an `Operation` or `DialectOperation` CST node. This uses the CST boundary to skip an optional result child and excludes strings nested in attributes, regions, or other operation components. Current cost is the required eight-byte source range for each occurrence. The candidate retains that range because lossless reconstruction and syntax ranges still require source position, then adds a four-byte stable ID per occurrence and, per unique spelling, a four-byte ID, eight-byte stored hash, two machine-word range/pointer fields, and the unique spelling bytes. It deliberately does not claim that duplicate spelling bytes can be removed from the retained lossless source. Actual hash-table control bytes and spare capacity would make the candidate somewhat larger.

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

The absent exclamation and hash classes are still scanned and reported as zero for these fixtures. The representative prebuilt hash-table lookup pass took 0.934 ms for 37,120 primary-10 lookups, 14.781 ms for 371,180 primary-100 lookups, and 21.881 ms for 966,813 block-rich lookups. This is a deliberately small lookup experiment, not a production interner design.

The complete parser spelling-comparison inventory has three groups. Fixed grammar words in `parser.rs` are `attributes` in the registered `builtin.module` and `func.func` callbacks; the function visibility spellings `public`, `private`, and `nested`; `overflow` plus its `nsw`, `nuw`, and `none` flags in `arith.constant`; and `type` and `unit` in attribute parsing. Dictionary parsing also compares a bare or quoted key with `no_inline` to admit the registered unit-attribute shorthand. Finally, every bare operation name is passed to `DialectRegistry::operation`, which compares it with registered descriptor names to select custom parsing. A scan of every source-to-text conversion in `parser.rs` found no other spelling comparisons. Typed syntax views perform structural `SyntaxKind` comparisons and do not compare source spellings. Semantic lowering has its own document-local interning and is outside this parser-storage decision.

Decision: keep source ranges without adding stable IDs in parser tokens or typed syntax. The realizable range-preserving candidate adds about 1.42 MiB on the repetitive 100 MiB primary fixture before hash-table overhead, while CST compaction peaks at 266.62 MiB. On the block-rich fixture it adds about 5.98 MiB across the measured classes because 69,058 symbol names are unique. Registered bare operation dispatch is the one frequency-sensitive comparison site, but the current registry is small, the event-production measurement does not identify dispatch as a dominant cost, and adding IDs to every measured identifier class would charge substantially more storage than that lookup can justify. The remaining parser comparisons use a small fixed vocabulary, `no_inline` is checked only while parsing dictionary keys, and typed syntax performs no spelling comparisons. Post-parse consumers may benefit from narrower purpose-specific identity tables, but that requires evidence from their own lifetimes and access patterns.

`processing_benchmark` generates deterministic, exact-size MLIR fixtures in temporary storage with seed `0x5a495249554d0028`. Measurements below were recorded on Apple M1 Max, Darwin 25.5.0, `aarch64-apple-darwin`, release profile, rustc 1.97.1. Each result is the median of three measured runs after one untimed warm-up. Peak values are process-scoped live allocation above the stage baseline.

## Reproduction

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

## Historical packed Python syntax table measurement

The base-045 measurement used the same deterministic 10 MiB primary fixture,
seed, release profile, one warm-up, and three measured runs as the processing
baseline above. The native traversal took 0.171 ms. Constructing the same eight
packed payload columns in the Rust counting-allocator harness took 2.102 ms,
produced 6,489,810 payload bytes, and added 6,489,810 bytes at peak. This stage
retains its eight final Rust vectors only for the measurement.

The release Python `SyntaxTable` construction took 1.984 ms and returned the
same 6,489,810 payload bytes. Tracemalloc reported 6,490,218 bytes of retained
growth, or 1.00006 times the payload, below the 1.5-times threshold. Each final
Python `bytes` allocation is filled in place with `PyBytes::new_with`; no Rust
temporary column buffer is allocated, so there is no material transient column
duplication.

Python construction was 11.6 times native traversal, above the required
three-times threshold, and remains follow-up work. Most of that comparison is
the necessary encoding and allocation of 6.49 MB rather than traversal alone:
Python construction was 0.94 times the matching native payload-construction
stage. These figures are observations on the recorded machine, not performance
guarantees.

`--depth` is required for nested fixtures and rejected for other shapes. The generator raises the parser delimiter limit to cover the selected depth; padding still makes every fixture exactly the requested byte count. Release smoke checks covered primary, block-rich, trivia, payload, and nested depths 8, 64, and 256.

## Current parser baseline details

The current parser baseline uses direct compaction. The measurement machine is
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

### Historical comparison with base-039

The 100 MiB primary result is the direct comparison with base-039. Median
latency fell from 234.975 ms to 181.661 ms (22.7%), and peak live allocation
fell from 406.04 MiB to 325.88 MiB (19.7%). The resulting CST contains
4,944,201 tokens and 1,030,046 nodes in 76,107,628 retained bytes. These exact
counts should be used with this post-evaluation baseline.

The machine was suitably provisioned for a direct 500 MiB primary parse (64 GiB
RAM and 295 GiB free temporary storage). The measured result, not a projection,
was 941.563 ms at 531.032 MiB/s, with 1,960,199,833 bytes (1,869.39 MiB) peak
live allocation and 430,868,492 bytes (410.91 MiB) retained for 24,720,897
tokens and 5,150,191 nodes. That is 26.255 million tokens/s and 5.470 million
nodes/s. Compared with the base-039 direct 500 MiB result of 1.235528 s and
2,380,455,133 bytes peak, latency fell 23.8% and peak allocation fell 17.7%.
It also remains consistent with the earlier base-039 projection of 1.184 s and
2,030.3 MiB peak.

### Current interpretation

The remaining parser cost is syntax density, not raw byte scanning or nesting.
At 10 MiB the block-rich fixture is about 8.6 times slower than primary and
retains about ten times as many CST bytes, while its token rate remains close
to primary. Payload, trivia, and exact-depth nested fixtures stay
scanning-dominated. Primary throughput and normalized token/node rates remain
stable through the direct 500 MiB run without a paging cliff. The existing
event parser and flat CST therefore remain justified; these measurements do
not support a more invasive representation change. Future parser work should
first isolate per-token grammar/event overhead on syntax-dense input rather
than replace the representation wholesale.

Reproduce the current matrix with the temporary-fixture harness:

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

## Historical full-pipeline matrix

All fixture byte counts were exact: 1,048,576; 10,485,760; 26,214,400; 52,428,800; 78,643,200; and 104,857,600. Cells contain median milliseconds / peak live MiB.

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

The operation counts were 2,062; 20,603; 51,504; 103,006; 154,508; and 206,009. Canonical output sizes were 84,525; 844,706; 2,111,647; 4,223,229; 6,334,811; and 8,446,352 bytes. Preserving output matched each input size exactly.

Earlier canonical and editor figures (for example 100 MiB at 11.635 s and 46.875 s) were pre-linear-validation baselines. The table replaces them with current reruns after structural validation became linear; those old figures must not be used for current projections.

## Historical projection and first 500 MiB result

The report fits only 1, 10, 25, and 50 MiB. It checks predictions against held-out 75 and 100 MiB measurements, requiring at most 10% error for both latency and peak allocation. It separately checks the 25–50, 50–75, and 75–100 per-MiB slopes, requiring every slope to remain within 10% of their median. The existing 10 MiB latency signal floor is 5 ms. A projection must pass every check. Assumptions are the same fixture mix, allocator, hardware, release build, and no paging cliff beyond 100 MiB.

Parse fit was `seconds = -0.000785087 + 0.002370543 × MiB` and `peak MiB = -0.102681 + 4.060791 × MiB`. Held-out latency errors were 0.037% and 0.551%; peak errors were 1.320% and 0.015%. Latency slopes were 0.002420388, 0.002356072, and 0.002316182 seconds/MiB; peak slopes were 4.060363, 4.220363, and 3.900324 MiB/MiB. Both slope checks passed. The resulting parse estimate is clearly labeled **projected**: 1.184 seconds and 2,030.3 MiB peak at 500 MiB.

Lowering did not earn a credible projection. Its held-out latency errors were 2.057% and 1.640%, but peak errors were 20.962% and 1.170%; both latency and peak slope-stability checks failed. Other stages either failed held-out/stability checks or the 5 ms signal floor. The report prints the fit, held-out errors, slopes, stability decisions, assumptions, and `label=projected` for every stage.

The documented machine completed the bounded direct command, so these values are **measured**, not projected:

| stage | input | median | peak live allocation | retained/result detail |
| --- | ---: | ---: | ---: | --- |
| parse | 524,288,000 bytes | 1.235528 s | 2,380,455,133 bytes | 5,150,191 CST nodes |
| lower (hybrid) | 524,288,000 bytes | 1.375655 s | 744,434,528 bytes | 1,030,038 operations; 356,287,558 direct-owned bytes |

Canonical and editor stages were intentionally excluded from the 500 MiB command.

## Historical nested and block-rich evidence

Each nested fixture was exactly 10,485,760 bytes. Padding sits inside the deepest region and does not change the requested nesting depth.

| depth | parse ms / peak MiB | traversal ms / peak MiB | lower ms / peak MiB | semantic operations / regions / blocks |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 6.616 / 10.00 | 0.000125 / 0 | 4.100 / 0.015 | 9 / 8 / 8 |
| 64 | 6.785 / 10.00 | 0.000458 / 0 | 29.147 / 0.106 | 65 / 64 / 64 |
| 256 | 7.041 / 10.00 | 0.001791 / 0 | 115.684 / 0.420 | 257 / 256 / 256 |

The bounded 10 MiB block-rich run produced 414,349 operations, 69,059 regions, 207,175 blocks, and 2,486,096 dominance-index entries. Parse was 232.784 ms, lower 1,018.031 ms, full verification 354.030 ms, and dominance-index construction 144.366 ms. Their peak live allocations were 357.57, 483.78, 87.08, and 134.13 MiB respectively. This extends dense CFG coverage beyond the earlier 1 MiB point without creating a shape/size cross-product.

## Historical Python boundary evidence

Recorded with CPython 3.14.6 on macOS arm64 at exactly 10,485,760 input bytes, one warm-up, and three measured runs:

| operation | median |
| --- | ---: |
| parse file | 272.005 ms |
| syntax operation access | 9.287 ms |
| operation component access | 71.903 ms |
| strict hybrid lowering | 292.910 ms |
| bulk operation snapshot | 18.645 ms |
| canonical file output | 94.233 s |
| preserving file output | 2.160 ms |

The result contained 20,603 semantic operations, wrote 844,706 canonical bytes and 10,485,760 preserving bytes, and reported 10,336,767 direct-owned bytes, 988,944 document-index bytes, 10,485,760 retained-source bytes, and 8,030,636 retained-CST bytes. These are document-owned statistics only. They do not measure Python interpreter memory, wrapper-object memory, or total process RSS.

### Buffered Python file-output rerun

After changing canonical and custom file output to preflight once and stream
through Rust's default `BufWriter`, the release Python benchmark was rerun on
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
canonical-bytes-plus-buffered-write baseline, within the task's 2x bound. The
benchmark reports all five output paths separately. Timings are observations,
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

Packed runs flag construction above 3x native traversal, retained Python
growth above 1.5x packed payload, or any material transient column duplication
as follow-up work.

Release measurements on the documented Apple M1 Max used three runs after one
warm-up. At 10 MiB primary, native operation traversal was 0.247 ms and native
payload construction 0.508 ms (267,903 output bytes; 262,240 peak allocation).
Python construction was 0.594 ms and filtering 0.552 ms, with 20,603 rows, four
names/44 name bytes, 267,903 packed bytes, and 268,269 retained bytes. At the
bounded 1 MiB block-rich fixture, native traversal was 0.733 ms and payload
construction 1.342 ms (542,281 output bytes; 524,400 peak allocation). Python
construction was 1.383 ms and filtering 0.468 ms, with 41,707 rows, six
names/62 name bytes, 542,281 packed bytes, and 542,647 retained bytes. Both
equivalent native payload builders stayed below 3x native operation traversal,
retained Python growth stayed below 1.5x payload, and direct fill avoided
material transient duplication. Fresh warmed Python measurements nevertheless
recorded 1.198 ms construction versus a 0.273 ms bare native walk for primary
(4.39x), and 2.871 ms versus 0.682 ms for block-rich (4.21x). These warmed
results exceed the task's literal 3x bare-walk threshold and are follow-up
evidence for reducing the Python snapshot boundary overhead. The closer
equivalent comparison is Rust payload construction rather than a walk that
does not allocate or fill columns. Cold import and setup remain separately
excluded and are not treated as regression evidence.

## Packed syntax operation relationships

`File.operation_table()` is now the primary Python access path for parsed
operation results, operands, successors, and regions. It returns nine frozen,
native-endian u32 byte columns: the operation node index and one offsets/value
pair for each relationship family. Callers use
`memoryview(column).cast("I")`; `File.operation_count` and bounds-checked
`File.operation(index)` retain lazy typed handles. Specialized semantic
snapshots remain available. Attributes, types, uses, blocks, dominance,
symbols, and semantic relationships are deliberately absent.

Release measurements used Apple M1 Max, macOS 26.5.2 arm64, CPython 3.14.5,
rustc 1.97.1, one untimed warm-up, and three measured runs. Cold Python startup
is setup cost, not boundary evidence.

| fixture | native walk | native packed / peak / payload | Python packed / traversal | retained / payload | Python/native packed | Python/native walk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| primary, 10 MiB | 0.890 ms | 2.153 ms / 936,376 B / 412,088 B | 2.830 ms / 0.001 ms | 412,504 B / 412,088 B | 1.31x | 3.18x |
| block-rich, 1 MiB | 1.869 ms | 4.809 ms / 2,021,756 B / 973,180 B | 9.107 ms / 0.001 ms | 973,629 B / 973,180 B | 1.89x | 4.87x |

Python fills its `bytes` directly with no temporary byte-column copy. Retained
growth is 1.001x payload for both fixtures, below 1.5x, and construction stays
below 3x the equivalent native packed builder. Both comparisons to a bare
native walk exceed 3x, at 3.18x and 4.87x, so reducing repeated syntax walks is
follow-up work.
