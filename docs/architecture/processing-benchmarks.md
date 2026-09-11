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

For evaluator-only and generated-graph checks, see [query profiling](query-profiling.md).
For production-shaped inputs, see the [compiler-input stress benchmark](stress-instance-benchmark.md).
