# Production-shaped stress benchmark

`python/benchmarks/stress_instance_benchmark.py` generates a deterministic approximation of the private build described in usability brief 28. It keeps the properties useful during development: forward-referenced locations, a very wide block-argument line, shallow fan-out, varied symbols, representative `stir.*` forms, both type trailers, and hexadecimal floats.

It compares the old registry, where only `stir.lambda` is recognized, with the complete shape registry. Generation is excluded from the timings; parsing and lowering have one warm-up and three measured runs.

```sh
uv run maturin develop --release
uv run python python/benchmarks/stress_instance_benchmark.py --smoke
uv run python python/benchmarks/stress_instance_benchmark.py
uv run python python/benchmarks/stress_instance_benchmark.py --full
```

The default is 16 MiB. `--smoke` uses 2 MiB and `--full` uses 268 MiB. Pass `--output PATH` to keep the generated input. Results are JSON on stdout.

This is a development workload, not a scientific reproduction of the private corpus. Its purpose is to catch scaling regressions and confirm that registering the added shapes removes recovered `stir.*` operations without materially changing parse or lowering cost.

## Development baseline

Measured on 2026-09-09 with the release Python extension, one warm-up and three runs. The generated 268 MiB file contained 359,722 operations, 375,533 lines, and a 106,554-byte block-argument line.

| registry | parse median | lower median | parse diagnostics | unparsed `stir.*` |
| --- | ---: | ---: | ---: | ---: |
| legacy | 0.384 s | 0.715 s | 8,670 | 8,670 |
| full | 0.494 s | 0.728 s | 0 | 0 |

Recognizing the additional custom forms costs about 29% in the parse stage on this synthetic mix; lowering changes by about 2%. The useful result is that the new registry reaches complete parsing without a lowering regression or semantic diagnostics. The absolute times are much lower than the private corpus because most bulk operations here are intentionally simple padding operations.
