# Production-shaped stress benchmark

`python/benchmarks/stress_instance_benchmark.py` generates deterministic MLIR shaped like the production build described in usability brief 28. The input has forward-referenced locations, a 100 KiB block-argument line, shallow fan-out, varied symbols, representative `stir.*` forms, both type trailers, and hexadecimal floats.

The benchmark times two registries against the same input. The legacy registry recognizes `stir.lambda`. The full registry recognizes every generated `stir.*` form. Parsing and lowering each run once to warm up, then three times for measurement. Input generation happens before timing starts.

```sh
uv run maturin develop --release
uv run python python/benchmarks/stress_instance_benchmark.py --smoke
uv run python python/benchmarks/stress_instance_benchmark.py
uv run python python/benchmarks/stress_instance_benchmark.py --full
```

The default is 16 MiB. `--smoke` uses 2 MiB and `--full` uses 268 MiB. Pass `--output PATH` to keep the generated input. Results are JSON on stdout. Use it to check parser scaling and the cost of registering the added operation shapes.

## Development baseline

Measured on 2026-09-09 with the release Python extension, one warm-up and three runs. The generated 268 MiB file contained 359,722 operations, 375,533 lines, and a 106,554-byte block-argument line.

| registry | parse median | lower median | parse diagnostics | unparsed `stir.*` |
| --- | ---: | ---: | ---: | ---: |
| legacy | 0.384 s | 0.715 s | 8,670 | 8,670 |
| full | 0.494 s | 0.728 s | 0 | 0 |

Recognizing the additional custom forms adds about 29% to parse time on this input. Lowering adds about 2%. The full registry parses every generated `stir.*` operation and emits no syntax or semantic diagnostics.
