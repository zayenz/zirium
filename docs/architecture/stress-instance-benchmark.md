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
