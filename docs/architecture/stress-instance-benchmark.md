# Compiler-input stress benchmark

`python/benchmarks/stress_instance_benchmark.py` generates deterministic MLIR
modeled on a production compiler dump. The input has forward-referenced
locations, a 100 KiB block-argument line, shallow fan-out, varied symbols,
representative `stir.*` forms, both type trailers, and hexadecimal floats.

The benchmark times two registries against the same input. The lambda-only registry
recognizes `stir.lambda`. The full registry recognizes every generated `stir.*`
form. Each measurement gets one warm-up run, then three timed runs. Input
generation happens before timing starts.

The Python traversal measurement parses the file and visits every operation
through `File.operation(index)`, counting syntax errors and checking that all
operations were visited.

```sh
uv run maturin develop --release
uv run python python/benchmarks/stress_instance_benchmark.py --smoke
uv run python python/benchmarks/stress_instance_benchmark.py
uv run python python/benchmarks/stress_instance_benchmark.py --full
```

The default is 16 MiB; `--smoke` uses 2 MiB and `--full` uses 268 MiB. Pass
`--output PATH` to keep the input. JSON results on stdout compare parser scaling
and registry costs.

## Python operation access

`File.operation(index)` uses a lazy index of syntax operation nodes. The first
operation count or lookup builds it; parsing alone does not. The traversal
measurement includes that construction cost.
