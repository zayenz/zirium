# Compiler-input stress benchmark

`python/benchmarks/stress_instance_benchmark.py` generates deterministic MLIR
modeled on a production compiler dump. The input has forward-referenced
locations, a 100 KiB block-argument line, shallow fan-out, varied symbols,
representative `stir.*` forms, both type trailers, and hexadecimal floats.

The benchmark times two registries against the same input. The legacy registry
recognizes `stir.lambda`. The full registry recognizes every generated `stir.*`
form. Each measurement gets one warm-up run, then three timed runs. Input
generation happens before timing starts.

The Python traversal measurement parses the file, visits every operation through
`File.operation(index)`, and counts operations whose syntax contains an error.
This exercises the ordinary object interface after parsing. The count also
checks that the complete operation sequence was visited.

```sh
uv run maturin develop --release
uv run python python/benchmarks/stress_instance_benchmark.py --smoke
uv run python python/benchmarks/stress_instance_benchmark.py
uv run python python/benchmarks/stress_instance_benchmark.py --full
```

The default is 16 MiB. `--smoke` uses 2 MiB and `--full` uses 268 MiB. Pass
`--output PATH` to keep the generated input. Results are JSON on stdout. Use the
results to compare parser scaling and the cost of the two registries.

## Recorded measurements

Measured on 2026-09-09 with the release Python extension, one warm-up and three
runs. The generated 268 MiB file contained 359,722 operations, 375,533 lines,
and a 106,554-byte block-argument line.

| registry | parse median | parse and walk median | lower median | operations with errors | unparsed `stir.*` |
| --- | ---: | ---: | ---: | ---: | ---: |
| legacy | 0.458 s | 0.605 s | 0.746 s | 10,116 | 8,670 |
| full | 0.583 s | 0.608 s | 0.737 s | 0 | 0 |

The Python walk covers all 359,722 operations. With the full registry, parsing
and walking takes 25 ms more than parsing alone, about 4%. The full registry
parses every generated `stir.*` operation and emits no syntax or semantic
diagnostics.

## Python operation access

`File.operation(index)` uses a lazy index of syntax operation nodes. The first
operation count or lookup builds the index; parsing alone does not build it.

In the recorded comparison on a 16 MiB input, indexing reduced the time to walk
21,318 operations from 1.51 seconds to 2.44 milliseconds. Parsing took about
28 milliseconds in both runs.
