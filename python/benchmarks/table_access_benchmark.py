#!/usr/bin/env python3
"""Compare packed operation-table reuse, wrappers, and native aggregation.

Build the extension first, then run ``uv run python python/benchmarks/table_access_benchmark.py --smoke``.
Timings are workload probes, not pass/fail thresholds; compare on the same host.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
import tracemalloc

import zirium
from zirium.query import ops


def source_for(count: int) -> str:
    return "\n".join('"bench.op"() : () -> ()' for _ in range(count))


def timed(action, warmups: int, runs: int) -> list[int]:
    samples = []
    for index in range(warmups + runs):
        started = time.perf_counter_ns()
        action()
        elapsed = time.perf_counter_ns() - started
        if index >= warmups:
            samples.append(elapsed)
    return samples


def measure(count: int, warmups: int, runs: int) -> dict[str, object]:
    lowered = zirium.parse_text(source_for(count)).lower_strict()
    assert lowered.document is not None, lowered.diagnostics
    document = lowered.document
    expression = ops().count()

    def fresh() -> None:
        fresh_table = document.operation_table()
        assert fresh_table.count == count

    table = document.operation_table()

    def reused() -> None:
        assert table.count == count

    def wrappers() -> None:
        assert (
            sum(table.operation(index).name == "bench.op" for index in range(count))
            == count
        )

    def native_aggregate() -> None:
        assert document.query(expression) == count

    cases = {
        "fresh_table": fresh,
        "reuse_table_count": reused,
        "wrapper_access": wrappers,
        "native_count": native_aggregate,
    }
    measurements: dict[str, dict[str, object]] = {}
    for name, action in cases.items():
        samples = timed(action, warmups, runs)
        measurements[name] = {
            "median_ns": int(statistics.median(samples)),
            "min_ns": min(samples),
            "max_ns": max(samples),
        }

    def python_allocation() -> None:
        tracemalloc.start()
        fresh()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        measurements["python_allocation"] = {"peak_allocated_bytes": peak}

    python_allocation()
    measurements["fresh_table"]["category"] = "mixed_table_construction"
    measurements["python_allocation"]["category"] = "python_allocation"
    measurements["reuse_table_count"]["category"] = "python_access"
    measurements["wrapper_access"]["category"] = "python_wrapper_traversal"
    measurements["native_count"]["category"] = "native_aggregation"
    return {"operations": count, "measurements": measurements}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    counts = [32, 128] if args.smoke else [1_000, 10_000]
    print(
        json.dumps(
            {
                "benchmark": "python-table-access",
                "python": platform.python_version(),
                "platform": platform.platform(),
                "build_profile": os.environ.get("ZIRIUM_BUILD_PROFILE", "unknown"),
                "warmups": args.warmups,
                "measured_runs": args.runs,
                "measurements": [measure(c, args.warmups, args.runs) for c in counts],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
