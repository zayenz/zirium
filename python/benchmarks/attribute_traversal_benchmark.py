#!/usr/bin/env python3
"""Measure first-pass SemanticAttribute traversal from 1k through 8k elements."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
import tracemalloc
from typing import cast

import zirium


def rustc_version() -> str:
    try:
        return subprocess.run(
            ["rustc", "-V"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def source_for(shape: str, size: int) -> str:
    values = ", ".join(str(index) for index in range(size))
    if shape == "array":
        spelling = f"[{values}]"
    elif shape == "dictionary":
        spelling = (
            "{" + ", ".join(f"k{index:05d} = {index}" for index in range(size)) + "}"
        )
    else:
        spelling = f"array<i64: {values}>"
    return f'"bench"() {{value = {spelling}}} : () -> ()'


def measure(shape: str, size: int, warmups: int, runs: int) -> dict[str, object]:
    source = source_for(shape, size)
    document = zirium.parse_text(source).lower_strict().document
    assert document is not None
    operation = document.operation_table().operation(0)
    expected_sum = size * (size - 1) // 2

    samples = []
    for run in range(warmups + runs):
        attribute = operation.attribute_by_name("value")
        assert attribute is not None and attribute.element_count == size
        started = time.perf_counter_ns()
        total = 0
        for index in range(size):
            element = attribute.element(index)
            assert element is not None
            total += element.integer_value or 0
            if shape == "dictionary":
                assert element.name == f"k{index:05d}"
        elapsed = time.perf_counter_ns() - started
        assert total == expected_sum
        assert attribute.element(size) is None
        if run >= warmups:
            samples.append(elapsed)

    median = int(statistics.median(samples))

    # Keep allocation tracking out of the timed samples. This peak covers Python
    # allocations made while wrappers are traversed; Rust allocations and the
    # already-built document are outside this boundary.
    allocation_peaks = []
    for _ in range(runs):
        tracemalloc.start()
        attribute = operation.attribute_by_name("value")
        assert attribute is not None
        total = 0
        for index in range(size):
            element = attribute.element(index)
            assert element is not None
            total += element.integer_value or 0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert total == expected_sum
        allocation_peaks.append(peak)

    return {
        "shape": shape,
        "elements": size,
        "input_bytes": len(source.encode()),
        "median_ns": median,
        "min_ns": min(samples),
        "max_ns": max(samples),
        "spread_ns": max(samples) - min(samples),
        "ns_per_element": median / size,
        "peak_python_allocated_bytes": int(statistics.median(allocation_peaks)),
        "peak_python_allocated_spread_bytes": max(allocation_peaks)
        - min(allocation_peaks),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.warmups < 0 or args.runs < 1:
        parser.error("runs must be positive and warmups non-negative")
    sizes = [100, 200] if args.smoke else [1_000, 2_000, 4_000, 8_000]

    measurements = [
        measure(shape, size, args.warmups, args.runs)
        for shape in ("array", "dictionary", "dense")
        for size in sizes
    ]
    previous: dict[str, dict[str, object]] = {}
    for result in measurements:
        prior = previous.get(str(result["shape"]))
        result["size_ratio"] = (
            None
            if prior is None
            else (cast(int, result["median_ns"]) / cast(int, prior["median_ns"]))
        )
        previous[str(result["shape"])] = result

    print(
        json.dumps(
            {
                "benchmark": "python-attribute-traversal",
                "python": platform.python_version(),
                "platform": platform.platform(),
                "rustc": rustc_version(),
                "build_profile": os.environ.get("ZIRIUM_BUILD_PROFILE", "unknown"),
                "warmups": args.warmups,
                "measured_runs": args.runs,
                "measurement_boundary": (
                    "first-pass Python SemanticAttribute child-wrapper traversal; "
                    "fixture parsing/lowering excluded"
                ),
                "measurements": measurements,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
