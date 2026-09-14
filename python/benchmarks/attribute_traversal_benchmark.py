#!/usr/bin/env python3
"""Measure first-pass SemanticAttribute traversal from 1k through 8k elements."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from typing import cast

import zirium


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
    document = zirium.parse_text(source_for(shape, size)).lower_strict().document
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
    return {
        "shape": shape,
        "elements": size,
        "median_ns": median,
        "min_ns": min(samples),
        "max_ns": max(samples),
        "ns_per_element": median / size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
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
                "warmups": args.warmups,
                "measured_runs": args.runs,
                "measurements": measurements,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
