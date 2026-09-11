#!/usr/bin/env python3
"""Generate and profile a lightweight production-shaped MLIR stress case."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path

import zirium

MIB = 1024 * 1024
ARITHMETIC = ("mul", "add", "sub", "div")
COMPARISONS = ("less_than", "equal")
UNARY = ("reciprocal", "exp")
LITERALS = ("iter_index", "imm", "arg_in")


def registry(full: bool) -> zirium.DialectRegistry:
    shapes = {"stir.lambda": zirium.OperationShape.FUNC_LIKE}
    if full:
        shapes.update(
            {
                f"stir.{name}": zirium.OperationShape.BINARY_OPERANDS
                for name in ARITHMETIC
            }
        )
        shapes.update(
            {
                f"stir.{name}": zirium.OperationShape.BINARY_OPERANDS
                for name in COMPARISONS
            }
        )
        shapes.update(
            {f"stir.{name}": zirium.OperationShape.UNARY_OPERAND for name in UNARY}
        )
        shapes.update(
            {
                f"stir.{name}": zirium.OperationShape.LITERAL_ATTRIBUTE
                for name in LITERALS
            }
        )
        shapes["stir.select"] = zirium.OperationShape.VARIADIC_OPERANDS
        shapes["stir.return"] = zirium.OperationShape.VARIADIC_OPERANDS
    return zirium.DialectRegistry.with_operation_shapes(shapes)


def symbol(index: int) -> str:
    if index % 3 == 0:
        return f'@"<lambda>_{index}"'
    if index % 3 == 1:
        return f"@lambda_hash{10**19 + index}"
    return f"@synthetic_{index}"


def generated_operation(index: int) -> str:
    cases = (
        f"%v{index} = stir.mul %a, %b : bf16",
        f"%v{index} = stir.add %a, %b : bf16",
        f"%v{index} = stir.less_than %i, %i : i32 -> i32",
        f"%v{index} = stir.select %i, %a, %b : i32, bf16 -> bf16",
        f"%v{index} = stir.reciprocal %a : bf16",
        f'%v{index} = stir.iter_index "default_{index}" : i32',
        f"%v{index} = stir.imm {index % 17} : i64 : i32",
        f"%v{index} = stir.arg_in -1.099510e+12 : bf16 : bf16",
    )
    return cases[index % len(cases)]


def generate(path: Path, size_mib: int, full: bool) -> None:
    target = size_mib * MIB
    location_count = 14_362 if full else max(64, size_mib * 50)
    lambda_count = 1_445 if full else max(8, size_mib * 5)
    wide_arguments = 1_648 if size_mib >= 4 else 256
    locations = 0

    def located() -> str:
        nonlocal locations
        value = f" loc(#loc{locations % location_count})"
        locations += 1
        return value

    suffix = "}) : () -> () loc(#loc0)\n" + "".join(
        f'#loc{index} = loc("synthetic/layer.mlir":{index % 997 + 1}:1)\n'
        for index in range(location_count)
    )
    with path.open("wb") as output:
        written = output.write(b'"builtin.module"() ({\n"tlir.Wide"() ({\n')
        arguments = ", ".join(
            f"%wide_block_argument_{index}: tensor<3x1x2x1x256xf32>{located()}"
            for index in range(wide_arguments)
        )
        line = f"^wide({arguments}):\n}}) : () -> (){located()}\n".encode()
        output.write(line)
        written += len(line)

        operation_index = 0
        for lambda_index in range(lambda_count):
            body = [
                f"stir.lambda {symbol(lambda_index % 260)}(%a: bf16, %b: bf16, %i: i32) {{\n"
            ]
            for _ in range(5):
                body.append(f"  {generated_operation(operation_index)}{located()}\n")
                operation_index += 1
            two_operands = lambda_index % 8 == 0
            operands = "%a, %b" if two_operands else "%a"
            types = "bf16, bf16" if two_operands else "bf16"
            body.append(f"  stir.return {operands} : {types}{located()}\n")
            body.append(f"}}{located()}\n")
            data = "".join(body).encode()
            output.write(data)
            written += len(data)

        suffix_bytes = suffix.encode()
        generic_index = 0
        while written + len(suffix_bytes) + 800 <= target:
            dialect = "air" if generic_index % 20 == 0 else "tlir"
            attrs = ""
            if generic_index < 260:
                target_symbol = (
                    "@table::@function" if generic_index == 0 else symbol(generic_index)
                )
                attrs = f" {{target = {target_symbol}}}"
            elif generic_index < 276:
                attrs = " {limit = 0xFFF0000000000000 : f64}"
            operation = f'"{dialect}.Synthetic"(){attrs} : () -> (){located()}'
            data = (
                operation + " // " + "x" * max(0, 795 - len(operation)) + "\n"
            ).encode()
            output.write(data)
            written += len(data)
            generic_index += 1

        output.write(suffix_bytes)
        written += len(suffix_bytes)
        if written < target:
            padding = target - written
            output.write(b"//" + b"x" * (padding - 3) + b"\n")


def inspect(path: Path) -> dict[str, int]:
    lines = longest = location_references = 0
    with path.open("rb") as source:
        for line in source:
            lines += 1
            longest = max(longest, len(line.rstrip(b"\n")))
            location_references += line.count(b"loc(#loc")
    return {
        "bytes": path.stat().st_size,
        "lines": lines,
        "longest_line": longest,
        "location_references": location_references,
    }


def timed(runs: int, action):
    samples = []
    result = None
    for _ in range(runs):
        gc.collect()
        started = time.perf_counter()
        result = action()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), samples, result


def parse_and_count_errors(
    path: Path, selected: zirium.DialectRegistry
) -> tuple[int, int]:
    parsed = zirium.parse_file(path, registry=selected)
    count = parsed.operation_count
    errors = sum(parsed.operation(index).has_error for index in range(count))
    return count, errors


def measure(path: Path, registry_name: str, runs: int) -> dict[str, object]:
    selected = registry(full=registry_name == "full")
    zirium.parse_file(path, registry=selected)
    parse_median, parse_samples, parsed = timed(
        runs, lambda: zirium.parse_file(path, registry=selected)
    )
    assert parsed is not None
    parse_and_count_errors(path, selected)
    walk_median, walk_samples, walk_result = timed(
        runs, lambda: parse_and_count_errors(path, selected)
    )
    assert walk_result is not None
    walked_operations, error_operations = walk_result
    parsed.lower_best_effort("semantic")
    lower_median, lower_samples, lowered = timed(
        runs, lambda: parsed.lower_best_effort("semantic")
    )
    assert lowered is not None and lowered.document is not None
    unparsed = 0
    names = ARITHMETIC + COMPARISONS + UNARY + LITERALS
    for operation_name in ("stir.lambda", "stir.return", "stir.select", *names):
        full_name = (
            operation_name
            if operation_name.startswith("stir.")
            else f"stir.{operation_name}"
        )
        table = lowered.document.operation_table(full_name)
        unparsed += sum(
            table.operation(index).is_unparsed for index in range(table.count)
        )
    return {
        "registry": registry_name,
        "parse_seconds": parse_samples,
        "parse_median_seconds": parse_median,
        "parse_and_walk_seconds": walk_samples,
        "parse_and_walk_median_seconds": walk_median,
        "lower_seconds": lower_samples,
        "lower_median_seconds": lower_median,
        "throughput_mib_s": path.stat().st_size / MIB / parse_median,
        "operations": parsed.operation_count,
        "walked_operations": walked_operations,
        "operations_with_errors": error_operations,
        "parse_diagnostics": dict(Counter(item.kind for item in parsed.diagnostics)),
        "semantic_diagnostics": dict(
            Counter(item.kind for item in lowered.diagnostics)
        ),
        "unparsed_stir_operations": unparsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--smoke", action="store_true", help="2 MiB quick check")
    size.add_argument(
        "--full", action="store_true", help="268 MiB production-scale check"
    )
    parser.add_argument("--size-mib", type=int, default=16)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    size_mib = 268 if args.full else 2 if args.smoke else args.size_mib
    if size_mib < 1 or args.runs < 1:
        parser.error("size and runs must be positive")

    temporary = None
    if args.output is None:
        temporary = tempfile.TemporaryDirectory(prefix="zirium-stress-")
        output = Path(temporary.name) / "stress.mlir"
    else:
        output = args.output
    generate(output, size_mib, args.full)
    report = {
        "benchmark": "production-shaped-stress",
        "zirium": zirium.__version__,
        "geometry": inspect(output),
        "measurements": [
            measure(output, name, args.runs) for name in ("lambda-only", "full")
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output is not None:
        print(f"generated_file={output}")
    if temporary is not None:
        temporary.cleanup()


if __name__ == "__main__":
    main()
