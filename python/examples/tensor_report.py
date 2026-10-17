#!/usr/bin/env -S uv run --script
"""Rank MLIR tensor results by their logical byte estimates.

Run from the checkout after building its Python extension. Uses the project
uv environment; no isolated script environment or extra dependencies are needed.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import zirium


@dataclass
class TensorResult:
    function: str | None
    operation: str
    operation_index: int
    result_index: int
    dimensions: list[int | None] | None
    element_type: str | None
    elements: int | None
    logical_bytes: int | None
    source_range: tuple[int, int] | None
    location: str | None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("input", type=Path, help="textual MLIR file")
    parser.add_argument(
        "--top", type=positive_count, default=20, help="maximum rows (default: 20)"
    )
    args = parser.parse_args()
    registry = zirium.DialectRegistry.from_name("stablehlo")
    parsed = zirium.parse_file(args.input, registry=registry)
    lowered = parsed.lower_strict("semantic")
    if parsed.diagnostics or lowered.document is None:
        messages = [item.message for item in parsed.diagnostics + lowered.diagnostics]
        parser.exit(1, "could not read complete MLIR:\n" + "\n".join(messages) + "\n")
    rows = tensor_results(lowered.document)
    rows.sort(key=lambda row: (row.logical_bytes is None, -(row.logical_bytes or 0)))
    print(
        json.dumps(
            {
                "document": str(args.input),
                "measurement": "Logical tensor bytes; not peak allocation or runtime. Unknown sizes are null.",
                "tensor_results": len(rows),
                "unknown_byte_sizes": sum(row.logical_bytes is None for row in rows),
                "results": [asdict(row) for row in rows[: args.top]],
            },
            indent=2,
        )
    )


def tensor_results(document: zirium.Document) -> list[TensorResult]:
    """Visit each operation once, carrying its nearest func.func owner."""
    table = document.operation_table()
    pending: list[tuple[zirium.SemanticOperation, str | None]] = [
        (table.operation(index), None)
        for index, is_root in reversed(list(enumerate(table.root_flags)))
        if is_root
    ]
    rows: list[TensorResult] = []
    operation_index = 0
    while pending:
        operation, function = pending.pop()
        if operation.name == "func.func":
            function = operation.symbol_name
        for result_index in range(operation.result_count()):
            result_type = operation.result_type(result_index)
            if result_type.kind != "tensor":
                continue
            dimensions = None if result_type.unranked else result_type.dimensions
            elements = None
            if dimensions is not None and all(size is not None for size in dimensions):
                elements = math.prod(size for size in dimensions if size is not None)
            element = result_type.element_type
            width = element_bytes(element)
            rows.append(
                TensorResult(
                    function=function,
                    operation=operation.name,
                    operation_index=operation_index,
                    result_index=result_index,
                    dimensions=dimensions,
                    element_type=element.spelling if element is not None else None,
                    elements=elements,
                    logical_bytes=elements * width
                    if elements is not None and width is not None
                    else None,
                    source_range=operation.source_range,
                    location=operation.location,
                )
            )
        operation_index += 1
        for region_index in reversed(range(operation.region_count())):
            region = operation.region(region_index)
            for block_index in reversed(range(region.block_count())):
                block = region.block(block_index)
                pending.extend(
                    (block.operation(index), function)
                    for index in reversed(range(block.operation_count()))
                )
    return rows


def element_bytes(element: zirium.SemanticType | None) -> int | None:
    """Estimate only ordinary byte-sized scalar elements; leave other types unknown."""
    if element is None:
        return None
    if element.kind == "integer":
        width = element.integer_width
        return width // 8 if width in (8, 16, 32, 64) else None
    if element.kind == "float":
        return {"bf16": 2, "f16": 2, "f32": 4, "f64": 8}.get(element.float_name or "")
    return None


def positive_count(value: str) -> int:
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("--top must be positive")
    return count


if __name__ == "__main__":
    main()
