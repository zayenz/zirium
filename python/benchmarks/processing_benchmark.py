#!/usr/bin/env python3
"""Small file-oriented companion to the Rust processing benchmark."""

from __future__ import annotations

import argparse
import gc
import platform
import statistics
import tempfile
import time
import tracemalloc
from pathlib import Path

import zirium

MIB = 1024 * 1024
SEED = 0x5A495249554D0028


def fixture(size: int) -> bytes:
    prefix = b'"bench.container"() ({\n^bb:\n%seed = "bench.source"() : () -> i32\n"bench.use"(%seed) : (i32) -> ()\n'
    line = (
        b'"bench.op"() {tag = "zirium"} : () -> () // deterministic operation '
        + b"x" * 440
        + b"\n"
    )
    suffix = b"}) : () -> ()\n"
    result = bytearray(prefix)
    while len(result) + len(line) + len(suffix) <= size:
        result += line
    result += b" " * (size - len(result) - len(suffix))
    result += suffix
    assert len(result) == size
    return bytes(result)


def block_rich_fixture(size: int) -> bytes:
    result = bytearray(b"builtin.module {\n")
    suffix = b"}\n"
    index = 0
    while True:
        function = f'func.func @f{index}() {{\n^entry:\n%value = arith.constant 1 : i32\ncf.br ^middle\n^middle:\n"bench.use"(%value) : (i32) -> ()\ncf.br ^exit\n^exit:\nfunc.return\n}}\n'.encode()
        if len(result) + len(function) + len(suffix) > size:
            break
        result += function
        index += 1
    result += b" " * (size - len(result) - len(suffix)) + suffix
    return bytes(result)


def timed(runs: int, action):
    samples = []
    result = None
    for _ in range(runs):
        started = time.perf_counter_ns()
        result = action()
        samples.append(time.perf_counter_ns() - started)
    return int(statistics.median(samples)), result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--size-mib", type=int, default=10)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--shape", choices=("primary", "block-rich"), default="primary")
    args = parser.parse_args()
    size = 64 * 1024 if args.smoke else args.size_mib * MIB
    runs = 1 if args.smoke else args.runs
    if size == 500 * MIB:
        parser.error("500 MiB is projection-only")
    print(
        f"benchmark=python-processing python={platform.python_version()} platform={platform.platform()} seed=0x{SEED:016x} input_bytes={size} warmups=1 measured_runs={runs}"
    )
    with tempfile.TemporaryDirectory(prefix="zirium-python-benchmark-") as directory:
        source = Path(directory) / "input.mlir"
        canonical_buffered = Path(directory) / "canonical-buffered.mlir"
        canonical = Path(directory) / "canonical.mlir"
        custom = Path(directory) / "custom.mlir"
        original = Path(directory) / "original.mlir"
        preserving = Path(directory) / "preserving.mlir"
        source.write_bytes(
            block_rich_fixture(size) if args.shape == "block-rich" else fixture(size)
        )
        registry = zirium.DialectRegistry.proving()
        zirium.parse_file(source, registry=registry)  # warm-up
        parse_ns, parsed = timed(
            runs, lambda: zirium.parse_file(source, registry=registry)
        )
        parsed.syntax_table()  # warm-up
        syntax_table_ns, syntax_table = timed(runs, parsed.syntax_table)
        columns = (
            syntax_table.node_kind,
            syntax_table.node_start,
            syntax_table.node_end,
            syntax_table.node_subtree_end,
            syntax_table.node_flags,
            syntax_table.token_kind,
            syntax_table.token_start,
            syntax_table.token_end,
        )
        syntax_payload_bytes = sum(map(len, columns))
        del syntax_table, columns
        gc.collect()
        tracemalloc.start()
        retained_before = tracemalloc.get_traced_memory()[0]
        _syntax_table = parsed.syntax_table()
        syntax_table_retained_growth = (
            tracemalloc.get_traced_memory()[0] - retained_before
        )
        tracemalloc.stop()
        parsed.operation_table()  # warm-up
        syntax_operation_table_ns, syntax_operation_table = timed(
            runs, parsed.operation_table
        )
        relationship_columns = (
            syntax_operation_table.operation_node,
            syntax_operation_table.result_offsets,
            syntax_operation_table.result_nodes,
            syntax_operation_table.operand_offsets,
            syntax_operation_table.operand_nodes,
            syntax_operation_table.successor_offsets,
            syntax_operation_table.successor_nodes,
            syntax_operation_table.region_offsets,
            syntax_operation_table.region_nodes,
        )
        syntax_operation_payload_bytes = sum(map(len, relationship_columns))

        def count_components(table=syntax_operation_table) -> int:
            return sum(
                memoryview(column).cast("I")[-1]
                for column in (
                    table.result_offsets,
                    table.operand_offsets,
                    table.successor_offsets,
                    table.region_offsets,
                )
            )

        count_components()  # warm-up
        syntax_operation_traversal_ns, component_count = timed(runs, count_components)
        del syntax_operation_table, relationship_columns
        gc.collect()
        tracemalloc.start()
        retained_before = tracemalloc.get_traced_memory()[0]
        _retained_syntax_operations = parsed.operation_table()
        syntax_operation_retained_growth = (
            tracemalloc.get_traced_memory()[0] - retained_before
        )
        tracemalloc.stop()
        parsed.lower_strict("hybrid")  # warm-up
        lower_ns, lowered = timed(runs, lambda: parsed.lower_strict("hybrid"))
        document = lowered.document
        assert document is not None
        document.operation_table()  # warm-up
        operation_table_ns, operation_table = timed(runs, document.operation_table)
        filter_name = "arith.constant" if args.shape == "block-rich" else "bench.op"
        filter_ns, filtered = timed(runs, lambda: document.operation_table(filter_name))
        offsets = memoryview(operation_table.name_offsets).cast("I")
        distinct_names = len(offsets) - 1
        packed_payload_bytes = sum(
            map(
                len,
                (
                    operation_table.name_code,
                    operation_table.source_start,
                    operation_table.source_end,
                    operation_table.root_flags,
                    operation_table.name_offsets,
                    operation_table.name_bytes,
                ),
            )
        )
        gc.collect()
        tracemalloc.start()
        retained_before = tracemalloc.get_traced_memory()[0]
        _retained_table = document.operation_table()
        operation_table_retained_growth = (
            tracemalloc.get_traced_memory()[0] - retained_before
        )
        tracemalloc.stop()
        if args.shape == "block-rich":
            stats = document.statistics()
            print(
                f"measurement shape={args.shape} syntax_operation_table_ns={syntax_operation_table_ns} syntax_operation_traversal_ns={syntax_operation_traversal_ns} syntax_operation_payload_bytes={syntax_operation_payload_bytes} syntax_operation_tracemalloc_retained_growth={syntax_operation_retained_growth} syntax_operation_direct_pybytes_fill=true syntax_operation_temporary_column_duplication_bytes=0 operation_components={component_count} operation_table_ns={operation_table_ns} operation_filter_ns={filter_ns} operations={operation_table.count} filtered_operations={filtered.count} distinct_operation_names={distinct_names} operation_name_bytes={len(operation_table.name_bytes)} operation_table_payload_bytes={packed_payload_bytes} operation_table_tracemalloc_retained_growth={operation_table_retained_growth} operation_table_direct_pybytes_fill=true operation_table_temporary_column_duplication_bytes=0 stored_u32_filter=true old_eager_snapshot_cost=see_processing_benchmarks_doc old_eager_wrapper_cost=see_processing_benchmarks_doc lower_ns={lower_ns} direct_owned_bytes={stats.direct_owned_bytes}"
            )
            return

        def write_canonical_bytes_buffered() -> None:
            with canonical_buffered.open("wb") as output:
                output.write(document.canonical_bytes(compact=True))

        write_canonical_bytes_buffered()  # warm-up
        canonical_bytes_buffered_ns, _ = timed(runs, write_canonical_bytes_buffered)
        document.write_canonical(canonical, compact=True)  # warm-up
        canonical_ns, _ = timed(
            runs, lambda: document.write_canonical(canonical, compact=True)
        )
        document.write_custom(custom, compact=True)  # warm-up
        custom_ns, _ = timed(runs, lambda: document.write_custom(custom, compact=True))
        parsed.write_original(original)  # warm-up
        original_ns, _ = timed(runs, lambda: parsed.write_original(original))
        document.write_preserving(preserving, compact=True)  # warm-up
        preserving_ns, _ = timed(
            runs, lambda: document.write_preserving(preserving, compact=True)
        )
        stats = document.statistics()
        print(
            f"measurement shape={args.shape} parse_ns={parse_ns} syntax_table_ns={syntax_table_ns} syntax_table_payload_bytes={syntax_payload_bytes} syntax_table_tracemalloc_retained_growth={syntax_table_retained_growth} syntax_table_direct_pybytes_fill=true syntax_operation_table_ns={syntax_operation_table_ns} syntax_operation_traversal_ns={syntax_operation_traversal_ns} syntax_operation_payload_bytes={syntax_operation_payload_bytes} syntax_operation_tracemalloc_retained_growth={syntax_operation_retained_growth} syntax_operation_direct_pybytes_fill=true syntax_operation_temporary_column_duplication_bytes=0 operation_table_ns={operation_table_ns} operation_filter_ns={filter_ns} operations={operation_table.count} filtered_operations={filtered.count} distinct_operation_names={distinct_names} operation_name_bytes={len(operation_table.name_bytes)} operation_table_payload_bytes={packed_payload_bytes} operation_table_tracemalloc_retained_growth={operation_table_retained_growth} operation_table_direct_pybytes_fill=true operation_table_temporary_column_duplication_bytes=0 stored_u32_filter=true old_eager_snapshot_cost=see_processing_benchmarks_doc old_eager_wrapper_cost=see_processing_benchmarks_doc operation_components={component_count} lower_ns={lower_ns} canonical_bytes_buffered_file_ns={canonical_bytes_buffered_ns} canonical_file_ns={canonical_ns} custom_file_ns={custom_ns} original_file_ns={original_ns} preserving_file_ns={preserving_ns} canonical_bytes={canonical.stat().st_size} custom_bytes={custom.stat().st_size} original_bytes={original.stat().st_size} preserving_bytes={preserving.stat().st_size} direct_owned_bytes={stats.direct_owned_bytes} document_index_bytes={stats.document_index_bytes} retained_source_bytes={stats.retained_source_bytes} retained_cst_bytes={stats.retained_cst_bytes} source_storage_shared={stats.source_storage_shared} cst_storage_shared={stats.cst_storage_shared} temporary_storage=true"
        )


if __name__ == "__main__":
    main()
