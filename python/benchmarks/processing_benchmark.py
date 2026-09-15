#!/usr/bin/env python3
"""Small file-oriented companion to the Rust processing benchmark."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import zirium

MIB = 1024 * 1024
SEED = 0x5A495249554D0028


def peak_rss_bytes() -> int:
    """Return ru_maxrss in bytes on the benchmark's supported host platforms."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(peak)
    if sys.platform.startswith("linux"):
        return int(peak) * 1024
    raise RuntimeError(f"RSS benchmark does not normalize ru_maxrss on {sys.platform}")


def current_rss_bytes() -> int:
    """Return current resident bytes for baselines around source loading."""
    if sys.platform == "darwin":
        import ctypes

        process_info = (ctypes.c_uint64 * 12)()
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
        returned = libproc.proc_pidinfo(
            os.getpid(), 4, 0, ctypes.byref(process_info), ctypes.sizeof(process_info)
        )
        if returned != ctypes.sizeof(process_info):
            raise RuntimeError("proc_pidinfo did not return PROC_PIDTASKINFO")
        return int(process_info[1])
    if sys.platform.startswith("linux"):
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    raise RuntimeError(f"RSS benchmark does not read current RSS on {sys.platform}")


def semantic_nesting_depth(document) -> int:
    table = document.operation_table()
    stack = [
        (table.operation(index), 1)
        for index, is_root in enumerate(table.root_flags)
        if is_root
    ]
    maximum = 0
    while stack:
        operation, depth = stack.pop()
        maximum = max(maximum, depth)
        for region_index in range(operation.region_count()):
            region = operation.region(region_index)
            for block_index in range(region.block_count()):
                block = region.block(block_index)
                stack.extend(
                    (block.operation(index), depth + 1)
                    for index in range(block.operation_count())
                )
    return maximum


def rss_child(stage: str, source_path: Path, max_delimiter_depth: int | None) -> None:
    imported_baseline = current_rss_bytes()
    input_bytes = source_path.stat().st_size
    source = source_path.read_text(encoding="utf-8")
    source_resident = current_rss_bytes()
    parsed = None
    document = None
    parse_started = time.perf_counter_ns()
    if stage != "source":
        parsed = zirium.parse_text(
            source,
            registry=zirium.DialectRegistry.baseline(),
            max_delimiter_depth=max_delimiter_depth,
        )
    parse_ns = time.perf_counter_ns() - parse_started if parsed is not None else 0
    parse_peak = peak_rss_bytes()
    lowering_ns = 0
    stage_ns = 0
    stage_baseline = current_rss_bytes()
    stage_peak_before = peak_rss_bytes()
    retained_semantic_bytes = 0
    report_bytes = 0
    checksum = 0
    if stage in {"lower", "semantic-retained", "traverse", "adapt"}:
        assert parsed is not None
        lowering_started = time.perf_counter_ns()
        lowered = parsed.lower_best_effort("semantic")
        lowering_ns = time.perf_counter_ns() - lowering_started
        assert lowered.document is not None
        document = lowered.document
    lower_peak = peak_rss_bytes()
    if stage in {"semantic-retained", "traverse", "adapt"}:
        del lowered
        parsed = None
        source = ""
        gc.collect()
        retained_semantic_bytes = current_rss_bytes()
        stage_baseline = retained_semantic_bytes
        stage_peak_before = peak_rss_bytes()
    if stage == "traverse":
        assert document is not None
        stage_baseline = current_rss_bytes()
        stage_peak_before = peak_rss_bytes()
        started = time.perf_counter_ns()
        table = document.operation_table()
        for index in range(table.count):
            operation = table.operation(index)
            checksum += (
                len(operation.name)
                + operation.operand_count()
                + operation.result_count()
                + sum(
                    len(name) + len(spelling)
                    for name, spelling in operation.attribute_snapshot()
                )
            )
        stage_ns = time.perf_counter_ns() - started
    elif stage == "adapt":
        assert document is not None
        stage_baseline = current_rss_bytes()
        stage_peak_before = peak_rss_bytes()
        started = time.perf_counter_ns()
        table = document.operation_table()
        report = [
            {
                "name": operation.name,
                "operands": operation.operand_count(),
                "results": operation.result_count(),
                "attributes": operation.attribute_snapshot(),
            }
            for operation in (table.operation(index) for index in range(table.count))
        ]
        report_bytes = len(json.dumps(report, separators=(",", ":")))
        stage_ns = time.perf_counter_ns() - started
    stage_peak = peak_rss_bytes()
    operation_count = parsed.operation_count if parsed is not None else 0
    operand_count = 0
    if parsed is not None:
        offsets = memoryview(parsed.operation_table().operand_offsets).cast("I")
        operand_count = offsets[-1] if offsets else 0
    elif document is not None:
        table = document.operation_table()
        operation_count = table.count
        operand_count = sum(
            table.operation(index).operand_count() for index in range(table.count)
        )
    stats = document.statistics() if document is not None else None
    nesting_depth = semantic_nesting_depth(document) if document is not None else 0
    print(
        json.dumps(
            {
                "stage": stage,
                "input_bytes": input_bytes,
                "operations": operation_count,
                "operands": operand_count,
                "regions": stats.regions if stats is not None else 0,
                "blocks": stats.blocks if stats is not None else 0,
                "nesting_depth": nesting_depth,
                "imported_process_baseline_bytes": imported_baseline,
                "source_resident_baseline_bytes": source_resident,
                "parse_peak_bytes": parse_peak,
                "lower_peak_bytes": lower_peak,
                "retained_semantic_resident_bytes": retained_semantic_bytes,
                "stage_baseline_bytes": stage_baseline,
                "stage_peak_before_bytes": stage_peak_before,
                "stage_peak_bytes": stage_peak,
                "parse_ns": parse_ns,
                "lower_ns": lowering_ns,
                "stage_ns": stage_ns,
                "report_bytes": report_bytes,
                "checksum": checksum,
            },
            sort_keys=True,
        )
    )


def rss_measurements(source_path: Path, max_delimiter_depth: int | None) -> None:
    if not source_path.is_file():
        raise ValueError(f"RSS input is not a file: {source_path}")
    if source_path.stat().st_size == 0:
        raise ValueError("RSS input must not be empty")
    print(
        "benchmark=python-processing-rss "
        f"python={platform.python_version()} platform={platform.platform()} "
        f"source_path={source_path.resolve()} threshold=none"
    )
    for stage in (
        "source",
        "parse",
        "lower",
        "semantic-retained",
        "traverse",
        "adapt",
    ):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--rss-input",
            str(source_path.resolve()),
            "--rss-child-stage",
            stage,
        ]
        if max_delimiter_depth is not None:
            command.extend(["--rss-max-delimiter-depth", str(max_delimiter_depth)])
        result = json.loads(subprocess.check_output(command, text=True))
        size = result["input_bytes"]
        imported_delta = (
            result["stage_peak_bytes"] - result["imported_process_baseline_bytes"]
        )
        source_delta = (
            result["stage_peak_bytes"] - result["source_resident_baseline_bytes"]
        )
        print(
            f"rss_measurement stage={stage} input_bytes={size} "
            f"operations={result['operations']} operands={result['operands']} "
            f"regions={result['regions']} blocks={result['blocks']} "
            f"nesting_depth={result['nesting_depth']} "
            f"imported_process_baseline_bytes={result['imported_process_baseline_bytes']} "
            f"source_resident_baseline_bytes={result['source_resident_baseline_bytes']} "
            f"parse_peak_bytes={result['parse_peak_bytes']} "
            f"lower_peak_bytes={result['lower_peak_bytes']} "
            f"retained_semantic_resident_bytes={result['retained_semantic_resident_bytes']} "
            f"stage_baseline_bytes={result['stage_baseline_bytes']} "
            f"stage_peak_before_bytes={result['stage_peak_before_bytes']} "
            f"stage_peak_bytes={result['stage_peak_bytes']} "
            f"additional_peak_from_imported_bytes={imported_delta} "
            f"additional_peak_from_source_resident_bytes={source_delta} "
            f"additional_peak_from_imported_per_input={imported_delta / size:.3f} "
            f"additional_peak_from_source_resident_per_input={source_delta / size:.3f} "
            f"parse_ns={result['parse_ns']} lower_ns={result['lower_ns']} "
            f"stage_ns={result['stage_ns']} report_bytes={result['report_bytes']} "
            f"checksum={result['checksum']}"
        )


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


def nested_fixture(size: int, depth: int) -> bytes:
    opening = b'"bench.region"() ({\n'
    closing = b"}) : () -> ()\n"
    result = bytearray()
    for _ in range(depth):
        result += opening
    result += b'"bench.op"() : () -> ()\n'
    closing_bytes = len(closing) * depth
    if len(result) + closing_bytes > size:
        raise ValueError("requested depth does not fit fixture size")
    result += b" " * (size - len(result) - closing_bytes)
    for _ in range(depth):
        result += closing
    return bytes(result)


def repeated_values_fixture(size: int) -> bytes:
    prefix = b"builtin.module {\n"
    line = b'"bench.op"() {enabled = true, tag = "same"} : () -> tensor<4x8xf32>\n'
    suffix = b"}\n"
    result = bytearray(prefix)
    while len(result) + len(line) + len(suffix) <= size:
        result += line
    result += b" " * (size - len(result) - len(suffix)) + suffix
    return bytes(result)


def long_operands_fixture(size: int) -> bytes:
    prefix = b'"bench.container"() ({\n%seed = "bench.source"() : () -> i32\n'
    operands = b", ".join([b"%seed"] * 64)
    types = b", ".join([b"i32"] * 64)
    line = b'"bench.use"(' + operands + b") : (" + types + b") -> ()\n"
    suffix = b"}) : () -> ()\n"
    result = bytearray(prefix)
    while len(result) + len(line) + len(suffix) <= size:
        result += line
    result += b" " * (size - len(result) - len(suffix)) + suffix
    return bytes(result)


def opaque_payload_fixture(size: int) -> bytes:
    prefix = b'"bench.payload"() {value = #vendor.attr<"'
    suffix = b'">} : () -> ()\n'
    if len(prefix) + len(suffix) > size:
        raise ValueError("fixture is too small for an opaque payload")
    return prefix + b"x" * (size - len(prefix) - len(suffix)) + suffix


def build_fixture(shape: str, size: int, depth: int | None) -> bytes:
    if shape == "nested":
        assert depth is not None
        return nested_fixture(size, depth)
    if shape == "block-rich":
        return block_rich_fixture(size)
    if shape == "repeated-values":
        return repeated_values_fixture(size)
    if shape == "long-operands":
        return long_operands_fixture(size)
    if shape == "opaque-payload":
        return opaque_payload_fixture(size)
    return fixture(size)


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
    parser.add_argument(
        "--shape",
        choices=(
            "primary",
            "block-rich",
            "nested",
            "long-operands",
            "repeated-values",
            "opaque-payload",
        ),
        default="primary",
    )
    parser.add_argument("--depth", type=int)
    parser.add_argument(
        "--write-fixture",
        type=Path,
        help="write a deterministic fixture for a later RSS run",
    )
    parser.add_argument(
        "--rss-input",
        type=Path,
        help="measure fresh-process RSS for an existing MLIR file",
    )
    parser.add_argument(
        "--rss-child-stage",
        choices=(
            "source",
            "parse",
            "lower",
            "semantic-retained",
            "traverse",
            "adapt",
        ),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--rss-max-delimiter-depth",
        type=int,
        help="override the parse delimiter limit for an RSS input",
    )
    args = parser.parse_args()
    if args.rss_child_stage:
        if args.rss_input is None:
            parser.error("--rss-child-stage requires --rss-input")
        rss_child(args.rss_child_stage, args.rss_input, args.rss_max_delimiter_depth)
        return
    if args.rss_input is not None:
        rss_measurements(args.rss_input, args.rss_max_delimiter_depth)
        return
    if args.rss_max_delimiter_depth is not None:
        parser.error("--rss-max-delimiter-depth requires --rss-input")
    size = 64 * 1024 if args.smoke else args.size_mib * MIB
    runs = 1 if args.smoke else args.runs
    if size == 500 * MIB:
        parser.error("500 MiB is projection-only")
    if args.shape == "nested" and args.depth is None:
        parser.error("--shape nested requires --depth")
    if args.depth is not None and args.depth <= 0:
        parser.error("--depth must be positive")
    if args.shape != "nested" and args.depth is not None:
        parser.error("--depth requires --shape nested")
    if args.write_fixture is not None:
        contents = build_fixture(args.shape, size, args.depth)
        args.write_fixture.write_bytes(contents)
        print(
            f"fixture_path={args.write_fixture.resolve()} shape={args.shape} "
            f"depth={args.depth} input_bytes={len(contents)} seed=0x{SEED:016x}"
        )
        return
    if args.shape not in {"primary", "block-rich"}:
        parser.error(f"--shape {args.shape} is available only with --write-fixture")
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
        registry = zirium.DialectRegistry.baseline()
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
