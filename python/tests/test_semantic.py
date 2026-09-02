import os
from array import array
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, TypeAlias

import pytest
import zirium

Retention: TypeAlias = Literal[
    "semantic", "semantic_only", "syntax", "syntax_only", "hybrid"
]

ROOT = Path(__file__).parents[2]
VALID = ROOT / "tests/corpus/mlir-22.1/semantic-proving/valid.mlir"
UNRESOLVED = ROOT / "tests/corpus/mlir-22.1/semantic-proving/unresolved.mlir"
PAYLOADS = ROOT / "tests/corpus/mlir-22.1/payload-opaque/valid.mlir"


def lower(path: Path, *, retention: Retention = "semantic") -> zirium.LoweringResult:
    return zirium.parse_file(path).lower_strict(retention)


def operations(document: zirium.Document) -> list[zirium.SemanticOperation]:
    table = document.operation_table()
    return [table.operation(index) for index in range(table.count)]


def u32s(value: bytes) -> list[int]:
    result = array("I")
    result.frombytes(value)
    return list(result)


def test_operation_table_dictionary_filter_columns_and_indices():
    document = (
        zirium.parse_text(
            '"same"() : () -> ()\n"other"() : () -> ()\n"same"() : () -> ()'
        )
        .lower_strict()
        .document
    )
    assert document is not None
    table = document.operation_table()
    assert table.count == 3
    assert u32s(table.name_code) == [0, 1, 0]
    offsets = u32s(table.name_offsets)
    assert [table.name_bytes[offsets[i] : offsets[i + 1]] for i in range(2)] == [
        b"same",
        b"other",
    ]
    assert table.root_flags == b"\x01\x01\x01"
    assert all(value != 0xFFFFFFFF for value in u32s(table.source_start))
    assert all(value != 0xFFFFFFFF for value in u32s(table.source_end))
    assert table.operation(1).name == "other"
    with pytest.raises(IndexError):
        table.operation(3)

    revision = table.revision
    before = document.statistics().direct_owned_bytes
    missing = document.operation_table("not.in.document")
    assert missing.count == 0
    assert missing.name_bytes == b""
    assert u32s(missing.name_offsets) == [0]
    assert missing.revision == revision
    assert document.operation_table().revision == revision
    assert document.statistics().direct_owned_bytes == before
    for removed in (
        "operations",
        "root_operations",
        "operations_named",
        "operation_names",
        "operation_snapshot",
    ):
        assert not hasattr(document, removed)
    operation = table.operation(0)
    for removed in (
        "regions",
        "operands",
        "results",
        "result_types",
        "attributes",
        "blocks",
    ):
        assert not hasattr(operation, removed)


def test_runtime_and_stub_expose_only_packed_and_indexed_semantic_surface():
    removed_document = {
        "operations",
        "root_operations",
        "operations_named",
        "operation_names",
        "operation_snapshot",
    }
    removed_operation = {
        "regions",
        "operands",
        "results",
        "result_types",
        "attributes",
        "blocks",
    }
    assert removed_document.isdisjoint(dir(zirium.Document))
    assert removed_operation.isdisjoint(dir(zirium.SemanticOperation))
    assert {"blocks"}.isdisjoint(dir(zirium.SemanticRegion))
    assert {"operations", "arguments"}.isdisjoint(dir(zirium.SemanticBlock))
    assert {
        "revision",
        "name_code",
        "source_start",
        "source_end",
        "root_flags",
        "name_offsets",
        "name_bytes",
        "count",
        "operation",
    }.issubset(dir(zirium.OperationTable))

    stub = Path(zirium.__file__).with_name("__init__.pyi").read_text()
    for declaration in (
        "class OperationTable:",
        "def operation_table(self, name: str | None = None) -> OperationTable:",
        "def region_count(self) -> int:",
        "def region(self, index: int) -> SemanticRegion:",
        "def operand_count(self) -> int:",
        "def operand(self, index: int) -> SemanticValue:",
        "def result_count(self) -> int:",
        "def result(self, index: int) -> SemanticValue:",
        "def result_type(self, index: int) -> SemanticType:",
        "def attribute_count(self) -> int:",
        "def attribute(self, index: int) -> SemanticAttribute:",
        "def block_count(self) -> int:",
        "def block(self, index: int) -> SemanticBlock:",
        "def operation_count(self) -> int:",
        "def argument_count(self) -> int:",
        "def argument(self, index: int) -> SemanticValue:",
    ):
        assert declaration in stub


def test_semantic_attribute_depth_limit_strict_and_best_effort():
    nested = "1"
    for depth in range(12):
        nested = f"[{nested}]" if depth % 2 == 0 else f"{{k = {nested}}}"
    parsed = zirium.parse_text(
        f'"deep"() {{a = {nested}}} : () -> ()', max_attribute_depth=3
    )
    strict = parsed.lower_strict()
    assert strict.document is None
    assert any(
        "attribute nesting depth limit" in diagnostic.message
        for diagnostic in strict.diagnostics
    )

    best = parsed.lower_best_effort()
    assert best.document is not None
    best.document.validate_structure()
    assert any(
        "attribute nesting depth limit" in diagnostic.message
        for diagnostic in best.diagnostics
    )


def test_lowering_ownership_handles_queries_and_concurrent_reads():
    result = lower(VALID, retention="hybrid")
    document = result.document
    assert document is not None
    operation = document.operation_table("vendor.make").operation(0)
    del result, document

    assert operation.name == "vendor.make"
    assert operation.result(0).kind == "operation_result"
    assert operation.result_type(0).kind == "opaque"
    assert operation.attribute_snapshot()[0][0] == "tag"

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert (
            list(pool.map(lambda _: operation.name, range(32))) == ["vendor.make"] * 32
        )


def test_retention_diagnostics_completeness_and_incomplete_preflight(tmp_path: Path):
    strict = zirium.parse_file(UNRESOLVED).lower_strict("syntax")
    assert strict.document is None
    assert not strict.semantically_complete
    assert strict.diagnostics

    best_effort = zirium.parse_file(UNRESOLVED).lower_best_effort("hybrid")
    assert best_effort.document is not None
    assert best_effort.document.retention == "hybrid"
    assert not best_effort.semantically_complete
    assert best_effort.diagnostics
    best_effort.document.validate_structure()
    with pytest.raises(ValueError, match="incomplete"):
        best_effort.document.canonical_bytes()
    output = tmp_path / "invalid.mlir"
    output.write_bytes(b"keep me")
    with pytest.raises(OSError, match="incomplete"):
        best_effort.document.write_canonical(output)
    assert output.read_bytes() == b"keep me"
    with pytest.raises(OSError, match="incomplete"):
        best_effort.document.write_custom(output)
    assert output.read_bytes() == b"keep me"


def test_dense_sparse_and_resource_payloads_are_raw_read_only_buffers():
    document = lower(PAYLOADS).document
    assert document is not None
    operation = operations(document)[0]
    attributes = {
        operation.attribute(i).name: operation.attribute(i)
        for i in range(operation.attribute_count())
    }
    for name, kind in (
        ("dense_value", "dense"),
        ("sparse_value", "sparse"),
        ("resource_value", "resource"),
    ):
        attribute = attributes[name]
        payload = attribute.raw_buffer()
        assert attribute.kind == kind
        assert isinstance(payload, bytes)
        assert attribute.payload_byte_length == len(payload)
        assert memoryview(payload).readonly


def test_canonical_round_trip_and_structural_equality(tmp_path: Path):
    first = lower(VALID).document
    assert first is not None
    first.validate_structure()
    output = tmp_path / "canonical.mlir"
    first.write_canonical(output)
    assert output.read_bytes() == first.canonical_bytes()

    second = zirium.parse_file(output).lower_strict().document
    assert second is not None
    assert first.structurally_equal(second)
    assert [operation.name for operation in operations(first)] == [
        operation.name for operation in operations(second)
    ]
    assert first.operation_table().count > 0


def test_proving_registry_is_used_for_lowering_verification_and_custom_printing():
    registry = zirium.DialectRegistry.proving()
    result = zirium.parse_text(
        "%value = arith.constant 7 : i32", registry=registry
    ).lower_strict()
    del registry
    document = result.document
    assert document is not None, result.diagnostics
    document.verify_semantics()
    assert b'"arith.constant"()' in document.canonical_bytes()
    assert document.custom_bytes() == b"%v0 = arith.constant 7 : i32\n"


def test_custom_file_output_matches_custom_bytes(tmp_path: Path):
    registry = zirium.DialectRegistry.proving()
    document = (
        zirium.parse_text("%value = arith.constant 7 : i32", registry=registry)
        .lower_strict()
        .document
    )
    assert document is not None
    output = tmp_path / "custom.mlir"
    document.write_custom(output)
    assert output.read_bytes() == document.custom_bytes()


def test_formatted_file_output_reports_create_errors(tmp_path: Path):
    document = lower(VALID).document
    assert document is not None
    with pytest.raises(OSError):
        document.write_canonical(tmp_path / "missing" / "canonical.mlir")
    with pytest.raises(OSError):
        document.write_custom(tmp_path / "missing" / "custom.mlir")


@pytest.mark.skipif(
    not os.path.exists("/dev/full"), reason="platform has no full device"
)
def test_formatted_file_output_reports_buffer_flush_errors():
    document = lower(VALID).document
    assert document is not None
    with pytest.raises(OSError):
        document.write_canonical("/dev/full")
    with pytest.raises(OSError):
        document.write_custom("/dev/full")


@pytest.mark.skipif(
    not os.path.exists("/dev/full"), reason="platform has no full device"
)
def test_formatted_file_output_reports_buffered_write_errors():
    source = '"large.output"() : () -> ()\n' * 1024
    document = zirium.parse_text(source).lower_strict().document
    assert document is not None
    assert len(document.canonical_bytes()) > 8192
    with pytest.raises(OSError):
        document.write_canonical("/dev/full")
    with pytest.raises(OSError):
        document.write_custom("/dev/full")


def test_declarative_subset_is_owned_and_used_end_to_end():
    registry = zirium.DialectRegistry.declarative(
        ["builtin.module", "func.func", "func.return", "func.call"]
    )
    parsed = zirium.parse_text(
        "builtin.module { func.func @callee() { func.return } "
        "func.func @caller() { func.call @callee() : () -> () func.return } }",
        registry=registry,
    )
    del registry
    lowered = parsed.lower_strict()
    document = lowered.document
    assert document is not None, lowered.diagnostics
    document.verify_semantics()
    call = document.operation_table("func.call").operation(0)
    callee = document.lookup_symbol(call, "@callee")
    assert callee is not None
    assert callee.name == "func.func"
    assert b"func.call @callee()" in document.custom_bytes()
