import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, TypeAlias

import pytest
import zirium

Retention: TypeAlias = Literal[
    "semantic", "semantic_only", "syntax", "syntax_only", "hybrid"
]


def operations(document: zirium.Document) -> list[zirium.SemanticOperation]:
    table = document.operation_table()
    return [table.operation(index) for index in range(table.count)]


def test_operation_table_survives_edits_but_lazy_erased_handle_is_stale():
    doc = generic_document('"dead"() : () -> ()\n"live"() : () -> ()')
    table = doc.operation_table()
    saved_names = table.name_bytes
    dead = table.operation(0)
    with doc.edit() as edit:
        edit.erase(dead)
    assert table.name_bytes == saved_names
    assert table.count == 2
    with pytest.raises(zirium.StaleHandleError):
        table.operation(0)
    assert table.operation(1).name == "live"


ROOT = Path(__file__).parents[2]
VALID = ROOT / "tests/corpus/mlir-22.1/semantic-proving/valid.mlir"
FORWARD = ROOT / "tests/corpus/mlir-22.1/semantic-proving/forward.mlir"
UNRESOLVED = ROOT / "tests/corpus/mlir-22.1/semantic-proving/unresolved.mlir"
SUCCESSOR_SOURCE = """\
"container"() ({
^entry:
  %v = "def"() : () -> index
  "jump"(%v) [^next : (%v : index)] : (index) -> ()
^next(%arg : index):
  "end"() : () -> ()
}) : () -> ()
"""
PROPERTIES_SOURCE = (
    '"test.properties"() <{inherent = 7}> {discardable = "yes"} : () -> ()'
)


def document(*, retention: Retention = "hybrid") -> zirium.Document:
    lowered = zirium.parse_file(VALID).lower_strict(retention)
    assert lowered.document is not None
    return lowered.document


def generic_document(
    source: str, *, retention: Retention = "semantic"
) -> zirium.Document:
    lowered = zirium.parse_text(
        source, registry=zirium.DialectRegistry.proving()
    ).lower_strict(retention)
    assert lowered.document is not None, lowered.diagnostics
    return lowered.document


def test_edit_commit_keeps_selected_proving_registry():
    lowered = zirium.parse_text(
        "%value = arith.constant 7 : i32", registry=zirium.DialectRegistry.proving()
    ).lower_strict()
    assert lowered.document is not None, lowered.diagnostics
    doc = lowered.document
    with doc.edit() as edit:
        edit.compact_pools()
    doc.verify_semantics()
    assert doc.custom_bytes() == b"%v0 = arith.constant 7 : i32\n"


def test_edit_commit_keeps_declarative_registry():
    registry = zirium.DialectRegistry.declarative(["arith.constant"])
    lowered = zirium.parse_text(
        "%value = arith.constant 7 : i32", registry=registry
    ).lower_strict()
    assert lowered.document is not None, lowered.diagnostics
    doc = lowered.document
    with doc.edit() as edit:
        edit.compact_pools()
    doc.verify_semantics()
    assert b"arith.constant 7 : i32" in doc.custom_bytes()


def test_buffered_rauw_replaces_a_used_value_and_invalidates_use_index():
    doc = generic_document(
        '%a = "def.a"() : () -> i32\n'
        '%b = "def.b"() : () -> i32\n'
        '"use"(%a) : (i32) -> ()'
    )
    a, b, use = operations(doc)
    old_value, replacement = a.result(0), b.result(0)
    assert len(doc.uses(old_value)) == 1
    assert doc.statistics().use_index_entries == 1

    with doc.edit() as edit:
        edit.replace_all_uses(old_value, replacement)

    assert doc.statistics().use_index_entries == 0
    assert doc.uses(old_value) == []
    assert [
        (site.operation.name, site.kind, site.index) for site in doc.uses(replacement)
    ] == [("use", "operand", 0)]
    assert use.operand(0).valid
    assert doc.statistics().use_index_entries == 1


def test_erased_handles_are_stale_and_failed_transaction_is_atomic():
    doc = document()
    make = doc.operation_table("vendor.make").operation(0)
    consume = doc.operation_table("vendor.consume").operation(0)
    before = doc.canonical_bytes()
    with (
        pytest.raises(zirium.SemanticEditError, match="still has live uses"),
        doc.edit() as edit,
    ):
        edit.erase(make)
    assert doc.canonical_bytes() == before

    with doc.edit() as edit:
        edit.erase(consume)
    with pytest.raises(zirium.StaleHandleError, match="stale"):
        _ = consume.name


def test_failed_transaction_is_atomic_and_exceptions_are_specific():
    doc = document()
    make = doc.operation_table("vendor.make").operation(0)
    before = doc.canonical_bytes()

    with (
        pytest.raises(zirium.SemanticEditError, match="still has live uses"),
        doc.edit() as edit,
    ):
        edit.erase(make)

    assert doc.canonical_bytes() == before
    assert make.name == "vendor.make"

    other = document()
    with (
        pytest.raises(zirium.ForeignHandleError, match="another document"),
        doc.edit() as edit,
    ):
        edit.erase(operations(other)[0])


def test_exception_in_edit_body_discards_commands_and_preserves_output_and_stats():
    doc = generic_document('%a = "a"() : () -> i32')
    operation = operations(doc)[0]
    before_output = doc.canonical_bytes()
    before_stats = doc.statistics().pooled_list_entries

    with pytest.raises(RuntimeError, match="body failure"), doc.edit() as edit:
        edit.remove_attribute(operation, "not_present")
        raise RuntimeError("body failure")

    assert doc.canonical_bytes() == before_output
    assert doc.statistics().pooled_list_entries == before_stats
    assert operation.name == "a"


def test_lazy_queries_verification_statistics_and_output(tmp_path: Path):
    doc = document()
    make = doc.operation_table("vendor.make").operation(0)
    consume = doc.operation_table("vendor.consume").operation(0)
    value = make.result(0)
    assert doc.statistics().use_index_entries == 0
    uses = doc.uses(value)
    assert [(use.kind, use.operation.name, use.index) for use in uses] == [
        ("operand", "vendor.consume", 0)
    ]
    assert doc.statistics().use_index_entries == 1
    assert doc.dominates(value, consume)
    assert doc.symbol_diagnostics() == []
    doc.validate_structure()
    doc.verify_semantics()

    canonical = tmp_path / "canonical.mlir"
    preserving = tmp_path / "preserving.mlir"
    doc.write_canonical(canonical)
    doc.write_preserving(preserving)
    assert canonical.read_bytes() == doc.canonical_bytes()
    assert preserving.read_bytes() == VALID.read_bytes()


def test_preserving_file_sink_failure_is_an_oserror(tmp_path: Path):
    doc = document()
    with pytest.raises(OSError):
        doc.write_preserving(tmp_path / "missing" / "preserving.mlir")


def test_buffering_holds_no_document_lock_across_python_execution():
    doc = document()
    consume = doc.operation_table("vendor.consume").operation(0)
    with doc.edit() as edit:
        edit.erase(consume)
        with ThreadPoolExecutor(max_workers=2) as pool:
            outputs = list(pool.map(lambda _: doc.canonical_bytes(), range(8)))
        assert len(set(outputs)) == 1


def test_bounded_operation_and_attribute_specs_snapshot_existing_values():
    doc = generic_document(
        '!fn = type () -> (i32)\n%f = "fn"() : () -> !fn\n'
        '%v = "value"() {tag = #vendor.tag<"x">} : () -> i32'
    )
    function_type = doc.operation_table("fn").operation(0).result_type(0)
    value = doc.operation_table("value").operation(0)
    result_type = value.result_type(0)
    tag = zirium.AttributeSpecHandle(value.attribute(0), "copied_tag")
    spec = zirium.OperationSpec("vendor.copy", [], [result_type], function_type, [tag])

    with doc.edit() as edit:
        edit.insert_root(1, spec)

    inserted = doc.operation_table("vendor.copy").operation(0)
    assert inserted.attribute_snapshot() == [("copied_tag", '#vendor.tag<"x">')]
    assert inserted.result_count() == 1
    inserted_table = doc.operation_table("vendor.copy")
    assert int.from_bytes(inserted_table.source_start, sys.byteorder) == 0xFFFFFFFF
    assert int.from_bytes(inserted_table.source_end, sys.byteorder) == 0xFFFFFFFF


def test_operand_and_successor_rewiring_are_buffered_and_indexed():
    doc = generic_document(
        '%a = "a"() : () -> i32\n%b = "b"() : () -> i32\n"use"(%a) : (i32) -> ()'
    )
    a, b, use = operations(doc)
    old, replacement = a.result(0), b.result(0)
    with doc.edit() as edit:
        edit.rewire_operand(use, 0, replacement)
    assert doc.uses(old) == []
    assert doc.uses(replacement)[0].operation.name == "use"

    successor_doc = generic_document(SUCCESSOR_SOURCE)
    successor = successor_doc.operation_table("jump").operation(0)
    value = successor.operand(0)
    before = len(successor_doc.uses(value))
    with successor_doc.edit() as edit:
        edit.rewire_successor_argument(successor, 0, 0, value)
    assert len(successor_doc.uses(value)) == before


def test_erased_handles_do_not_become_negative_query_results():
    doc = generic_document('%dead = "dead"() : () -> i32\n"live"() : () -> ()')
    dead, live = operations(doc)
    stale_value = dead.result(0)
    with doc.edit() as edit:
        edit.erase(dead)

    assert not stale_value.valid
    with pytest.raises(zirium.StaleHandleError):
        _ = stale_value.type_value
    with pytest.raises(zirium.StaleHandleError):
        doc.uses(stale_value)
    with pytest.raises(zirium.StaleHandleError):
        doc.dominates(stale_value, live)
    with pytest.raises(zirium.StaleHandleError):
        doc.lookup_symbol(dead, "@missing")


def test_fixed_result_types_attrs_properties_and_pool_compaction():
    doc = generic_document('%a = "a"() : () -> i32\n%b = "b"() : () -> i64')
    first, second = operations(doc)
    result = first.result(0)
    original = doc.canonical_bytes()
    with (
        pytest.raises(
            zirium.SemanticVerificationError,
            match="result types do not match the stored function type outputs",
        ),
        doc.edit() as edit,
    ):
        edit.replace_result_types(
            first, [second.result_type(i) for i in range(second.result_count())]
        )
    assert first.result(0).valid
    assert first.result_type(0).spelling == "i32"
    result_type = result.type_value
    assert result_type is not None
    assert result_type.spelling == "i32"
    assert doc.canonical_bytes() == original

    properties_doc = generic_document(PROPERTIES_SOURCE)
    operation = properties_doc.operation_table("test.properties").operation(0)
    source_attribute = operation.attribute(0)
    attribute = zirium.AttributeSpecHandle(source_attribute, "copied")
    with properties_doc.edit() as edit:
        edit.set_attribute(operation, attribute)
        edit.set_property(operation, attribute)
    assert ("copied", '"yes"') in operation.attribute_snapshot()
    assert operation.property_snapshot() == [("inherent", "7"), ("copied", '"yes"')]

    before_compaction = properties_doc.statistics().pooled_list_entries
    with properties_doc.edit() as edit:
        edit.set_attribute(operation, attribute)
        edit.set_attribute(
            operation, zirium.AttributeSpecHandle(source_attribute, "copied_again")
        )
        edit.compact_pools()
    assert properties_doc.statistics().pooled_list_entries <= before_compaction + 2
    assert operation.name == "test.properties"


def test_semantic_verification_failures_have_distinct_kinds_and_classes():
    incomplete = zirium.parse_file(UNRESOLVED).lower_best_effort("semantic")
    assert incomplete.document is not None
    incomplete.document.validate_structure()
    with pytest.raises(
        zirium.SemanticVerificationError, match="contains invalid values"
    ):
        incomplete.document.verify_semantics()

    dominance = zirium.parse_file(FORWARD).lower_strict("semantic")
    assert dominance.document is not None
    dominance.document.validate_structure()
    with pytest.raises(
        zirium.SemanticVerificationError, match="SSA definition does not dominate"
    ):
        dominance.document.verify_semantics()

    unresolved_call = generic_document('"func.call"() {callee = @missing} : () -> ()')
    with pytest.raises(
        zirium.SemanticVerificationError, match="callee does not resolve"
    ):
        unresolved_call.verify_semantics()

    schema = generic_document('%x = "arith.constant"() {value = 1 : i32} : () -> i32')
    constant = operations(schema)[0]
    with (
        pytest.raises(zirium.SemanticVerificationError, match="violates its schema"),
        schema.edit() as edit,
    ):
        edit.remove_attribute(constant, "value")

    bad_operation = generic_document(
        '%x = "arith.constant"() {value = "bad"} : () -> i32'
    )
    with pytest.raises(zirium.SemanticVerificationError, match="failed verification"):
        bad_operation.verify_semantics()
