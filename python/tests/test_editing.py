import inspect
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
VALID = ROOT / "tests/corpus/mlir-22.1/semantic-baseline/valid.mlir"
FORWARD = ROOT / "tests/corpus/mlir-22.1/semantic-baseline/forward.mlir"
UNRESOLVED = ROOT / "tests/corpus/mlir-22.1/semantic-baseline/unresolved.mlir"
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
        source, registry=zirium.DialectRegistry.baseline()
    ).lower_strict(retention)
    assert lowered.document is not None, lowered.diagnostics
    return lowered.document


def test_edit_commit_keeps_selected_baseline_registry():
    lowered = zirium.parse_text(
        "%value = arith.constant 7 : i32", registry=zirium.DialectRegistry.baseline()
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


def test_replace_all_uses_accepts_documented_from_keyword():
    signature = inspect.signature(zirium.SemanticEdit.replace_all_uses)
    assert "from_" in signature.parameters
    assert "from" not in signature.parameters

    doc = generic_document(
        '%a = "def.a"() : () -> i32\n'
        '%b = "def.b"() : () -> i32\n'
        '"use"(%a) : (i32) -> ()'
    )
    a, b, _ = operations(doc)
    with doc.edit() as edit:
        edit.replace_all_uses(from_=a.result(0), to=b.result(0))

    assert doc.uses(a.result(0)) == []


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


def test_semantic_handles_have_stable_document_scoped_identity():
    doc = generic_document(
        '"container"() ({\n^entry(%arg : i32):\n'
        '%value = "value"() {tag = #vendor.tag<"x">} : () -> i32\n'
        "}) : () -> ()"
    )
    container = doc.operation_table("container").operation(0)
    region = container.region(0)
    block = region.block(0)
    value = block.operation(0).result(0)

    repeated = [
        doc.operation_table("container").operation(0),
        block.parent_region().parent_operation(),
    ]
    assert repeated == [container, container]
    assert len({container, *repeated}) == 1
    assert {container: "same"}[repeated[0]] == "same"
    assert len({region, container.region(0)}) == 1
    assert len({block, region.block(0)}) == 1
    assert len({value, block.operation(0).result(0)}) == 1
    assert len({block.argument(0), block.argument(0)}) == 1

    operation = block.operation(0)
    first_type, second_type = operation.result_type(0), operation.result_type(0)
    first_attribute, second_attribute = operation.attribute(0), operation.attribute(0)
    assert first_type is not second_type and first_type != second_type
    assert first_type.spelling == second_type.spelling == "i32"
    assert (
        first_attribute is not second_attribute and first_attribute != second_attribute
    )
    assert first_attribute.spelling == second_attribute.spelling == '#vendor.tag<"x">'

    other = generic_document(
        '"container"() ({\n^entry(%arg : i32):\n'
        '%value = "value"() {tag = #vendor.tag<"x">} : () -> i32\n'
        "}) : () -> ()"
    )
    other_container = other.operation_table("container").operation(0)
    assert container != other_container
    assert value != other_container.region(0).block(0).operation(0).result(0)


def test_stale_handle_identity_is_hashable_and_never_matches_reused_slot():
    doc = generic_document('"dead"() : () -> ()')
    dead = doc.operation_table().operation(0)
    same_dead = doc.operation_table().operation(0)
    dead_hash = hash(dead)
    spec = zirium.OperationSpec("replacement", [], [], dead.function_type())

    with doc.edit() as edit:
        edit.erase(dead)
    assert hash(dead) == dead_hash
    assert dead == same_dead
    assert dead in {dead}

    with doc.edit() as edit:
        edit.insert_root(0, spec)
    replacement = doc.operation_table().operation(0)
    assert dead != replacement
    assert len({dead, replacement}) == 2
    with pytest.raises(zirium.StaleHandleError):
        _ = dead.name


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


def test_operation_insertion_uses_an_existing_complete_function_type():
    doc = generic_document('%v = "value"() {tag = #vendor.tag<"x">} : () -> i32')
    value = doc.operation_table("value").operation(0)
    result_type = value.result_type(0)
    tag = zirium.AttributeSpecHandle(value.attribute(0), "copied_tag")
    spec = zirium.OperationSpec(
        "vendor.copy", [], [result_type], value.function_type(), [tag]
    )

    with doc.edit() as edit:
        edit.insert_root(1, spec)

    inserted = doc.operation_table("vendor.copy").operation(0)
    assert inserted.attribute_snapshot() == [("copied_tag", '#vendor.tag<"x">')]
    assert inserted.result_count() == 1
    assert inserted.result_type(0).spelling == "i32"
    assert inserted.function_type().spelling == "() -> i32"
    inserted_table = doc.operation_table("vendor.copy")
    assert int.from_bytes(inserted_table.source_start, sys.byteorder) == 0xFFFFFFFF
    assert int.from_bytes(inserted_table.source_end, sys.byteorder) == 0xFFFFFFFF

    canonical = doc.canonical_bytes()
    reparsed = zirium.parse_bytes(
        canonical, registry=zirium.DialectRegistry.baseline()
    ).lower_strict("semantic")
    assert reparsed.document is not None, reparsed.diagnostics
    assert doc.structurally_equal(reparsed.document)


def test_operation_function_type_is_checked_and_specs_reject_foreign_types():
    doc = generic_document('%value = "value"() : () -> i32')
    operation = doc.operation_table().operation(0)
    function_type = operation.function_type()
    assert function_type.kind == "function"

    foreign = generic_document('%other = "other"() : () -> i64')
    foreign_result_type = foreign.operation_table().operation(0).result_type(0)
    before = doc.canonical_bytes()
    with pytest.raises(zirium.ForeignHandleError):
        zirium.OperationSpec("copy", [], [foreign_result_type], function_type)
    assert doc.canonical_bytes() == before

    with doc.edit() as edit:
        edit.erase(operation)
    with pytest.raises(zirium.StaleHandleError):
        operation.function_type()


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
    doc = generic_document("%a = arith.constant 1 : i32\n%b = arith.constant 2 : i64")
    first, second = operations(doc)
    result = first.result(0)
    with doc.edit() as edit:
        edit.replace_result_types(
            first, [second.result_type(i) for i in range(second.result_count())]
        )
    assert first.result(0).valid
    result_type = result.type_value
    assert result_type is not None
    assert result_type.spelling == "i64"
    assert first.result_type(0).spelling == "i64"
    doc.verify_semantics()
    canonical = doc.canonical_bytes()
    assert b'"arith.constant"' in canonical
    assert b": () -> i64" in canonical
    assert b"arith.constant 1 : i64" in doc.custom_bytes()

    reparsed = zirium.parse_text(
        canonical.decode(), registry=zirium.DialectRegistry.baseline()
    ).lower_strict()
    assert reparsed.document is not None, reparsed.diagnostics
    reparsed.document.verify_semantics()
    reparsed_first = operations(reparsed.document)[0]
    assert reparsed_first.result_type(0).spelling == "i64"

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


def test_aliased_attribute_elements_have_printable_recursive_spellings():
    doc = generic_document(
        '#items = [1, {nested = [2]}]\n"test.alias"() {items = #items} : () -> ()'
    )
    operation = doc.operation_table("test.alias").operation(0)
    items = operation.attribute_by_name("items")
    assert items is not None
    first = items.element(0)
    nested_dictionary = items.element(1)
    assert first is not None and first.spelling == "1"
    assert (
        nested_dictionary is not None and nested_dictionary.spelling == "{nested = [2]}"
    )
    nested_array = nested_dictionary.element(0)
    assert nested_array is not None and nested_array.spelling == "[2]"
    nested_value = nested_array.element(0)
    assert nested_value is not None and nested_value.spelling == "2"

    with doc.edit() as edit:
        edit.set_attribute(
            operation, zirium.AttributeSpecHandle(nested_value, "copied")
        )
    assert ("copied", "2") in operation.attribute_snapshot()
    assert b"copied = 2" in doc.canonical_bytes()


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
