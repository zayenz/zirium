from pathlib import Path

import pytest
import zirium

ROOT = Path(__file__).parents[2]
FIXTURE = ROOT / "tests/corpus/mlir-22.1/cross-contract/valid.mlir"
REGISTRY = zirium.DialectRegistry.baseline()


def test_construct_edit_verify_print_and_reparse_cross_contract():
    lowered = zirium.parse_file(FIXTURE, registry=REGISTRY).lower_strict("semantic")
    assert lowered.document is not None, lowered.diagnostics
    document = lowered.document

    lhs = document.operation_table("arith.constant").operation(0)
    value = lhs.attribute_by_name("value")
    assert value is not None
    replacement_spec = zirium.OperationSpec(
        "arith.constant",
        [],
        [lhs.result_type(0)],
        lhs.function_type(),
        [zirium.AttributeSpecHandle(value, "value")],
    )
    with document.edit() as edit:
        edit.insert_root(2, replacement_spec)

    replacement = document.operation_table("arith.constant").operation(2)
    before_failed_edit = document.canonical_bytes()
    with (
        pytest.raises(zirium.SemanticEditError, match="still has live uses"),
        document.edit() as edit,
    ):
        edit.erase(lhs)
    assert document.canonical_bytes() == before_failed_edit

    with document.edit() as edit:
        edit.replace_all_uses(lhs.result(0), replacement.result(0))
        edit.erase(lhs)

    document.verify_semantics()
    assert document.operation_table("arith.constant").count == 2

    for printed in (document.canonical_bytes(), document.custom_bytes()):
        reparsed = zirium.parse_bytes(printed, registry=REGISTRY)
        assert reparsed.diagnostics == []
        restored = reparsed.lower_strict("semantic")
        assert restored.document is not None, restored.diagnostics
        restored.document.verify_semantics()
        assert document.structurally_equal(restored.document)
