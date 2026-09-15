import json

import pytest
import zirium
from zirium.query import changed, changes, dialect, diff_op_input, op


def lower(source: str) -> zirium.Document:
    result = zirium.parse_text(
        source, registry=zirium.DialectRegistry.baseline()
    ).lower_strict()
    assert result.document is not None
    return result.document


def test_diff_owns_immutable_snapshots_and_exposes_changes():
    before = lower("module { %x = arith.constant 4 : i32 }")
    after = lower("module { %renamed = arith.constant 8 : i32 }")

    comparison = zirium.diff(before, after)
    assert len(comparison) == 1
    change = comparison.changes[0]
    assert change.kind == "modified"
    assert change.fields == ["attributes"]
    assert change.before is not None and change.before.name == "arith.constant"
    assert change.after is not None and change.after.name == "arith.constant"
    assert json.loads(comparison.to_json())[0]["kind"] == "modified"
    assert comparison.statistics["matched_operations"] == 2

    constants = changes().filter(changed("attributes") & dialect("arith"))
    assert [item.kind for item in comparison.query(constants)] == ["modified"]
    assert [operation.name for operation in comparison.query(constants.after())] == [
        "arith.constant"
    ]
    assert comparison.query(constants.count()) == 1
    assert comparison.query(constants.attr("value")) == ["8"]
    assert comparison.query(constants.result_types()) == ["i32"]
    assert comparison.query(constants.after().users().operand_types()) == ["i32"]
    assert comparison.query(changes().names().sort().min()) == ["arith.constant"]
    assert [
        operation.name
        for operation in comparison.query(constants.after().users().reachable())
    ] == ["test.use"]
    with pytest.raises(ValueError, match="cannot determine reference semantics"):
        comparison.query(constants.after().users().reachable(), strict=True)
    assert [
        operation.name
        for operation in comparison.query(
            constants.after().users().fixpoint(diff_op_input().slice())
        )
    ] == ["arith.constant", "test.use"]
    assert [
        operation.name for operation in comparison.query(constants.after().parent())
    ] == ["builtin.module"]
    assert [
        operation.name
        for operation in comparison.query(
            constants.after()
            .root(op("builtin.module"))
            .subtree()
            .filter(op("arith.constant"))
        )
    ] == ["arith.constant"]
    assert len(comparison.query(changes().reverse().head(1).tail(1))) == 1

    with after.edit() as edit:
        edit.erase(after.operation_table("arith.constant").operation(0))
    assert change.after.name == "arith.constant"


def test_diff_requires_one_registry_context():
    first_registry = zirium.DialectRegistry.declarative(["arith.constant"])
    second_registry = zirium.DialectRegistry.declarative(["arith.constant"])
    source = '"arith.constant"() : () -> ()'
    before = zirium.parse_text(source, registry=first_registry).lower_strict().document
    after = zirium.parse_text(source, registry=second_registry).lower_strict().document
    assert before is not None and after is not None
    try:
        zirium.diff(before, after)
    except ValueError as error:
        assert "same registry instance" in str(error)
    else:
        raise AssertionError("registry mismatch should fail")


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("max_work", True),
        ("max_work", 0),
        ("max_work", -1),
        ("max_changes", False),
        ("max_changes", 0),
        ("max_changes", -1),
    ],
)
def test_diff_rejects_non_positive_and_boolean_limits(keyword, value):
    document = lower("module {}")
    with pytest.raises(ValueError, match="positive integer"):
        zirium.diff(document, document, **{keyword: value})


@pytest.mark.parametrize("keyword,value", [("max_work", True), ("max_items", -1)])
def test_diff_query_rejects_invalid_limits(keyword, value):
    before = lower("module { %x = arith.constant 1 : i32 }")
    after = lower("module { %x = arith.constant 2 : i32 }")
    comparison = zirium.diff(before, after)
    with pytest.raises(ValueError, match="positive integer"):
        comparison.query(changes(), **{keyword: value})
