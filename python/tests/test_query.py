from typing import assert_type

import pytest
import zirium
from zirium.query import QueryExpr, always, has_attr, input, op, ops

SOURCE = """
"test.scope"() ({
  %c = "test.seed"() {tag = "start"} : () -> i32
  %a = "test.add"(%c, %c) : (i32, i32) -> i32
  "test.end"(%a) : (i32) -> ()
}) {sym_name = "first"} : () -> ()
"test.scope"() ({
  "test.idle"() : () -> ()
}) {sym_name = "second"} : () -> ()
"""


def document() -> zirium.Document:
    result = zirium.parse_text(SOURCE).lower_strict()
    assert result.document is not None
    return result.document


def test_navigation_composition_and_native_types():
    doc = document()
    seed = ops().filter(op("test.seed"))
    assert isinstance(seed, QueryExpr)
    assert doc.query(seed.users().count()) == 2
    assert doc.query(seed.users().unique().names()) == ["test.add"]
    assert doc.query(seed.users().union(seed).names()) == ["test.seed", "test.add"]
    assert doc.query(seed.users().difference(seed.users()).count()) == 0
    assert doc.query(seed.users().intersect(ops().filter(op("test.add"))).count()) == 1
    assert doc.query(seed.fixpoint(input().union(input().users())).names()) == [
        "test.seed",
        "test.add",
        "test.end",
    ]
    selection = doc.query(seed)
    assert_type(selection, list[zirium.SemanticOperation])
    assert selection[0].name == "test.seed"
    types = doc.query(seed.result_types())
    assert_type(types, list[zirium.SemanticType])
    assert types[0].kind == "integer"
    assert doc.query(seed.users().operand_types().unique().spellings()) == ["i32"]
    attributes = doc.query(seed.attributes("tag"))
    assert_type(attributes, list[zirium.SemanticAttribute])
    assert attributes[0].name == "tag"
    assert doc.query(seed.attributes("tag").spellings()) == ['"start"']
    assert doc.query(seed.string_attr("tag").one()) == "start"


def test_nested_queries_maps_ordering_and_relative_input():
    doc = document()
    scopes = ops().filter(op("test.scope"))
    key = input().string_attr("sym_name").one()
    report = scopes.map_by(key, input().children().subtree().names().tally())
    result = doc.query(report)
    assert_type(result, dict[str, dict[str, int]])
    assert result == {
        "first": {"test.seed": 1, "test.add": 1, "test.end": 1},
        "second": {"test.idle": 1},
    }
    assert doc.query(scopes.map_by(key, ops().count())) == {"first": 6, "second": 6}
    selections = doc.query(scopes.map_by(key, input().children()))
    assert_type(selections, dict[str, list[zirium.SemanticOperation]])
    assert selections["first"][0].name == "test.seed"
    assert doc.query(
        ops().where_exists(input().users().filter(op("test.add"))).names()
    ) == ["test.seed"]
    assert doc.query(
        scopes.sort_by(input().children().count()).string_attr("sym_name")
    ) == ["second", "first"]
    assert (
        doc.query(
            scopes.max_by(input().children().count()).string_attr("sym_name").one()
        )
        == "first"
    )


def test_expressions_are_reusable_and_selections_are_live_handles():
    doc = document()
    tagged = ops().filter(has_attr("tag"))
    selected = doc.query(tagged)
    saved_values = doc.query(tagged.string_attr("tag"))
    with doc.edit() as edit:
        edit.remove_attribute(selected[0], "tag")
    assert doc.query(tagged.count()) == 0
    assert selected[0].attribute_by_name("tag") is None
    assert saved_values == ["start"]
    assert (
        doc.query(
            ops().filter((op("test.seed") | op("test.add")) & ~has_attr("tag")).count()
        )
        == 2
    )
    assert document().query(tagged.count()) == 1
    end = doc.query(ops().filter(op("test.end")))[0]
    with doc.edit() as edit:
        edit.erase(end)
    with pytest.raises(zirium.StaleHandleError):
        _ = end.name


def test_errors_are_explicit_and_limits_apply_without_a_parser():
    doc = document()
    with pytest.raises(TypeError, match="truth value"):
        bool(op("test.seed"))
    with pytest.raises(TypeError, match="truth value"):
        bool(ops())
    with pytest.raises(ValueError, match="exactly one"):
        doc.query(ops().filter(always(False)).names().one())
    with pytest.raises(ValueError, match="exactly one"):
        doc.query(ops().names().one())
    with pytest.raises(ValueError, match="duplicate key"):
        doc.query(
            ops()
            .filter(op("test.seed"))
            .users()
            .map_by(input().names().one(), input().count())
        )
    with pytest.raises(ValueError, match="work limit"):
        doc.query(ops().fixpoint(input().subtree()), max_work=20)
    with pytest.raises(ValueError, match="stream size limit"):
        doc.query(ops(), max_items=1)
    nested = input()
    for _ in range(1000):
        nested = input().where_exists(nested)
    with pytest.raises(ValueError, match="nesting limit"):
        doc.query(nested)


def test_reachable_treats_unknown_operations_as_leaves_unless_strict():
    result = zirium.parse_text(
        '%seed = "test.seed"() : () -> i32\n%value = "vendor.unknown"(%seed) : (i32) -> i32'
    ).lower_strict()
    assert result.document is not None
    query = ops().filter(op("vendor.unknown")).reachable().names()
    assert result.document.query(query) == ["vendor.unknown"]
    with pytest.raises(ValueError, match="vendor.unknown"):
        result.document.query(query, strict=True)


def test_projection_errors_and_nested_native_results():
    lowered = zirium.parse_text(
        '%v = "test.op"() {key = "x", number = 1 : i32, bytes = "\\FF"} : () -> i32'
    ).lower_strict()
    assert lowered.document is not None
    doc = lowered.document
    with pytest.raises(ValueError, match="not a string"):
        doc.query(ops().string_attr("number"))
    with pytest.raises(ValueError, match="UTF-8"):
        doc.query(ops().string_attr("bytes"))
    assert doc.query(ops().attributes("bytes").spellings()) == ['"\\FF"']
    key = input().string_attr("key").one()
    result = doc.query(ops().map_by(key, input().map_by(key, input().result_types())))
    assert_type(result, dict[str, dict[str, list[zirium.SemanticType]]])
    assert result["x"]["x"][0].spelling == "i32"
    assert doc.query(ops().filter(always(False)).map_by(key, input().count())) == {}
