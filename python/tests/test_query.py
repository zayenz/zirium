from typing import TYPE_CHECKING, assert_type

import pytest
import zirium
from zirium.query import QueryExpr, always, has_attr, input, op, ops

if TYPE_CHECKING:
    from zirium.query import ReportInput

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
    assert doc.query(seed.forward_slice().names()) == [
        "test.seed",
        "test.add",
        "test.end",
    ]
    end = ops().filter(op("test.end"))
    assert doc.query(end.backward_slice().names()) == [
        "test.seed",
        "test.add",
        "test.end",
    ]
    assert doc.query(end.slice().names()) == doc.query(end.backward_slice().names())
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


def test_grouped_report_builder_returns_first_seen_function_groups():
    from pathlib import Path

    from zirium.query import ReportExpr as E
    from zirium.query import bound_ops, report

    source = (
        Path(__file__).parents[2] / "examples/cli/function-report.mlir"
    ).read_text()
    lowered = zirium.parse_text(
        source, registry=zirium.DialectRegistry.baseline()
    ).lower_strict()
    assert lowered.document is not None
    f = bound_ops("f")
    rows = (
        ops()
        .filter(op("func.func"))
        .project(
            "f",
            {
                "name": f.attr("sym_name").one(),
                "matches": f.children().subtree().filter(op("arith.addi")).count(),
            },
        )
    )
    assert lowered.document.query(report(E(rows).group_by("matches"))) == [
        {
            "key": 1,
            "items": [{"name": "double", "matches": 1}, {"name": "mix", "matches": 1}],
        },
        {"key": 0, "items": [{"name": "plain", "matches": 0}]},
    ]


def test_correlated_report_builders_return_ordered_python_data():
    from pathlib import Path

    from zirium.query import (
        ReportExpr as E,
    )
    from zirium.query import (
        ReportGenerator as G,
    )
    from zirium.query import (
        bound_ops,
        op_expr,
        report,
        require,
    )

    source = (
        Path(__file__).parents[2] / "examples/cli/function-report.mlir"
    ).read_text()
    lowered = zirium.parse_text(
        source, registry=zirium.DialectRegistry.baseline()
    ).lower_strict()
    assert lowered.document is not None
    doc = lowered.document
    f = bound_ops("f")
    rows = (
        ops()
        .filter(op("func.func"))
        .project(
            "f",
            {
                "name": f.attr("sym_name").one(),
                "operations": f.children().subtree().count(),
                "matches": f.children()
                .subtree()
                .filter(op_expr(E.name("operation_name")))
                .count(),
                "kinds": f.children().subtree().names().unique(),
            },
        )
    )
    selected = E.comprehension(
        E.name("r"),
        [
            G.in_values(
                "r",
                E.name("rows"),
                E.name("r").field("operations") >= E.name("threshold"),
            )
        ],
    )
    total_ops = E.comprehension(
        E.name("r").field("operations"), [G.in_values("r", E.name("selected"))]
    ).sum()
    total_matches = E.comprehension(
        E.name("r").field("matches"), [G.in_values("r", E.name("selected"))]
    ).sum()
    bindings: dict[str, ReportInput] = {
        "operation_name": "arith.addi",
        "minimum_ops": 2,
        "threshold": require(
            E.name("minimum_ops") >= 0,
            E.name("minimum_ops"),
            "minimum_ops must be nonnegative",
        ),
        "rows": rows,
        "selected": selected,
    }
    result_expression = {
        "functions": E.name("selected"),
        "total_operations": total_ops,
        "total_matches": total_matches,
    }
    query = report(E.scope(bindings, result_expression))
    assert isinstance(query, QueryExpr)
    result = doc.query(query)
    assert result == {
        "functions": [
            {
                "name": "double",
                "operations": 2,
                "matches": 1,
                "kinds": ["arith.addi", "func.return"],
            },
            {
                "name": "mix",
                "operations": 3,
                "matches": 1,
                "kinds": ["arith.muli", "arith.addi", "func.return"],
            },
        ],
        "total_operations": 5,
        "total_matches": 2,
    }
    assert isinstance(result, dict)
    assert list(result) == ["functions", "total_operations", "total_matches"]
    parameter_bindings = dict(bindings)
    del parameter_bindings["operation_name"]
    del parameter_bindings["minimum_ops"]
    parameter_query = report(
        E.scope(parameter_bindings, result_expression)
    ).with_parameters({"operation_name": "string", "minimum_ops": "int"})
    assert (
        doc.query(
            parameter_query,
            parameters={"operation_name": "arith.addi", "minimum_ops": 2},
        )
        == result
    )
    bad_arguments: tuple[dict[str, bool | int | str], ...] = (
        {"operation_name": "arith.addi"},
        {"operation_name": "arith.addi", "minimum_ops": True},
        {"operation_name": "arith.addi", "minimum_ops": 2, "extra": 0},
    )
    for arguments in bad_arguments:
        with pytest.raises(ValueError):
            doc.query(parameter_query, parameters=arguments)
    for bad, error_type in (
        (2**100, OverflowError),
        (1.5, TypeError),
        ([2], TypeError),
    ):
        with pytest.raises(error_type):
            doc.query(
                parameter_query,
                parameters={"operation_name": "arith.addi", "minimum_ops": bad},  # ty: ignore[invalid-argument-type]
            )
    flags = report(E.name("flag")).with_parameters({"flag": "bool"})
    assert doc.query(flags, parameters={"flag": True}) is True
    exact = report(E.name("text")).with_parameters({"text": "string"})
    assert doc.query(exact, parameters={"text": '  "x"=\\  '}) == '  "x"=\\  '
    with pytest.raises(Exception, match="limit exceeded"):
        doc.query(exact, parameters={"text": "x" * 10000}, max_work=100)

    assert doc.query(report(require(True, [False, 1, 2.5, "x"], "stop"))) == [
        False,
        1,
        2.5,
        "x",
    ]
    with pytest.raises(Exception, match="stop"):
        doc.query(
            report(
                require(
                    False,
                    ops().filter(op("func.func")).string_attr("absent").one(),
                    "stop",
                )
            )
        )
    with pytest.raises(Exception, match="work limit"):
        doc.query(query, max_work=1)


def test_native_ssa_capture_builders_project_the_original_function_report():
    from pathlib import Path

    from zirium.query import (
        CaptureQuery,
        bound_ops,
        bound_ssa,
        capture_ops,
        matches,
        op_expr,
        report,
        same_value,
    )
    from zirium.query import (
        ReportExpr as E,
    )

    source = (
        Path(__file__).parents[2] / "examples/cli/function-report.mlir"
    ).read_text()
    lowered = zirium.parse_text(
        source, registry=zirium.DialectRegistry.baseline()
    ).lower_strict()
    assert lowered.document is not None
    doc = lowered.document
    lhs, rhs = bound_ssa("lhs"), bound_ssa("rhs")
    captures = (
        matches()
        .from_("add", ops().filter(op("arith.addi")))
        .bind("lhs", bound_ops("add").operand(0))
        .bind("rhs", bound_ops("add").operand(1))
        .where_(same_value(lhs, rhs) & ~~same_value(lhs, rhs))
        .build()
    )
    assert isinstance(captures, CaptureQuery)
    assert isinstance(captures, QueryExpr)
    result = doc.query(captures)
    assert_type(
        result,
        list[
            dict[
                str,
                zirium.SemanticOperation
                | zirium.SemanticValue
                | list[zirium.SemanticOperation],
            ]
        ],
    )
    assert len(result) == 1
    assert list(result[0]) == ["add", "lhs", "rhs"]
    add, left, right = result[0]["add"], result[0]["lhs"], result[0]["rhs"]
    assert isinstance(add, zirium.SemanticOperation)
    assert isinstance(left, zirium.SemanticValue)
    assert isinstance(right, zirium.SemanticValue)
    assert add.name == "arith.addi"
    assert left == right == add.operand(0) == add.operand(1)
    assert left.kind == "block_argument"
    assert left.valid
    assert left.type_value is not None
    assert left.type_value.spelling == "i32"
    subject = capture_ops("m", "add")
    rows = captures.project(
        "m",
        {
            "function": subject.root(op("func.func")).attr("sym_name").one(),
            "operation": subject.names().one(),
        },
    )
    full = report(
        E.scope(
            {"rows": rows},
            {"matches": E.name("rows"), "count": E.name("rows").length()},
        )
    )
    assert doc.query(full) == {
        "matches": [{"function": "double", "operation": "arith.addi"}],
        "count": 1,
    }
    parameterized = (
        matches()
        .from_("add", ops().filter(op_expr(E.name("selected"))))
        .bind("lhs", bound_ops("add").operand(0))
        .bind("rhs", bound_ops("add").operand(1))
        .where_(same_value(lhs, rhs))
        .build()
        .with_parameters({"selected": "string"})
    )
    assert len(doc.query(parameterized, parameters={"selected": "arith.addi"})) == 1
    assert doc.query(
        parameterized.project("m", {"name": capture_ops("m", "add").names().one()}),
        parameters={"selected": "arith.addi"},
    ) == [{"name": "arith.addi"}]
    with pytest.raises(TypeError, match="truth value"):
        bool(same_value(lhs, rhs))
    with pytest.raises(ValueError, match="unavailable"):
        doc.query(ops().filter(op("func.return")).head(1).operand(1))
    # The returned handles retain their owner after the Document wrapper is gone.
    del doc, lowered
    assert add.name == "arith.addi"
    assert left.valid


def test_native_ssa_same_owner_counterexample_and_empty_field_preflight():
    from zirium.query import (
        bound_ops,
        bound_ssa,
        capture_ops,
        matches,
        same_value,
    )

    source = """func.func @different() -> i32 {
      %left, %right = "test.pair"() : () -> (i32, i32)
      %sum = arith.addi %left, %right : i32
      func.return %sum : i32
    }"""
    lowered = zirium.parse_text(
        source, registry=zirium.DialectRegistry.baseline()
    ).lower_strict()
    assert lowered.document is not None
    doc = lowered.document
    adds = ops().filter(op("arith.addi"))
    left, right = doc.query(adds.operand(0)), doc.query(adds.operand(1))
    assert_type(left, zirium.SemanticValue)
    assert left.defining_operation == right.defining_operation
    assert left != right
    query = (
        matches()
        .from_("add", adds)
        .bind("lhs", bound_ops("add").operand(0))
        .bind("rhs", bound_ops("add").operand(1))
        .where_(same_value(bound_ssa("lhs"), bound_ssa("rhs")))
        .build()
    )
    assert doc.query(query) == []
    assert (
        doc.query(query.project("m", {"name": capture_ops("m", "add").names().one()}))
        == []
    )
    with pytest.raises(ValueError, match="incompatible native kind"):
        doc.query(query.project("m", {"bad": capture_ops("m", "lhs").names().one()}))


def test_native_users_paths_keep_sequence_steps_owner_and_projection():
    from pathlib import Path

    from zirium.query import ReportExpr as E
    from zirium.query import capture_ops, matches, op_expr

    source = (
        Path(__file__).parents[2] / "examples/cli/function-report.mlir"
    ).read_text()
    lowered = zirium.parse_text(
        source, registry=zirium.DialectRegistry.baseline()
    ).lower_strict()
    assert lowered.document is not None
    doc = lowered.document
    query = (
        ops().filter(op("arith.muli")).paths("users", to=op("func.return"), max_depth=2)
    )
    native = doc.query(query)
    assert len(native) == 1
    steps = native[0]["steps"]
    assert isinstance(steps, list)
    assert [step.name for step in steps] == ["arith.muli", "arith.addi", "func.return"]
    assert native[0]["source"] == steps[0]
    assert native[0]["sink"] == steps[-1]
    assert doc.query(
        query.project("p", {"steps": capture_ops("p", "steps").names()})
    ) == [{"steps": ["arith.muli", "arith.addi", "func.return"]}]
    zero = doc.query(
        ops().filter(op("arith.muli")).paths("users", to=op("arith.muli"), max_depth=0)
    )
    assert isinstance(zero[0]["steps"], list)
    assert zero[0]["steps"] == [zero[0]["source"]]
    ordinary = doc.query(
        matches().from_("steps", ops().filter(op("arith.muli"))).build()
    )
    assert isinstance(ordinary[0]["steps"], zirium.SemanticOperation)
    parameterized = (
        ops()
        .filter(op("arith.muli"))
        .paths("users", to=op_expr(E.name("target")), max_depth=2)
        .with_parameters({"target": "string"})
    )
    assert len(doc.query(parameterized, parameters={"target": "func.return"})) == 1
    with pytest.raises(OverflowError):
        ops().paths("users", to=op("func.return"), max_depth=-1)
    with pytest.raises(OverflowError):
        ops().paths("users", to=op("func.return"), max_depth=2**100)
    del doc, lowered, native
    assert [step.name for step in steps] == ["arith.muli", "arith.addi", "func.return"]
