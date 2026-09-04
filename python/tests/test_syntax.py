from pathlib import Path

import pytest
import zirium

VALID = b'"builtin.module"() : () -> ()\n'


def test_keyword_resource_limits_are_lossless_except_file_size(tmp_path: Path):
    with pytest.raises(zirium.ResourceLimitError):
        zirium.parse_bytes(VALID, max_file_bytes=len(VALID) - 1)

    parsed = zirium.parse_text(VALID.decode(), max_tokens=1)
    assert parsed.original_bytes() == VALID
    assert any(d.kind == "lexer.TokenLimit" for d in parsed.diagnostics)

    path = tmp_path / "limited.mlir"
    path.write_bytes(VALID)
    with pytest.raises(zirium.ResourceLimitError):
        zirium.parse_file(path, max_file_bytes=1)


def test_nested_attribute_limit_recovers_following_operation():
    nested = b"1"
    for depth in range(24):
        nested = b"[" + nested + b"]" if depth % 2 == 0 else b"{k = " + nested + b"}"
    source = b'"deep"() {a = ' + nested + b"} : () -> ()\n" + VALID
    parsed = zirium.parse_bytes(source, max_delimiter_depth=4)
    assert parsed.original_bytes() == source
    assert any(d.kind == "parser.DepthLimit" for d in parsed.diagnostics)
    assert parsed.operation_count == 2


def test_pretty_custom_operations_survive_as_consecutive_syntax_nodes():
    source = b"stablehlo.add %lhs, %rhs\nstablehlo.return %lhs\n"
    parsed = zirium.parse_bytes(source)
    assert parsed.original_bytes() == source
    assert parsed.operation_count == 2


def test_handles_keep_the_parse_alive():
    parsed = zirium.parse_bytes(VALID)
    root = parsed.root
    children = list(memoryview(root.child_indices()).cast("I"))
    del root
    assert parsed.node(children[0]).text().startswith(b'"builtin.module"')


def test_registry_selects_custom_syntax_and_is_owned_by_file():
    source = b"%value = arith.constant 7 : i32"
    proving = zirium.DialectRegistry.proving()
    parsed = zirium.parse_bytes(source, registry=proving)
    del proving
    assert parsed.diagnostics == []
    assert parsed.lower_strict().document is not None

    generic = zirium.parse_bytes(source, registry=zirium.DialectRegistry.empty())
    assert generic.diagnostics


def test_core_registry_accepts_ordinary_and_nested_modules_only_when_selected():
    source = b"module { module { } }"
    parsed = zirium.parse_bytes(source, registry=zirium.DialectRegistry.core())
    assert parsed.diagnostics == []
    assert parsed.operation_count == 2

    assert zirium.parse_bytes(source).diagnostics
    assert zirium.parse_bytes(
        source, registry=zirium.DialectRegistry.proving()
    ).diagnostics


def test_declarative_registry_rejects_unknown_and_duplicate_operations():
    with pytest.raises(ValueError, match="unknown declarative operation"):
        zirium.DialectRegistry.declarative(
            ["vendor.unknown"]  # ty: ignore[invalid-argument-type]
        )
    with pytest.raises(ValueError, match="duplicate declarative operation"):
        zirium.DialectRegistry.declarative(["arith.constant", "arith.constant"])


def test_operation_shape_registry_validates_owned_per_mnemonic_mappings():
    registry = zirium.DialectRegistry.with_operation_shapes(
        {"vendor.function": zirium.OperationShape.FUNC_LIKE}
    )
    assert (
        zirium.parse_text("vendor.function @decl()", registry=registry).diagnostics
        == []
    )
    with pytest.raises(ValueError, match="must not be empty"):
        zirium.DialectRegistry.with_operation_shapes(
            {"": zirium.OperationShape.FUNC_LIKE}
        )
    with pytest.raises(ValueError, match="conflicts with core operation"):
        zirium.DialectRegistry.with_operation_shapes(
            {"func.func": zirium.OperationShape.FUNC_LIKE}
        )


def test_invalid_utf8_bytes_and_exact_original_output():
    source = VALID + b"\xff\xfe"
    parsed = zirium.parse_bytes(source)
    assert parsed.original_bytes() == source
    assert any(d.kind == "lexer.InvalidByte" for d in parsed.diagnostics)


def test_text_is_encoded_as_utf8():
    source = '"op"() {name = "räksmörgås"} : () -> ()'
    assert zirium.parse_text(source).original_bytes() == source.encode("utf-8")


def test_malformed_input_has_ranged_diagnostics():
    parsed = zirium.parse_bytes(b'"broken"(')
    assert parsed.diagnostics
    assert all(start <= end for start, end in (d.range for d in parsed.diagnostics))


def _columns(parsed):
    table = parsed.syntax_table()
    return table, {
        "node_kind": list(memoryview(table.node_kind).cast("H")),
        "node_start": list(memoryview(table.node_start).cast("I")),
        "node_end": list(memoryview(table.node_end).cast("I")),
        "node_subtree_end": list(memoryview(table.node_subtree_end).cast("I")),
        "node_flags": list(table.node_flags),
        "token_kind": list(memoryview(table.token_kind).cast("H")),
        "token_start": list(memoryview(table.token_start).cast("I")),
        "token_end": list(memoryview(table.token_end).cast("I")),
    }


@pytest.mark.parametrize(
    "source",
    [
        VALID + b'"other"() : () -> ()\n',
        b'"broken"(',
        b"",
        bytes(range(256)),
    ],
)
def test_packed_table_matches_indexed_syntax(source):
    parsed = zirium.parse_bytes(source)
    table, columns = _columns(parsed)
    assert all(
        isinstance(value, bytes)
        for value in (
            table.node_kind,
            table.node_start,
            table.node_end,
            table.node_subtree_end,
            table.node_flags,
            table.token_kind,
            table.token_start,
            table.token_end,
        )
    )
    assert all(
        len(columns[name]) == parsed.node_count
        for name in (
            "node_kind",
            "node_start",
            "node_end",
            "node_subtree_end",
            "node_flags",
        )
    )
    assert all(
        len(columns[name]) == parsed.token_count
        for name in (
            "token_kind",
            "token_start",
            "token_end",
        )
    )
    for index in range(parsed.node_count):
        node = parsed.node(index)
        assert table.node_kind_name(columns["node_kind"][index]) == node.kind
        expected_range = node.range or (2**32 - 1, 2**32 - 1)
        assert (
            columns["node_start"][index],
            columns["node_end"][index],
        ) == expected_range
        assert columns["node_subtree_end"][index] == node.descendant_range()[1]
        assert columns["node_flags"][index] == int(node.has_error)
        assert columns["node_flags"][index] & ~1 == 0
    for index in range(parsed.token_count):
        token = parsed.token(index)
        assert table.token_kind_name(columns["token_kind"][index]) == token.kind
        assert (
            columns["token_start"][index],
            columns["token_end"][index],
        ) == token.range


def test_indexed_and_packed_traversal_is_ordered_and_linear():
    parsed = zirium.parse_bytes(VALID + b'"other"() : () -> ()\n')
    assert parsed.operation_count == 2
    child_indices = list(memoryview(parsed.root.child_indices()).cast("I"))
    operation = parsed.node(child_indices[0]).as_operation()
    assert operation is not None
    assert operation.text().startswith(b'"builtin.module"')
    assert parsed.root.descendant_range() == (0, parsed.node_count)
    assert all(
        index < end <= parsed.node_count
        for index, end in enumerate(
            memoryview(parsed.syntax_table().node_subtree_end).cast("I")
        )
    )
    start, end = parsed.root.token_range()
    assert (start, end) == (0, parsed.token_count)
    assert (
        b"".join(
            parsed.token(index).text()
            for index in range(start, end)
            if parsed.token(index).kind != "Eof"
        )
        == parsed.original_bytes()
    )
    table = parsed.operation_table()
    assert list(memoryview(table.operation_node).cast("I")) == child_indices
    for offsets in (
        table.result_offsets,
        table.operand_offsets,
        table.successor_offsets,
        table.region_offsets,
    ):
        assert list(memoryview(offsets).cast("I")) == [0, 0, 0]


def test_syntax_node_downcasts_generic_registered_and_unrelated_nodes():
    generic = zirium.parse_bytes(VALID)
    generic_node = generic.node(
        next(iter(memoryview(generic.root.child_indices()).cast("I")))
    )
    assert generic_node.kind == "Operation"
    assert generic_node.as_operation() is not None

    registered = zirium.parse_text("module { }", registry=zirium.DialectRegistry.core())
    registered_node = registered.node(
        next(iter(memoryview(registered.root.child_indices()).cast("I")))
    )
    assert registered_node.kind == "DialectOperation"
    assert registered_node.as_operation() is not None
    assert registered.root.as_operation() is None


@pytest.mark.parametrize(
    "source",
    [
        b"",
        b'%0 = "make"() : () -> i32\n"use"(%0)[^next] : (i32) -> ()',
        b'"outer"() ({ "inner"() : () -> () }) : () -> ()',
        b'%0 = "broken"(%0)[^next] ({ "inner"(',
    ],
)
def test_operation_relationship_table_has_index_provenance(source):
    parsed = zirium.parse_bytes(source)
    table = parsed.operation_table()
    operation_nodes = list(memoryview(table.operation_node).cast("I"))
    assert len(operation_nodes) == parsed.operation_count
    assert [
        parsed.operation(i).syntax().text() for i in range(parsed.operation_count)
    ] == [parsed.node(index).text() for index in operation_nodes]
    for offsets_bytes, nodes_bytes in (
        (table.result_offsets, table.result_nodes),
        (table.operand_offsets, table.operand_nodes),
        (table.successor_offsets, table.successor_nodes),
        (table.region_offsets, table.region_nodes),
    ):
        offsets = list(memoryview(offsets_bytes).cast("I"))
        nodes = list(memoryview(nodes_bytes).cast("I"))
        assert len(offsets) == parsed.operation_count + 1
        assert offsets[0] == 0 and offsets[-1] == len(nodes)
        assert offsets == sorted(offsets)
        assert all(0 <= index < parsed.node_count for index in nodes)


def test_index_and_kind_conversion_errors_and_snapshot_provenance():
    first = zirium.parse_bytes(VALID)
    second = zirium.parse_bytes(b"")
    table = first.syntax_table()
    for index in (-1, first.node_count):
        with pytest.raises((IndexError, OverflowError)):
            first.node(index)
    for index in (-1, first.token_count):
        with pytest.raises((IndexError, OverflowError)):
            first.token(index)
    for index in (-1, first.operation_count):
        with pytest.raises((IndexError, OverflowError)):
            first.operation(index)
    with pytest.raises(ValueError, match="unknown syntax kind"):
        table.node_kind_code("operation")
    with pytest.raises(ValueError, match="unknown syntax kind code"):
        table.node_kind_name(65535)
    with pytest.raises(ValueError, match="unknown token kind"):
        table.token_kind_code("eof")
    with pytest.raises(ValueError, match="unknown token kind code"):
        table.token_kind_name(65535)
    assert table.node_kind_name(table.node_kind_code("Operation")) == "Operation"
    assert table.token_kind_name(table.token_kind_code("Eof")) == "Eof"
    assert (
        first.node_count != second.node_count or first.token_count != second.token_count
    )


def test_runtime_and_stub_expose_only_indexed_bulk_syntax_surface():
    removed_file = {"nodes", "tokens", "nodes_of_kind", "ranges_of_kind", "operations"}
    removed_node = {"children", "descendants", "tokens"}
    removed_operation = {"results", "operands", "successors", "regions"}
    assert removed_file.isdisjoint(dir(zirium.File))
    assert removed_node.isdisjoint(dir(zirium.SyntaxNode))
    assert removed_operation.isdisjoint(dir(zirium.Operation))
    stub = (Path(zirium.__file__).with_name("__init__.pyi")).read_text()
    assert zirium.SyntaxOperationTable is not None
    assert "SyntaxOperationTable" in vars(zirium)["__all__"]
    assert "class SyntaxOperationTable:" in stub
    for name in (
        "SyntaxTable",
        "node_count",
        "token_count",
        "syntax_table",
        "operation_count",
        "operation_table",
        "operation_node",
        "result_offsets",
        "child_indices",
        "descendant_range",
        "token_range",
    ):
        assert name in stub


def test_raw_file_input_and_output_preserve_invalid_utf8(tmp_path: Path):
    source = VALID + b"\x80"
    input_path = tmp_path / "input.mlir"
    output_path = tmp_path / "output.mlir"
    input_path.write_bytes(source)
    parsed = zirium.parse_file(input_path)
    input_path.unlink()
    assert parsed.original_bytes() == source
    parsed.write_original(output_path)
    assert output_path.read_bytes() == source
