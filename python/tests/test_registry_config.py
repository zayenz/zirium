import json
from pathlib import Path

import pytest
import zirium
from pydantic import ValidationError

EXAMPLES = Path(__file__).parents[2] / "examples" / "cli"


def test_file_dict_and_pydantic_registry_agree():
    config = json.loads((EXAMPLES / "registry.json").read_text())
    model = zirium.RegistryConfig.model_validate(config)
    assert model.model_dump(mode="json") == config
    assert json.loads(model.model_dump_json()) == config
    assert model.model_json_schema()["additionalProperties"] is False
    registries = [
        zirium.DialectRegistry.from_file(EXAMPLES / "registry.json"),
        zirium.DialectRegistry.from_config(config),
        zirium.DialectRegistry.from_config(model),
    ]
    config["operation_shapes"].clear()
    model.operation_shapes.clear()
    documents = []
    for registry in registries:
        parsed = zirium.parse_file(
            EXAMPLES / "registered-shapes.mlir", registry=registry
        )
        assert parsed.diagnostics == []
        document = parsed.lower_strict().document
        assert document is not None
        function = document.operation_table("vendor.function").operation(0)
        call = document.operation_table("vendor.invoke").operation(0)
        assert function.symbol_name == "identity"
        assert function.signature == "(i32) -> i32"
        assert call.callee == "identity"
        block = function.region(0).block(0)
        assert block.operation(0).operand(0).key == block.argument(0).key
        definition = call.operand(0).defining_operation
        assert definition is not None
        assert definition.name == "arith.constant"
        documents.append(document)
    assert all(documents[0].structurally_equal(doc) for doc in documents[1:])


@pytest.mark.parametrize(
    "config",
    [
        {},
        [[], []],
        {"builtins": [], "operation_shapes": [["a.b", "func_like"]]},
        {"builtins": [], "operation_shapes": [], "typo": True},
        {"builtins": [], "operation_shapes": [{"name": "a.b", "shape": "other"}]},
        {"builtins": None, "operation_shapes": []},
    ],
)
def test_schema_errors_in_dict_model_and_file(config, tmp_path: Path):
    with pytest.raises(ValidationError):
        zirium.RegistryConfig.model_validate(config)
    with pytest.raises(ValueError):
        zirium.DialectRegistry.from_config(config)
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        zirium.DialectRegistry.from_file(path)


def test_registration_errors_and_file_errors(tmp_path: Path):
    model = zirium.RegistryConfig(
        builtins=["builtin.module"],
        operation_shapes=[
            zirium.OperationShapeConfig(name="module", shape="func_like")
        ],
    )
    with pytest.raises(ValueError, match="conflicts"):
        zirium.DialectRegistry.from_config(model)
    with pytest.raises(OSError):
        zirium.DialectRegistry.from_file(tmp_path / "missing.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{")
    with pytest.raises(ValueError, match="line 1"):
        zirium.DialectRegistry.from_file(broken)
    with pytest.raises(TypeError):
        zirium.DialectRegistry.from_config(
            {"builtins": [], "operation_shapes": {object()}}
        )
    empty = zirium.DialectRegistry.from_config({"builtins": [], "operation_shapes": []})
    assert zirium.parse_text("module {}", registry=empty).diagnostics


def test_multiple_configs_share_identical_entries_and_reject_conflicts(tmp_path: Path):
    first = {
        "builtins": ["builtin.module"],
        "operation_shapes": [{"name": "vendor.function", "shape": "func_like"}],
    }
    second = {
        "builtins": ["builtin.module", "arith.constant"],
        "operation_shapes": [
            {"name": "vendor.function", "shape": "func_like"},
            {"name": "vendor.invoke", "shape": "call_like"},
        ],
    }
    paths = [tmp_path / "first.json", tmp_path / "second.json"]
    for path, config in zip(paths, [first, second]):
        path.write_text(json.dumps(config))
    for registry in [
        zirium.DialectRegistry.from_config(
            first, zirium.RegistryConfig.model_validate(second)
        ),
        zirium.DialectRegistry.from_file(*paths),
        zirium.DialectRegistry.from_config(second, first),
    ]:
        parsed = zirium.parse_file(
            EXAMPLES / "registered-shapes.mlir", registry=registry
        )
        assert parsed.diagnostics == []
        assert parsed.lower_strict().document is not None
    conflicting = {
        "builtins": [],
        "operation_shapes": [{"name": "vendor.function", "shape": "call_like"}],
    }
    with pytest.raises(ValueError, match="conflicting operation shapes"):
        zirium.DialectRegistry.from_config(first, conflicting)
    duplicate = {"builtins": [], "operation_shapes": first["operation_shapes"] * 2}
    with pytest.raises(ValueError, match="duplicate.*operation"):
        zirium.DialectRegistry.from_config(first, duplicate)
    shadow = {
        "builtins": [],
        "operation_shapes": [{"name": "arith.constant", "shape": "call_like"}],
    }
    with pytest.raises(ValueError, match="conflicts with registered"):
        zirium.DialectRegistry.from_config(shadow, second)
