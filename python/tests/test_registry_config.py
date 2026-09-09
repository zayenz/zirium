import json
from pathlib import Path

import pytest
import zirium
from pydantic import ValidationError

EXAMPLES = Path(__file__).parents[2] / "examples" / "cli"


def test_file_dict_and_pydantic_registry_agree():
    config = json.loads((EXAMPLES / "registry.json").read_text())
    model = zirium.RegistryConfig.model_validate(config)
    config["operation_formats"] = []
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


def test_named_stablehlo_registry_and_config_preset(tmp_path: Path):
    assert zirium.DialectRegistry.preset_names() == (
        "stablehlo",
        "tosa",
        "scf",
        "linalg",
        "acc",
        "affine",
        "amdgpu",
        "amx",
        "arith",
        "arm_neon",
        "arm_sme",
        "arm_sve",
        "async",
        "bufferization",
        "cf",
        "complex",
        "dlti",
        "emitc",
        "func",
        "gpu",
        "index",
        "irdl",
        "llvm",
        "math",
        "memref",
        "ml_program",
        "mpi",
        "shard",
        "nvgpu",
        "nvvm",
        "omp",
        "pdl",
        "pdl_interp",
        "ptr",
        "quant",
        "rocdl",
        "shape",
    )
    config = {"presets": ["stablehlo"], "builtins": [], "operation_shapes": []}
    path = tmp_path / "stablehlo.json"
    path.write_text(json.dumps(config))
    registries = [
        zirium.DialectRegistry.from_name("stablehlo"),
        zirium.DialectRegistry.from_config(config),
        zirium.DialectRegistry.from_file(path),
    ]
    source = """module {
      func.func @add(%lhs: tensor<2xf32>, %rhs: tensor<2xf32>) -> tensor<2xf32> {
        %sum = stablehlo.add %lhs, %rhs : tensor<2xf32>
        func.return %sum : tensor<2xf32>
      }
    }"""
    for registry in registries:
        parsed = zirium.parse_text(source, registry=registry)
        assert parsed.diagnostics == []
        lowered = parsed.lower_strict()
        assert lowered.diagnostics == []
        assert lowered.document is not None
        add = lowered.document.operation_table("stablehlo.add").operation(0)
        assert add.operand_count() == 2
        assert add.result_count() == 1

    with pytest.raises(ValueError, match="unknown registry preset"):
        zirium.DialectRegistry.from_name("unknown")


@pytest.mark.parametrize(
    ("spelling", "classattr"),
    [
        ("unary_operand", zirium.OperationShape.UNARY_OPERAND),
        ("variadic_operands", zirium.OperationShape.VARIADIC_OPERANDS),
        (
            "attr_first_optional_typed_operands",
            zirium.OperationShape.ATTR_FIRST_OPTIONAL_TYPED_OPERANDS,
        ),
        ("literal_attribute", zirium.OperationShape.LITERAL_ATTRIBUTE),
        ("operand_clauses", zirium.OperationShape.OPERAND_CLAUSES),
        ("region_clauses", zirium.OperationShape.REGION_CLAUSES),
    ],
)
def test_extended_operation_shapes_are_configurable(spelling, classattr):
    model = zirium.OperationShapeConfig(name="vendor.op", shape=spelling)
    zirium.DialectRegistry.from_config(
        {"builtins": [], "operation_shapes": [model.model_dump()]}
    )
    zirium.DialectRegistry.with_operation_shapes({"vendor.op": classattr})


def test_operation_formats_round_trip_and_parse_captured_roles():
    formats = [
        zirium.OperationFormatConfig(
            name="a.Op",
            format=("$operands attr-dict `:` type($operands) `to` type($results)"),
        ),
        zirium.OperationFormatConfig(
            name="a.Imm",
            format="$value `:` type($value) attr-dict `:` type($result)",
        ),
    ]
    config = zirium.RegistryConfig(
        builtins=[], operation_shapes=[], operation_formats=formats
    )
    assert zirium.RegistryConfig.model_validate_json(config.model_dump_json()) == config
    registry = zirium.DialectRegistry.from_config(config)
    source = """"builtin.module"() ({
^bb0:
  %a = "test.source"() : () -> i16
  %b = "test.source"() : () -> i16
  %r = a.Op %a, %b : i16 to i16
  %0 = a.Imm 1 : i64 {k = 2 : i64} : i32
}) : () -> ()"""
    parsed = zirium.parse_text(source, registry=registry)
    assert parsed.diagnostics == []
    lowered = parsed.lower_strict()
    assert lowered.diagnostics == []
    assert lowered.document is not None
    operation = lowered.document.operation_table("a.Op").operation(0)
    assert operation.operand_count() == 2
    assert operation.result_type(0).spelling == "i16"
    literal = lowered.document.operation_table("a.Imm").operation(0)
    assert literal.result_type(0).spelling == "i32"
    value = literal.attribute_by_name("value")
    key = literal.attribute_by_name("k")
    assert value is not None and value.spelling == "1 : i64"
    assert key is not None and key.spelling == "2 : i64"


def test_invalid_operation_format_names_its_entry():
    with pytest.raises(ValueError, match="a.Broken"):
        zirium.DialectRegistry.from_config(
            {
                "builtins": [],
                "operation_shapes": [],
                "operation_formats": [
                    {"name": "a.Broken", "format": "$value type($operands)"}
                ],
            }
        )


def test_operation_format_mismatch_uses_whole_operation_recovery():
    registry = zirium.DialectRegistry.from_config(
        {
            "builtins": [],
            "operation_shapes": [],
            "operation_formats": [
                {
                    "name": "a.Op",
                    "format": (
                        "$operands attr-dict `:` type($operands) `to` type($results)"
                    ),
                }
            ],
        }
    )
    parsed = zirium.parse_text(
        '%r = a.Op %arg : i16 -> i16\n"test.after"() : () -> ()\n',
        registry=registry,
    )
    assert [diagnostic.kind for diagnostic in parsed.diagnostics] == [
        "parser.FormatMismatch"
    ]
    document = parsed.lower_best_effort().document
    assert document is not None
    assert document.operation_table("test.after").count == 1


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
