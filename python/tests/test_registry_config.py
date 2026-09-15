import importlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pytest
import zirium
from pydantic import ValidationError

EXAMPLES = Path(__file__).parents[2] / "examples" / "cli"
BUNDLES = Path(__file__).parents[2] / "tests" / "fixtures" / "registry-bundles"
WIDEN_SOURCE = """"builtin.module"() ({
^bb0:
  %lhs = "test.source"() : () -> i16
  %rhs = "test.source"() : () -> i16
  %result = vendor.widen %lhs, %rhs {tag = true} : i16 to i32
}) : () -> ()"""


def assert_bundle_behavior(registry):
    assert registry.call_target_attribute("vendor.invoke") == "target"
    assert registry.operation_alternatives("vendor.choice") == [
        ("format", "$value `:` type($value) attr-dict `:` type($result)"),
        ("shape", "operand_clauses"),
    ]
    parsed = zirium.parse_text(WIDEN_SOURCE, registry=registry)
    assert parsed.diagnostics == []
    lowered = parsed.lower_strict()
    assert lowered.diagnostics == []
    document = lowered.document
    assert document is not None
    operation = document.operation_table("vendor.widen").operation(0)
    operand_types = [operation.operand(index).type_value for index in range(2)]
    assert all(operand_type is not None for operand_type in operand_types)
    assert [
        operand_type.spelling for operand_type in operand_types if operand_type
    ] == [
        "i16",
        "i16",
    ]
    assert operation.result_type(0).spelling == "i32"
    tag = operation.attribute_by_name("tag")
    assert tag is not None and tag.spelling == "true"


def test_file_dict_and_pydantic_registry_agree():
    config = json.loads((EXAMPLES / "registry.json").read_text())
    model = zirium.RegistryConfig.model_validate(config)
    config["operation_formats"] = []
    config["operation_alternatives"] = []
    config["imports"] = []
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
        "sparse_tensor",
        "smt",
        "spirv",
        "tensor",
        "transform",
        "ub",
        "vector",
        "wasmssa",
        "x86vector",
        "xegpu",
        "xevm",
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


def test_registry_contents_are_introspectable():
    registry = zirium.DialectRegistry.from_config(
        {
            "builtins": ["builtin.module"],
            "operation_shapes": [
                {"name": "vendor.shaped", "shape": "literal_attribute"}
            ],
            "operation_formats": [
                {
                    "name": "vendor.formatted",
                    "format": (
                        "$operands attr-dict `:` type($operands) `to` type($results)"
                    ),
                }
            ],
        }
    )

    assert registry.operation_names() == (
        "builtin.module",
        "vendor.formatted",
        "vendor.shaped",
    )
    assert registry.operation_shape("vendor.shaped") == "literal_attribute"
    assert registry.operation_shape("vendor.formatted") is None
    assert registry.operation_shape("vendor.missing") is None


def test_call_target_attribute_is_configurable_and_introspectable():
    call = zirium.OperationShapeConfig(
        name="vendor.invoke", shape="call_like", callee_attribute="target"
    )
    registry = zirium.DialectRegistry.from_config(
        {"builtins": [], "operation_shapes": [call.model_dump()]}
    )

    assert registry.call_target_attribute("vendor.invoke") == "target"
    assert (
        zirium.DialectRegistry.baseline().call_target_attribute("func.call") == "callee"
    )
    assert registry.call_target_attribute("vendor.missing") is None

    formatted = zirium.DialectRegistry.from_config(
        zirium.RegistryConfig(
            builtins=[],
            operation_shapes=[],
            operation_formats=[
                zirium.OperationFormatConfig(
                    name="vendor.formatted_invoke",
                    format="$callee attr-dict",
                    callee_attribute="target",
                )
            ],
        )
    )
    assert formatted.call_target_attribute("vendor.formatted_invoke") == "target"
    parsed = zirium.parse_text("vendor.formatted_invoke @worker", registry=formatted)
    document = parsed.lower_strict().document
    assert document is not None
    operation = document.operation_table("vendor.formatted_invoke").operation(0)
    target = operation.attribute_by_name("target")
    assert target is not None and target.symbol_value == "worker"
    assert operation.callee == "worker"
    assert operation.callee_segments == ["worker"]
    assert operation.callee_spelling == "@worker"
    assert operation.attribute_by_name("callee") is None

    with pytest.raises(ValueError, match="requires a call_like shape"):
        zirium.OperationShapeConfig(
            name="vendor.function",
            shape="func_like",
            callee_attribute="target",
        )

    assert "scf.for" in zirium.DialectRegistry.from_name("scf").operation_names()


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


def test_operation_format_preserves_independently_named_literals():
    registry = zirium.DialectRegistry.from_config(
        zirium.RegistryConfig(
            builtins=[],
            operation_shapes=[],
            operation_formats=[
                zirium.OperationFormatConfig(
                    name="a.Parameter",
                    format=(
                        "$attr(label) `,` $attr(default_value) `:` "
                        "type($attr(default_value)) `:` type($result)"
                    ),
                )
            ],
        )
    )
    parsed = zirium.parse_text(
        '%result = a.Parameter "threshold", 0.0 : f64 : f32', registry=registry
    )
    assert parsed.diagnostics == []
    lowered = parsed.lower_strict()
    assert lowered.diagnostics == []
    assert lowered.document is not None
    operation = lowered.document.operation_table("a.Parameter").operation(0)
    label = operation.attribute_by_name("label")
    default = operation.attribute_by_name("default_value")
    assert label is not None and label.string_value == "threshold"
    assert default is not None and default.spelling == "0.0 : f64"
    assert operation.result_type(0).spelling == "f32"


def test_operation_alternatives_are_validated_inspectable_and_lowered():
    alternatives = zirium.OperationAlternativesConfig(
        name="a.Choice",
        alternatives=[
            zirium.OperationGrammarConfig(
                format="$value `:` type($value) attr-dict `:` type($result)"
            ),
            zirium.OperationGrammarConfig(shape="operand_clauses"),
        ],
    )
    registry = zirium.DialectRegistry.from_config(
        zirium.RegistryConfig(
            builtins=[],
            operation_shapes=[],
            operation_alternatives=[alternatives],
        )
    )
    assert registry.operation_alternatives("a.Choice") == [
        (
            "format",
            "$value `:` type($value) attr-dict `:` type($result)",
        ),
        ("shape", "operand_clauses"),
    ]
    parsed = zirium.parse_text(
        "%literal = a.Choice 0.0 : f64 : f32\n"
        "%dimension = a.Choice dim(#a.dimension<3>) : i32",
        registry=registry,
    )
    assert parsed.diagnostics == []
    document = parsed.lower_strict().document
    assert document is not None
    choices = document.operation_table("a.Choice")
    assert choices.count == 2
    assert choices.operation(0).attribute_by_name("value") is not None
    assert choices.operation(1).attribute_by_name("value") is None

    with pytest.raises(ValidationError, match="exactly one"):
        zirium.OperationGrammarConfig(shape="operand_clauses", format="$value")
    with pytest.raises(ValidationError, match="at least two"):
        zirium.OperationAlternativesConfig(
            name="a.Bad",
            alternatives=[zirium.OperationGrammarConfig(shape="operand_clauses")],
        )

    call_alternatives = zirium.OperationAlternativesConfig(
        name="a.Invoke",
        callee_attribute="target",
        alternatives=[
            zirium.OperationGrammarConfig(format="$callee attr-dict"),
            zirium.OperationGrammarConfig(format="$callee `as` attr-dict"),
        ],
    )
    call_registry = zirium.DialectRegistry.from_config(
        zirium.RegistryConfig(
            builtins=[],
            operation_shapes=[],
            operation_alternatives=[call_alternatives],
        )
    )
    assert call_registry.call_target_attribute("a.Invoke") == "target"


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


def test_filesystem_bundle_matches_direct_composition_and_moves(tmp_path: Path):
    bundled = zirium.DialectRegistry.from_file(BUNDLES / "root.json")
    direct = zirium.DialectRegistry.from_file(
        BUNDLES / "leaves" / "builtins.json",
        BUNDLES / "leaves" / "shapes.json",
        BUNDLES / "leaves" / "formats.json",
    )
    assert bundled.operation_names() == direct.operation_names()
    assert bundled.operation_shape("vendor.function") == "func_like"
    assert_bundle_behavior(bundled)
    assert_bundle_behavior(direct)

    moved = tmp_path / "moved"
    shutil.copytree(BUNDLES, moved)
    assert_bundle_behavior(zirium.DialectRegistry.from_file(moved / "root.json"))


def test_import_models_stay_io_free_and_zip_resources_remain_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model = zirium.RegistryConfig(
        imports=["child.json"], builtins=[], operation_shapes=[]
    )
    assert model.imports == ["child.json"]
    with pytest.raises(ValueError, match="filesystem loader"):
        zirium.DialectRegistry.from_config(model)
    for invalid in ["", str(tmp_path / "absolute.json")]:
        with pytest.raises(ValidationError, match="relative paths"):
            zirium.RegistryConfig(imports=[invalid], builtins=[], operation_shapes=[])

    bundle = {
        "bundle.json": {
            "imports": ["functions.json", "arithmetic.json"],
            "builtins": [],
            "operation_shapes": [],
        },
        "functions.json": {
            "imports": ["common.json"],
            "builtins": [],
            "operation_shapes": [{"name": "vendor.function", "shape": "func_like"}],
        },
        "arithmetic.json": {
            "imports": ["common.json"],
            "builtins": [],
            "operation_shapes": [{"name": "vendor.add", "shape": "binary_operands"}],
        },
        "common.json": {
            "builtins": ["builtin.module"],
            "operation_shapes": [],
        },
    }
    directory = tmp_path / "directory"
    directory.mkdir()
    for name, config in bundle.items():
        (directory / name).write_text(json.dumps(config))
    filesystem = zirium.DialectRegistry.from_file(directory / "bundle.json")

    archive = tmp_path / "registry.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("resource_bundle/__init__.py", "")
        for name, config in bundle.items():
            output.writestr(f"resource_bundle/registries/{name}", json.dumps(config))
    monkeypatch.syspath_prepend(str(archive))
    importlib.invalidate_caches()
    resource = zirium.DialectRegistry.from_package_resources(
        "resource_bundle", "registries/bundle.json"
    )
    assert resource.operation_names() == filesystem.operation_names()
    assert resource.operation_shape("vendor.function") == "func_like"
    assert resource.operation_shape("vendor.add") == "binary_operands"
    sys.modules.pop("resource_bundle", None)
    archive.unlink()
    assert resource.operation_names() == filesystem.operation_names()


def test_registry_graph_errors_and_limits_use_public_exception_classes(tmp_path: Path):
    missing = tmp_path / "missing-root.json"
    missing.write_text(
        json.dumps({"imports": ["child.json"], "builtins": [], "operation_shapes": []})
    )
    with pytest.raises(OSError, match="child.json"):
        zirium.DialectRegistry.from_file(missing)

    cycle_files = {
        "root.json": "parent.json",
        "parent.json": "a.json",
        "a.json": "b.json",
        "b.json": "a.json",
    }
    for name, imported in cycle_files.items():
        (tmp_path / name).write_text(
            json.dumps({"imports": [imported], "builtins": [], "operation_shapes": []})
        )
    with pytest.raises(ValueError, match="cycle") as raised:
        zirium.DialectRegistry.from_file(tmp_path / "root.json")
    diagnostic = str(raised.value)
    positions = [
        diagnostic.find(name)
        for name in ["root.json", "parent.json", "a.json", "b.json"]
    ]
    positions.append(diagnostic.find("a.json", positions[-1]))
    assert positions == sorted(positions) and all(
        position >= 0 for position in positions
    )
    assert "(cycle:" in diagnostic

    with pytest.raises(zirium.ResourceLimitError, match="file count"):
        zirium.DialectRegistry.from_file(BUNDLES / "root.json", max_files=3)
    with pytest.raises(zirium.ResourceLimitError, match="import edges"):
        zirium.DialectRegistry.from_file(BUNDLES / "root.json", max_edges=2)
    with pytest.raises(zirium.ResourceLimitError, match="import depth"):
        zirium.DialectRegistry.from_file(BUNDLES / "root.json", max_depth=0)
    with pytest.raises(zirium.ResourceLimitError, match="registry bytes"):
        zirium.DialectRegistry.from_file(BUNDLES / "root.json", max_bytes=1)
