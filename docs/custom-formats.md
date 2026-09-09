# Custom formats across Rust, Python, and the CLI

Zirium separates syntax recovery from semantic support. Recovering an unknown
custom operation lets a tool inspect its name, source text, and nested regions.
It does not establish that the operation can be verified or rewritten.

## Registry choices

| Registry | Custom syntax |
| --- | --- |
| Empty, including the default Python parser | Unknown operations are recovered; generic quoted operations are parsed normally. |
| Core | `builtin.module`, `func.func`, `func.call`, and `func.return`. |
| Proving | Core plus `arith.constant`, `arith.addi`, `cf.br`, and `cf.cond_br`. |
| StableHLO preset | Core plus the binary elementwise and return custom forms listed below. |
| Declarative | A selected subset of the proving catalog. |
| Operation shapes | Caller-named operations using one of the supported structural grammars. |

Core, proving, and declarative registries containing `builtin.module` accept
`module` as its shorthand, including named and nested modules. The empty
registry does not enable this grammar.

The declarative registry selects existing implementations. It does not
interpret arbitrary MLIR assembly-format strings or load ODS/TableGen files.
Rust callers can also construct static descriptors with parser, lowering,
verification, and printing callbacks. Python does not expose those callbacks.

The bundled `stablehlo` preset is an initial syntax subset, not a complete
StableHLO implementation. It parses and lowers the standard binary forms for
`add`, `and`, `atan2`, `divide`, `maximum`, `minimum`, `multiply`, `or`,
`power`, `remainder`, the three shifts, `subtract`, and `xor`, along with typed
`stablehlo.return`. Other StableHLO operations remain available in generic
quoted form or through best-effort custom-syntax recovery. The embedded
[registry file](../crates/zirium/registries/stablehlo.json) is the exact preset
definition. This subset was checked against StableHLO 1.20.1. The unversioned
preset name tracks Zirium releases; it does not claim support for the complete
StableHLO 1.20.1 opset or its portable artifact format.

## Python

Pass the registry when parsing. The parsed file retains it, and its semantic
documents use it for verification, editing, and custom printing.

Load a bundled preset by name:

```python
registry = zirium.DialectRegistry.from_name("stablehlo")
```

```python
import zirium

registry = zirium.DialectRegistry.proving().extend_operation_shapes({
    "vendor.function": zirium.OperationShape.FUNC_LIKE,
    "vendor.invoke": zirium.OperationShape.CALL_LIKE,
})
source = '''module {
  vendor.function @declaration()
}'''
parsed = zirium.parse_text(source, registry=registry)
assert parsed.diagnostics == []
lowered = parsed.lower_strict("hybrid")
assert lowered.document is not None
operation = lowered.document.operation_table("vendor.function").operation(0)
assert operation.symbol_name == "declaration"
```

`with_operation_shapes(...)` starts with core operations.
`existing_registry.extend_operation_shapes(...)` preserves the existing
registry. Both accept Python mappings and return a new registry.

`FUNC_LIKE` and `CALL_LIKE` provide the existing symbol-oriented forms.
`BINARY_OPERANDS` accepts two SSA operands followed by either one shared type or
a function type. `OPTIONAL_TYPED_OPERANDS` accepts a variadic operand list with
a matching optional type list.

Shapes supply parsing and lowering conventions. They do not define a vendor
operation's verifier, symbol-table rules, or custom printer. In particular,
`custom_bytes()` prints caller-defined shapes in generic form. Registering a
func-like shape does not make it equivalent to `func.func` for every semantic
analysis.

`lower_strict()` rejects lowering errors. It does not replace
`document.verify_semantics()`. Best-effort lowering can return an incomplete
document; inspect its diagnostics and `semantically_complete` before choosing
an output or editing path.

`custom_bytes()` prefers the built-in assembly printers. It falls back to
generic form for unsupported structure, properties, locations, typed constant
attributes, and string-valued function or module names that the current custom
printer cannot represent faithfully. Use `canonical_bytes()` for deterministic
generic output and `preserving_bytes()` for eligible source-preserving edits.
Validation failures from `write_custom()` and `write_canonical()` raise
`ValueError`; file creation, write, and flush failures raise `OSError`.

## Rust

Use the same registry for parsing, lowering, verification, editing, and
`print_with_registry`. `ParsedFile` does not retain an owned registry. Text
edits on registered syntax need `apply_text_edits_with_registry`; the shorter
`apply_text_edits` method reparses with the empty registry.

## JSON configuration and Pydantic models

The CLI and Python accept the same complete registry configuration:

```json
{
  "presets": [],
  "builtins": ["builtin.module", "arith.constant", "arith.addi"],
  "operation_shapes": [
    {"name": "vendor.function", "shape": "func_like"},
    {"name": "vendor.invoke", "shape": "call_like"}
  ],
  "operation_formats": [
    {"name": "vendor.convert", "format": "$operands attr-dict `:` type($operands) `to` type($results)"}
  ]
}
```

`builtins` and `operation_shapes` are required and may be empty. `presets` and
`operation_formats` default to empty. `presets` adds bundled registries by
name. `builtins` selects operations from the declarative catalog.
`operation_shapes` assigns exact names to a supported grammar: `func_like`,
`call_like`, `binary_operands`, or `optional_typed_operands`.
`unary_operand`, `variadic_operands`, and `literal_attribute` cover the smaller
expression forms.

`operation_formats` describes the order of operands or a literal value, an
optional attribute dictionary, fixed `:` or `to` tokens, and type captures.
The registry checks each description when it is constructed. The typed-literal
form is ``$value `:` type($value) attr-dict `:` type($result)``.

This configuration replaces the caller's default registry.

For the bundled StableHLO subset:

```json
{
  "presets": ["stablehlo"],
  "builtins": [],
  "operation_shapes": []
}
```

Python accepts an ordinary JSON-compatible dictionary:

```python
config = {
    "presets": [],
    "builtins": ["builtin.module"],
    "operation_shapes": [
        {"name": "vendor.function", "shape": "func_like"},
    ],
}
registry = zirium.DialectRegistry.from_config(config)
parsed = zirium.parse_text("module { vendor.function @f() }", registry=registry)
```

Or use the Pydantic definitions:

```python
from pathlib import Path

config = zirium.RegistryConfig(
    builtins=["builtin.module"],
    operation_shapes=[
        zirium.OperationShapeConfig(name="vendor.function", shape="func_like"),
    ],
)
registry = zirium.DialectRegistry.from_config(config)
Path("registry.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")
registry = zirium.DialectRegistry.from_file("registry.json")
schema = zirium.RegistryConfig.model_json_schema()
```

The Pydantic models check the structure without coercing input types. The shared
Rust builder checks registered names, duplicates, and conflicts for both files
and Python data. Unknown fields, invalid shapes, and invalid registrations raise
`ValueError`; file I/O failures raise `OSError`. Non-JSON-compatible objects in
a dictionary raise `TypeError`. Loading does not retain the input dictionary.

Combine several configurations by passing additional arguments:

```python
registry = zirium.DialectRegistry.from_file("common.json", "vendor.json")
registry = zirium.DialectRegistry.from_config(common_config, vendor_config)
```

Built-ins and identical shape definitions shared across configurations are
included once. Conflicting shapes fail, as does a custom shape that collides
with a built-in selected by another configuration. Duplicates within one
configuration are errors. No file overrides another based on order.

## The binary

Build or install the binary separately with Cargo:

```sh
cargo install --path crates/zirium
zirium --registry examples/cli/registry.json \
  'filter(op("vendor.function")) | count' examples/cli/registered-shapes.mlir
zirium --registry common.json --registry vendor.json -f inspect.zirium input.mlir
```

The Python wheel provides the extension and Python API; it does not install
this binary. With no `--registry` flags the binary uses the proving registry.
With one or more flags it uses their combined configuration. Registry files
are read as UTF-8 JSON through Serde before any MLIR input is read. Relative
paths resolve against the working directory, and stdin remains reserved for
MLIR. Registry failures produce no query output.

Place `--registry` before an inline query. With `-f`/`--program-file`, registry
options may appear before or after the program-file pair, until the first
input path. Remaining arguments are input paths. `--` ends option parsing.

The CLI can select or count recovered unknown custom operations. It rejects
other syntax errors and semantic lowering diagnostics. Semantic mutations
require a complete document. Output uses the selected-fragment printer, including
when the selection contains the whole input. `closure` additionally
requires registered reference semantics; configuring a func-like or call-like
shape alone does not supply vendor dependency semantics. See
the [query language reference](query-language.md) for the query syntax and
selected-fragment output contract, or the [CLI examples](cli-examples.md) for
worked commands.
