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
| Declarative | A selected subset of the proving catalog. |
| Operation shapes | Caller-named operations using the fixed func-like or call-like grammar. |

Core, proving, and declarative registries containing `builtin.module` accept
`module` as its shorthand, including named and nested modules. The empty
registry does not enable this grammar.

The declarative registry selects existing implementations. It does not
interpret arbitrary MLIR assembly-format strings or load ODS/TableGen files.
Rust callers can also construct static descriptors with parser, lowering,
verification, and printing callbacks. Python does not expose those callbacks.

## Python

Pass the registry when parsing. The parsed file retains it, and its semantic
documents use it for verification, editing, and custom printing.

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
  "builtins": ["builtin.module", "arith.constant", "arith.addi"],
  "operation_shapes": [
    {"name": "vendor.function", "shape": "func_like"},
    {"name": "vendor.invoke", "shape": "call_like"}
  ]
}
```

Both fields are required and may be empty. `builtins` selects operations from
the declarative catalog. `operation_shapes` assigns exact names to the existing
func-like and call-like grammars. This configuration replaces the caller's
default registry; it does not implicitly add core or proving operations.

Python accepts an ordinary JSON-compatible dictionary:

```python
config = {
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
  'select(op("vendor.function")) | count' examples/cli/registered-shapes.mlir
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
other syntax errors and semantic lowering diagnostics. Whole-document output
and semantic mutations require a complete document. `closure` additionally
requires registered reference semantics; configuring a func-like or call-like
shape alone does not supply vendor dependency semantics. See
[CLI examples](cli-examples.md) for the query language and selected-fragment
output contract.
