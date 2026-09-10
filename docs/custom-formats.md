# Custom formats

A dialect registry tells Zirium how to parse and lower custom operation syntax.
Generic quoted operations need no registry. Unknown custom operations use
best-effort recovery, which preserves their name, source text, and nested regions
without establishing that they can be verified or rewritten.

Use a bundled registry for existing dialects, or configure operation shapes and
format descriptions for your own syntax. The [preset reference](registry-presets.md)
lists the available dialects and explains their coverage.

## Choose a registry

| Registry | When to use it |
| --- | --- |
| Empty | Inspect generic quoted MLIR or recover unknown custom syntax. This is the Python parser's default. |
| Core | Parse `builtin.module`, `func.func`, `func.call`, and `func.return`. |
| Baseline | Use core plus `arith.constant`, `arith.addi`, `cf.br`, and `cf.cond_br`. This is the CLI's default. |
| Bundled preset | Parse selected custom forms from a dialect such as StableHLO, SCF, or Linalg. |
| Configured registry | Combine presets, select built-ins, and register caller-named operations. |

Registries containing the built-in `builtin.module` grammar accept `module` as
its shorthand, including named and nested modules. The empty registry does not
enable that grammar.

## Use a registry in Python

Pass the registry when parsing:

```python
import zirium

registry = zirium.DialectRegistry.from_name("stablehlo")
parsed = zirium.parse_text(source, registry=registry)
```

The parsed file retains the registry. Its semantic documents use that registry
for verification, editing, and custom printing.

To add your own operation names to an existing registry:

```python
import zirium

registry = zirium.DialectRegistry.baseline().extend_operation_shapes({
    "vendor.function": zirium.OperationShape.FUNC_LIKE,
    "vendor.invoke": zirium.OperationShape.CALL_LIKE,
})
parsed = zirium.parse_text(
    "module { vendor.function @declaration() }",
    registry=registry,
)
lowered = parsed.lower_strict("hybrid")
operation = lowered.document.operation_table("vendor.function").operation(0)
assert operation.symbol_name == "declaration"
```

`DialectRegistry.with_operation_shapes(...)` starts with core operations.
`existing_registry.extend_operation_shapes(...)` preserves the existing
registry. Both accept mappings and return a new registry.

## Configure a registry with JSON

Python and the CLI use the same configuration format. A configuration replaces
the caller's default registry; include every preset and built-in you need.

```json
{
  "presets": ["scf", "arith"],
  "builtins": [],
  "operation_shapes": [
    {"name": "vendor.function", "shape": "func_like"},
    {"name": "vendor.invoke", "shape": "call_like"}
  ],
  "operation_formats": [
    {"name": "vendor.widen", "format": "$operands attr-dict `:` type($operands) `into` type($results)"}
  ]
}
```

| Field | Meaning |
| --- | --- |
| `presets` | Bundled registry names. Defaults to an empty list. |
| `builtins` | Operation names selected from the baseline catalog. Required; may be empty. |
| `operation_shapes` | Exact operation names paired with reusable grammars. Required; may be empty. |
| `operation_formats` | Exact operation names paired with validated format descriptions. Defaults to an empty list. |

The built-in catalog contains the eight operations listed under core and baseline
above. Selecting a built-in uses its existing implementation.

### Operation shapes

Choose a shape whose operands, types, and regions match the operation's syntax.
Python exposes the same names as uppercase `OperationShape` members.

| Shape | Syntax it represents |
| --- | --- |
| `func_like` | Symbol name, arguments, optional results and attributes, and optional body. |
| `call_like` | Callee, operands, optional attributes, and a function type. |
| `unary_operand` | One SSA operand and a shared type or function type. |
| `binary_operands` | Two SSA operands and a shared type or function type. |
| `variadic_operands` | Zero or more operands followed by result types or a function type. |
| `optional_typed_operands` | Zero or more operands with a matching optional type list. |
| `attr_first_optional_typed_operands` | The same typed operand form, with the optional attribute dictionary first. |
| `literal_attribute` | An inline literal attribute followed by one result type. |
| `operand_clauses` | SSA operands mixed with fixed clauses and simple named attributes before a trailing type signature. |
| `region_clauses` | Operands and clauses with parsed regions and entry-block header bindings. |

Clause shapes retain structural information for inspection. They do not interpret
the dialect-specific meaning of fixed clauses. A similar-looking type trailer is
not enough to justify a shape: a type describing a stored value, for example,
must not become a result type for a store operation with no results.

### Format descriptions

`operation_formats` describes the order of operands or a literal value, an
optional attribute dictionary, fixed tokens, and type captures. Operand forms
accept `to` or `into` as the result-type separator, as in the JSON example.
The supported literals are `:`, `to`, and `into`. The typed-literal form is:

```
$value `:` type($value) attr-dict `:` type($result)
```

The registry validates descriptions when constructed. This is a limited format
language; Zirium does not interpret arbitrary MLIR assembly-format strings or
load ODS/TableGen files.

### Load and combine configurations

Python accepts JSON-compatible dictionaries, Pydantic models, or JSON files:

```python
from pathlib import Path
import zirium

config = zirium.RegistryConfig(
    builtins=["builtin.module"],
    operation_shapes=[
        zirium.OperationShapeConfig(name="vendor.function", shape="func_like"),
    ],
)
registry = zirium.DialectRegistry.from_config(config)
# A JSON-compatible dictionary is accepted by from_config as well.
Path("registry.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")
registry = zirium.DialectRegistry.from_file("registry.json")
schema = zirium.RegistryConfig.model_json_schema()
```

Pass additional arguments to combine configurations:

```python
registry = zirium.DialectRegistry.from_file("common.json", "vendor.json")
registry = zirium.DialectRegistry.from_config(common_config, vendor_config)
```

Identical registrations shared across configurations are included once.
Conflicting definitions and collisions between built-ins, shapes, and formats
are errors. Duplicate explicit entries within one configuration are also errors.
File order does not determine which definition wins.

Pydantic models validate structure without coercing types. The shared Rust
builder checks registered names, duplicates, and conflicts. In Python, invalid
configurations raise `ValueError`, non-JSON-compatible dictionary values raise
`TypeError`, and file I/O failures raise `OSError`. Loading does not retain the
input dictionary.

## Use a registry in the CLI

Install the binary separately from the Python wheel:

```sh
cargo install --path crates/zirium
zirium --registry examples/cli/registry.json \
  'filter(op("vendor.function")) | count' examples/cli/registered-shapes.mlir
zirium --registry common.json --registry vendor.json -f inspect.zirium input.mlir
```

Without `--registry`, the CLI uses the baseline registry. With one or more flags,
it uses their combined configuration. Files are read as UTF-8 JSON before MLIR
input; relative paths resolve against the working directory. Stdin remains
reserved for MLIR, and registry failures produce no query output.

Place `--registry` before an inline query. With `-f`/`--program-file`, registry
options may appear before or after the program-file pair, until the first input
path. Remaining arguments are input paths. `--` ends option parsing.

The CLI can select or count recovered unknown custom operations. It rejects
other syntax errors and semantic lowering diagnostics. Semantic mutations
require a complete document. `closure` also requires registered reference
semantics; a func-like or call-like shape alone does not supply vendor dependency
semantics.

Output uses the selected-fragment printer, even when the selection contains the
whole input. See the [query language reference](query-language.md) for that
output contract and [CLI examples](cli-examples.md) for worked commands.

## Use a registry in Rust

`DialectRegistry::core()` and `DialectRegistry::baseline()` return the built-in
registries. `DialectRegistry::from_name("stablehlo")` builds a bundled preset;
`from_config_file` and `from_config_files` load JSON configurations.
`RegistryConfig::from_json` followed by `build` constructs a registry from JSON text,
and `DialectRegistry::declarative(...)` selects built-ins by name.

Use the same registry for parsing, lowering, verification, editing, and
`print_with_registry`. Unlike Python's parsed file, Rust's `ParsedFile` does not
retain an owned registry. Text edits on registered syntax need
`apply_text_edits_with_registry`; `apply_text_edits` reparses with the empty
registry.

Rust callers can also construct static descriptors with parser, lowering,
verification, and printing callbacks. Python does not expose those callbacks.
See the [dialect API source](../crates/zirium/src/dialect.rs) for descriptor
contracts.

## Lowering, verification, and output

Shapes and format descriptions supply parsing and lowering conventions. They
do not define a vendor operation's verifier, symbol-table rules, or custom
printer. A func-like shape exposes a symbol name but does not make the operation
equivalent to `func.func` for every semantic analysis. Bundled dialect presets
likewise provide selected structural support, not full dialect implementations.

`lower_strict()` rejects lowering errors; it does not replace
`document.verify_semantics()`. Best-effort lowering can return an incomplete
document. Inspect its diagnostics and `semantically_complete` before editing or
choosing an output path.

In Python, choose output according to what you need:

- `custom_bytes()` prefers built-in assembly printers. Caller-defined shapes
  print in generic form. Built-in printers also fall back to generic form when
  they cannot faithfully represent the structure, properties, locations, typed
  constant attributes, or string-valued function and module names.
- `canonical_bytes()` produces deterministic generic output.
- `preserving_bytes()` supports eligible source-preserving edits on hybrid
  documents.

Validation failures from `write_custom()` and `write_canonical()` raise
`ValueError`; file creation, write, and flush failures raise `OSError`.
