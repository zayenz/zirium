# Custom formats

A dialect registry tells Zirium how to parse and lower custom operation syntax.
Generic quoted operations need no registry. Unknown custom operations use
best-effort recovery, which preserves their name, source text, and nested regions
so you can inspect them even when semantic verification or rewriting is unavailable.

Use a bundled preset for an existing dialect, or configure operation shapes and
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

Registries containing `func.return` also accept `return` shorthand. Return type
lists may be parenthesized or comma-separated without parentheses. These names
are reserved against conflicting configured shapes and formats when the
corresponding built-in grammar is present.

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

registry = zirium.DialectRegistry.baseline().extend_operation_shapes(
    {
        "vendor.function": zirium.OperationShape.FUNC_LIKE,
        "vendor.invoke": zirium.OperationShape.CALL_LIKE,
    }
)
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

Registries can be inspected after construction:

```python
names = registry.operation_names()
assert "vendor.function" in names
assert registry.operation_shape("vendor.function") == "func_like"
assert registry.operation_shape("missing.operation") is None
assert registry.call_target_attribute("func.call") == "callee"
```

`operation_names()` returns a sorted tuple containing built-ins, preset entries,
caller-supplied shapes, configured formats, and alternatives. `operation_shape()` returns the
configuration spelling for shape-backed operations. It returns `None` both for
format-backed and unregistered names; use membership in `operation_names()` to
distinguish those cases.

`operation_alternatives(name)` returns the locally ordered alternatives as
`("shape", value)` or `("format", value)` pairs, and returns `None` for a
single-grammar or unregistered operation.

## Configure a registry with JSON

Python and the CLI use the same configuration format. A configuration replaces
the caller's default registry; include every preset and built-in you need.

```json
{
  "presets": ["scf", "arith"],
  "builtins": [],
  "operation_shapes": [
    {"name": "vendor.function", "shape": "func_like"},
    {"name": "vendor.invoke", "shape": "call_like", "callee_attribute": "target"}
  ],
  "operation_formats": [
    {"name": "vendor.widen", "format": "$operands attr-dict `:` type($operands) `into` type($results)"}
  ],
  "operation_alternatives": [
    {
      "name": "vendor.literal_or_dimension",
      "alternatives": [
        {"format": "$value `:` type($value) attr-dict `:` type($result)"},
        {"shape": "operand_clauses"}
      ]
    }
  ]
}
```

| Field | Meaning |
| --- | --- |
| `presets` | Bundled registry names. Defaults to an empty list. |
| `builtins` | Operation names selected from the baseline catalog. Required; may be empty. |
| `operation_shapes` | Exact operation names paired with reusable grammars. A `call_like` entry may set `callee_attribute`. Required; may be empty. |
| `operation_formats` | Exact operation names paired with validated format descriptions. Defaults to an empty list. |
| `operation_alternatives` | Exact operation names paired with two or more ordered shape or format alternatives. Defaults to an empty list. |

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

`call_like` uses the `callee` attribute by default. Set `callee_attribute` when
the generic form stores the target symbol under another dotted ASCII attribute
name. `reachable` and `closure` use this metadata for direct-call traversal.

Clause shapes retain structural information for inspection. They do not interpret
the dialect-specific meaning of fixed clauses. A similar-looking type trailer is
not enough to justify a shape: a type describing a stored value, for example,
must not become a result type for a store operation with no results.

### Format descriptions

`operation_formats` compiles a sequence of captures, exact tokens, an optional
attribute dictionary, and type assignments. The supported elements are:

| Element | Input and semantic role |
| --- | --- |
| `$operands` | Zero or more comma-separated SSA operands. |
| `$operands[0]`, `$operands[1]`, ... | A fixed number of SSA operands. Indices must occur once, in order from zero. Put a `` `,` `` literal between them. |
| `$value` | An inline literal attribute stored as `value`. |
| `$callee` | A symbol reference stored as `callee` and recognized by direct-call dependency traversal. |
| `attr-dict` | An attribute dictionary when one is present. The directive may occur at most once and may be omitted from the description. |
| `` `token` `` | One exact MLIR lexer token, such as `` `:` ``, `` `->` ``, `` `to` ``, or `` `as` ``. |
| `type(...)` | One type assigned to one or more listed targets. Aggregate operand and result targets also accept parenthesized type lists. |
| `types($operands)` | A bare comma-separated list containing exactly one type per operand. |

These common recipes are examples of the compositional rules above, rather than
a closed list of supported formats:

| Recipe | Format | Type bindings |
| --- | --- | --- |
| Variadic operands with a shared type | <code>$operands attr-dict `:` type($operands) `->` type($results)</code> | The input type is broadcast to every operand; the result binding accepts one type or a parenthesized list. |
| Variadic operands with individual types | <code>$operands attr-dict `:` types($operands) `->` type($results)</code> | The bare input list must contain one type per operand; results accept one type or a parenthesized list. |
| Fixed operands with different sharing groups | <code>$operands[0] `,` $operands[1] `,` $operands[2] `:` type($operands[0]) `,` type($operands[1], $operands[2]) `->` type($results)</code> | Indexed operands are captured once in order; each printed type binds to its listed operands or results. |
| One type shared by operands and results | <code>$operands attr-dict `:` type($operands, $results)</code> | The printed type binds to every operand and every result. |
| Typed literal result | <code>$value `:` type($value) attr-dict `:` type($result)</code> | The first type belongs to the literal attribute; the second binds the single result. |
| Direct call | <code>$callee `(` $operands `)` attr-dict `:` type($operands) `->` type($results)</code> | The callee is stored as a symbol reference; operand and result types follow the aggregate rules. |

The original conversion and typed-literal descriptions remain valid:

```
$operands attr-dict `:` type($operands) `to` type($results)
$value `:` type($value) attr-dict `:` type($result)
```

`type($operands)` preserves its original broadcast rule: one type applies to
every operand. It also accepts a parenthesized list. `type($results)` accepts one
result type or a parenthesized list. Use several targets when one printed type
has several roles:

```
$operands attr-dict `:` type($operands, $results)
```

Use indexed operands when arity or nonuniform sharing matters. This three-input
example assigns the second printed type to both branch values:

```
$operands[0] `,` $operands[1] `,` $operands[2] attr-dict `:` type($operands[0]) `,` type($operands[1], $operands[2]) `->` type($results)
```

Every SSA operand and result needs a type assignment for semantic lowering.
The registry validates directive structure when constructed and validates list
sizes against the operation when lowering. A malformed description violates
these rules, for example by mixing `$operands` with indexed operands, assigning
one target twice, or naming a type target that was not captured. Registry
construction rejects it with the operation name and failing rule. Syntax that
needs optional groups, repetition, regions, ODS/TableGen constructs, or other
unsupported directives cannot be expressed by a format description; use a
reusable shape or built-in implementation when one matches. Writing an
unsupported directive in a description reports it as unknown during registry
construction.

### Operation alternatives

Use `operation_alternatives` when one operation has two or more complete
spellings. Each alternative contains exactly one `shape` or `format`. Zirium
tries them in listed order and records the selected branch for lowering. If all
branches fail, it recovers the operation once and reports a format mismatch.

Alternatives are explicit so an accidental duplicate operation remains an
error. Identical ordered entries can be shared by multiple registry files. A
different alternative or a different order is a conflict; file order never
selects a definition.

### Load and combine configurations

Python accepts JSON-compatible dictionaries, Pydantic models, or JSON files:

```python
from pathlib import Path
import zirium

config = zirium.RegistryConfig(
    builtins=["builtin.module"],
    operation_shapes=[
        zirium.OperationShapeConfig(name="vendor.function", shape="func_like"),
        zirium.OperationShapeConfig(
            name="vendor.invoke", shape="call_like", callee_attribute="target"
        ),
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

Options may appear before or after an inline query or program-file pair. The
first non-option argument is the inline query; later non-option arguments are
input paths. Option parsing continues between input paths until `--`, which is
required before a path beginning with a dash.

The CLI can select or count recovered unknown custom operations. It rejects
other syntax errors and semantic lowering diagnostics. Semantic mutations
require a complete document. `closure` also requires registered reference
semantics. Function-like definitions and direct calls have built-in symbol
conventions; other vendor symbol uses must define their semantics in Rust.

Output uses the selected-fragment printer, even when the selection contains the
whole input. See the [query language reference](query-language.md) for that
output contract and [CLI examples](cli-examples.md) for worked commands.

## Use a registry in Rust

`DialectRegistry::core()` and `DialectRegistry::baseline()` return the built-in
registries. `DialectRegistry::from_name("stablehlo")` builds a bundled preset;
`from_config_file` and `from_config_files` load JSON configurations.
`RegistryConfig::from_json` followed by `build` constructs a registry from JSON text,
and `DialectRegistry::declarative(...)` selects built-ins by name.

Caller-owned operation names must be registered through configured formats or
shapes, such as `RegistryConfig` or `extend_operation_shapes(...)`.

`operation_names()` enumerates static, shape-backed, and format-backed
registrations. `operation_shape(name)` returns the `OperationShape` assigned to
a shape-backed name. `call_target_attribute(name)` returns the target attribute
for a registered direct call, if any. `declarative(...)` deliberately selects
only built-in implementations.

Use the same registry for parsing, lowering, verification, editing, and
`print_with_registry`. Unlike Python's parsed file, Rust's `ParsedFile` does not
retain an owned registry. Text edits on registered syntax need
`apply_text_edits_with_registry`; `apply_text_edits` reparses with the empty
registry.

Rust callers can also construct static descriptors with parser, lowering,
verification, and printing callbacks. Python does not expose those callbacks.
See the [dialect API source](../crates/zirium/src/dialect.rs) for descriptor
contracts.

## Verify and write semantic documents

Shapes and format descriptions supply parsing and lowering conventions. They
do not define a vendor operation's verifier, symbol-table rules, or custom
printer. A `func_like` shape defines its `sym_name` in the enclosing symbol
table, and a `call_like` shape declares a symbol use through `callee` or its
configured `callee_attribute`. These conventions support direct reference
traversal without making the operation equivalent to `func.func` for every semantic analysis. Bundled dialect presets
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
