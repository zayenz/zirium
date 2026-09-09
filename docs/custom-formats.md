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
| StableHLO preset | Core plus 96 common StableHLO custom forms. |
| TOSA preset | Core plus 93 TOSA tensor, shape, control-flow, and utility forms. |
| SCF preset | Core plus all 12 SCF operations, including structured regions and loop header bindings. |
| Linalg preset | Core plus 97 core, structured, and generated named Linalg operations. |
| OpenACC preset | Core plus 35 mapping, bounds-accessor, region, and terminator forms. |
| Affine preset | Core plus 4 of the 16 Affine operations. |
| Declarative | A selected subset of the proving catalog. |
| Operation shapes | Caller-named operations using one of the supported structural grammars. |

Core, proving, and declarative registries containing `builtin.module` accept
`module` as its shorthand, including named and nested modules. The empty
registry does not enable this grammar.

The declarative registry selects existing implementations. It does not
interpret arbitrary MLIR assembly-format strings or load ODS/TableGen files.
Rust callers can also construct static descriptors with parser, lowering,
verification, and printing callbacks. Python does not expose those callbacks.

The bundled `stablehlo` preset provides structural lowering for 96 custom
forms: unary and binary elementwise operations, constants, typed returns,
ordinary variadic signatures, and operations that mix operands with fixed or
named clauses before a trailing type signature. The latter group includes
`broadcast_in_dim`, `compare`, `concatenate`, `convolution`, `custom_call`,
`dot_general`, dynamic slices, `iota`, `pad`, `select`, `slice`, and
`transpose`. Their operands, result types, ordinary attribute dictionaries,
and simple `name = value` clauses are available to semantic and CLI queries.

The preset also retains the regions and block arguments of 14 region-bearing
forms, including `reduce`, `reduce_window`, `scatter`, `sort`, and `while`.
This is structural support, not a StableHLO implementation: Zirium does not
apply StableHLO verification, type inference, execution semantics, VHLO, or
portable-artifact compatibility. Unsupported custom forms continue through
best-effort recovery, and generic quoted StableHLO operations need no preset. The embedded
[registry file](../crates/zirium/registries/stablehlo.json) is the exact preset
definition. This surface was checked against StableHLO 1.20.1. The unversioned
preset name tracks Zirium releases; it does not claim support for the complete
StableHLO 1.20.1 opset or its portable artifact format.

The `tosa`, `scf`, `linalg`, `acc`, and `affine` presets were checked against LLVM 22.1.0.
TOSA registers 93 of the 94 operations defined by its main, utility, and shape
operation files; `tosa.variable` remains on the generic recovery path because
its custom symbol/type form has no reusable structural signature. SCF registers
all 12 operations. Linalg registers its 16 core/structured operations and 81
generated named operations. Tensor-result named Linalg forms expose their
trailing result types; buffer forms without a result signature remain usable
through recovery where their custom spelling has no safe structural boundary.

OpenACC registers 35 of its 54 operations. This includes all 16 data-entry
mapping operations, the four bounds accessors, 12 single-region constructs,
the single-region form of `acc.private.recipe`, and both terminators. Mapping
forms with a trailing `attributes` dictionary, loop result forms, and region
forms with trailing attributes use whole-operation recovery. The other 19
operations remain on that recovery path: `acc.bounds`, `acc.atomic.read`,
`acc.atomic.write`, `acc.copyout`, `acc.delete`, `acc.detach`,
`acc.update_host`, `acc.firstprivate.recipe`, `acc.reduction.recipe`,
`acc.enter_data`, `acc.exit_data`, `acc.declare_enter`, `acc.declare_exit`,
`acc.routine`, `acc.init`, `acc.shutdown`, `acc.set`, `acc.update`, and
`acc.wait`. Their custom spellings either have no safe trailing boundary in
the current shapes, require keyword-separated regions, or would incorrectly
imply SSA result types.

Affine registers 4 of its 16 operations: `affine.for`, `affine.if`,
`affine.linearize_index`, and `affine.yield`. The two structured control-flow
forms expose their regions, header bindings, operands, and result slots;
trailing attribute dictionaries after the final region use whole-operation
recovery. The linearization form exposes its index operands and single index
result, while the terminator preserves its optional typed operands. The other
12 operations remain on the recovery path: `affine.apply`, `affine.min`,
`affine.max`, `affine.parallel`, `affine.load`, `affine.store`,
`affine.vector_load`, `affine.vector_store`, `affine.prefetch`,
`affine.delinearize_index`, `affine.dma_start`, and `affine.dma_wait`.
Their custom forms either lack a safe trailing boundary, need parallel header
bindings, derive a result type from a memref element type, or spell types for a
no-result operation. Assigning the current clause shapes to those forms would
lose structure or invent semantic result types.

`region_clauses` is used for operations such as `scf.for`, `scf.if`,
`tosa.while_loop`, and `linalg.generic`. Their operation regions, explicit block
labels, and typed block arguments use Zirium's ordinary semantic region model.
Header bindings written as `%argument = %initial` are attached to implicit entry
blocks as opaque-typed arguments when the custom syntax does not spell their
types locally. This preserves definition/use structure without claiming dialect
type inference.

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
a matching optional type list. `OPERAND_CLAUSES` captures SSA operands and
simple named attributes around otherwise opaque fixed clauses, followed by a
shared, conversion, or function-type signature. It is useful for inspection;
it does not interpret the clauses' dialect-specific meaning. `REGION_CLAUSES`
adds one or more parsed operation regions and entry-block header bindings to
that structural model.

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
expression forms. `operand_clauses` accepts fixed and named clauses around SSA
operands before a trailing type signature. `region_clauses` additionally parses
operation regions and their block arguments.

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
