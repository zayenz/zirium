# Shared registry configuration for the binary and Python

Status: implemented, including the follow-up requests for Pydantic models and
multiple configurations. This refines item 2 in
[the repository review](2026-09-08-review.md). The first implementation uses
JSON, Serde, and the two operation shapes already supported by Zirium.

Each file describes a complete registry; multiple files combine by union. Python accepts the same data as an
ordinary dictionary containing lists, dictionaries, and strings. Loading the
file and passing its decoded contents must produce the same registry and the
same validation errors.

## Data format

```json
{
  "builtins": [
    "builtin.module",
    "func.func",
    "func.call",
    "func.return",
    "arith.constant",
    "arith.addi"
  ],
  "operation_shapes": [
    {"name": "vendor.function", "shape": "func_like"},
    {"name": "vendor.invoke", "shape": "call_like"}
  ]
}
```

Both fields are required; either list may be empty. An empty registry is
`{"builtins": [], "operation_shapes": []}`. No operations are added implicitly.
In particular, loading this configuration does not extend the CLI's default
proving registry or Python's default empty registry.

`builtins` selects names from the existing declarative catalog:
`builtin.module`, `func.func`, `func.call`, `func.return`, `arith.constant`,
`arith.addi`, `cf.br`, and `cf.cond_br`. Including `builtin.module` also enables
its `module` shorthand, as the existing registry constructor does.

An operation-shape entry assigns a supported grammar to one exact operation
name. The initial shape strings are `func_like` and `call_like`. The names are
case-sensitive; `air.Func` and `air.func` remain distinct.

A list of entries is a little longer than a name-to-shape object, but duplicate
operation names remain visible to the shared validator. Ordinary JSON object
deserialization can otherwise overwrite a duplicate key before validation.
It also gives each operation an ordinary typed record in Rust and a Pydantic model in Python.

Explicit built-in names avoid an implicit, growing `proving` preset in saved
files. Existing programmatic presets remain useful; the file format needs only
one way to select its built-ins. There is no format-version field in this first
experimental implementation. Unknown fields are rejected, and an incompatible
future format change needs an explicit compatibility decision.

JSON is the sole file syntax initially. It works directly with Python's
standard library and Serde JSON. Do not add JSON5 comments, YAML, includes,
merging, or format autodetection. A later TOML reader can deserialize into the
same Rust configuration if there is a concrete need for it.

## CLI

```sh
zirium --registry registry.json 'select(op("vendor.function")) | count' input.mlir
zirium --registry registry.json -f inspect.zirium input.mlir
zirium -f inspect.zirium --registry registry.json input.mlir
zirium --registry registry.json 'select(op("vendor.invoke"))' < input.mlir
```

`--registry PATH` supplies a registry for every MLIR input in that invocation.
Repeat the flag to combine configurations. Shared built-ins and identical shape
definitions are included once; conflicting definitions fail without overriding
earlier files. Duplicates inside one file remain errors. Load and validate it before reading MLIR, including stdin, and
before producing any output. Relative paths resolve against the process's
working directory. The file must be UTF-8 JSON regardless of its suffix.
Do not use `-` for registry stdin; stdin remains available for MLIR.

With no flag, retain the current proving registry. The explicit file replaces
that default. Omitting a registry path is a usage error. Configuration parse and validation failures
use the current nonzero error exit and leave stdout empty.

Argument parsing accepts options before the inline query. When the query is
provided by `-f`, allow `--registry` before or after that pair, until the first
input path. After an inline query or the first input path, remaining arguments
are input paths. Support `--` to end option parsing. Document this boundary so
an input filename cannot unexpectedly become an option. This is a small
argument-parsing change; it does not require a CLI framework migration.

Keep one owned registry for the invocation and pass `&registry` to parsing,
lowering, query evaluation, and selection printing. Do not reconstruct it for
each file or revert to the proving registry in later phases.

## Python

The JSON example is also the Python input structure:

```python
from pathlib import Path
import json
import zirium

config = {
    "builtins": ["builtin.module", "func.func", "func.call", "func.return"],
    "operation_shapes": [
        {"name": "vendor.function", "shape": "func_like"},
        {"name": "vendor.invoke", "shape": "call_like"},
    ],
}
registry = zirium.DialectRegistry.from_config(config)
parsed = zirium.parse_text(source, registry=registry)

path = Path("registry.json")
path.write_text(json.dumps(config, indent=2), encoding="utf-8")
from_file = zirium.DialectRegistry.from_file(path)
# The binary can now consume the exact same file.
```

The public `RegistryConfig` and `OperationShapeConfig` definitions are Pydantic
models in `python/zirium/config.py`, exported at the package root. They use
`ConfigDict(strict=True, extra="forbid")`. `RegistryConfig.model_json_schema()`
provides the schema; `model_dump_json()` produces a file usable by the binary.
Registration conflicts and valid operation names remain the responsibility of
the shared Rust builder.

```python
config = zirium.RegistryConfig(
    builtins=["builtin.module"],
    operation_shapes=[
        zirium.OperationShapeConfig(name="vendor.function", shape="func_like"),
    ],
)
registry = zirium.DialectRegistry.from_config(config)
registry = zirium.DialectRegistry.from_config(common_config, vendor_config)
registry = zirium.DialectRegistry.from_file("common.json", "vendor.json")
```

`from_config` accepts a Pydantic `RegistryConfig` instance or a JSON-compatible
dictionary with the same structure. Additional positional arguments combine
more configurations, and may mix models and dictionaries. Shape values are
strings, not `OperationShape.FUNC_LIKE` objects. Model instances are converted with `model_dump()` before serialization.
The current enum-based
`with_operation_shapes` and `extend_operation_shapes` APIs remain available
with their existing defaults and mapping support. No existing constructor is
silently changed to use the new file format.

For the first implementation, serialize the small Python configuration with
standard-library `json.dumps(..., allow_nan=False)` and call the same Rust JSON
reader used for files. This adds one small serialization step at registry
construction, outside MLIR parsing. It avoids separate field validation in
Python and a generic Python-to-Serde conversion layer. The method should not
retain the input dictionary or observe subsequent mutations to it.

File open/read errors become `OSError`. Malformed JSON, missing or unknown
fields, invalid shapes, and registry conflicts become `ValueError`.
Non-JSON-compatible Python objects fail with `TypeError` from serialization.
File errors include the path; JSON errors retain Serde's line and column.
Programmatic configuration errors need not claim a source-file location.

The returned registry is retained by `File` and `Document`, using the existing
owned registry path. Registry reuse and concurrent reads continue to work as
they do for `declarative` and `extend_operation_shapes`.

## Rust implementation

Use ordinary derived deserialization in `crates/zirium/src/dialect/config.rs`,
exposed through the dialect module:

```rust
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegistryConfig {
    pub builtins: Vec<String>,
    pub operation_shapes: Vec<OperationShapeConfig>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationShapeConfig {
    pub name: String,
    pub shape: OperationShape,
}

// Add Deserialize and #[serde(rename_all = "snake_case")]
// to the existing OperationShape enum.
```

Serde supports rejecting unknown struct fields with `deny_unknown_fields` and
renaming enum variants through `rename_all`; use those derives instead of
manual JSON field handling. See [Serde container attributes](https://serde.rs/container-attrs.html).

Add `serde` with `derive` and `serde_json` to the core crate. The core currently
has no dependencies, so this is a deliberate cost of the requested shared
reader. Keep configuration loading off the parse path. Do not add a separate
configuration crate or optional-feature matrix for this small API.

The shared API provides:

- `RegistryConfig::from_json(&str)` deserializes the schema.
- `RegistryConfig::build(&self)` validates and constructs an owned registry.
- `DialectRegistry::from_config_file(path)` reads UTF-8 JSON and invokes both.
- `RegistryConfig::build_many` combines individually validated configurations;
  `DialectRegistry::from_config_files` loads multiple files and uses it.

The file helper returns an error preserving the I/O, JSON, or registry failure;
CLI and Python add their own presentation. Reuse `DeclarativeRegistryError`
for registry failures where possible. Do not build a parallel diagnostic
framework.

Construction is the existing composition:

```text
config.builtins -> DialectRegistry::declarative(...)
config.operation_shapes -> registry.extend_operation_shapes(...)
```

Deserialization and construction have different jobs. Serde checks fields and
types. The builder checks operation names, duplicate registrations, and
conflicts. Python and the CLI call this builder, rather than separately
implementing the rules.

Validation rules:

- Reject missing fields, unknown fields, nulls, and unsupported shape strings.
- Reject unknown or repeated built-in names.
- Reject repeated custom operation names, even when the shapes agree.
- Reject a custom name that conflicts with a selected built-in or its enabled
  module alias. Preserve the existing rule for catalog names not selected in
  `builtins`; selecting no built-in does not reserve its name globally.
- Require a custom name to lex as exactly one complete `BareIdentifier`, with
  no diagnostics or surrounding trivia. Apply this validation in
  `extend_operation_shapes` so existing Rust/Python constructors agree too.
- Construct the whole registry successfully before returning it. Report the
  offending field or operation name; for semantic validation a source span is
  unnecessary.

## Meaning of a registered shape

This configuration supplies the same syntax and lowering support as the
existing programmatic API. Both generic and custom forms remain accepted for
registered operation names. Recovery still handles unregistered operations.

It does not add semantic roles. A func-like shape does not automatically become
a symbol table or acquire isolation and terminator rules. A call-like shape
does not make `closure` understand vendor call dependencies. Those rules need
explicit design and must not be inferred merely from a familiar grammar.

For complete documents, CLI selection/count and the existing edit operations
can use the configured registry subject to the current semantic checks. Custom
shape operations print in generic form when regenerated. Unknown operations
can still make a document incomplete and prevent edits or root output.

The first release must document and test a clear error from `closure` when it
reaches a vendor shape without supported reference semantics. This is preferable
to returning a dependency slice that silently omits its callee. Semantic roles
remain a separate follow-up, not a prerequisite for sharing a registry file.

## Implementation sequence and evidence

1. Add the data types, Serde reader, shared builder, and lexical name validation.
   Check construction against the existing declarative/shape constructors.
2. Add `--registry` and retain the resulting registry through the CLI pipeline.
   Preserve invocations without the new flag and both query input forms.
3. Add the Python constructors and runtime Pydantic definitions. Use one checked-in
   JSON fixture for the CLI and Python integration checks.
4. Update the custom-format guide and CLI examples. Keep the Cargo installation
   route; bundling the binary into Python wheels is separate work.

Implemented integration checks cover:

- Load one JSON file in the CLI and Python. Compare names, symbols, signatures,
  and operand bindings for vendor func/call input, rather than only op counts.
  The CLI count must match Python's operation table; reparse CLI root output
  with that registry and compare semantic structure.
- Pass the decoded JSON object to Python and check equivalence with `from_file`.
  Drop or mutate the input dictionary and confirm the registry remains usable.
- Exercise duplicate/conflicting names, a misspelled field, an unknown shape,
  malformed JSON, and a missing file. File failures must occur before consuming
  MLIR stdin and must not emit query output.
- Check that the explicit registry replaces defaults: an unlisted proving
  operation remains unregistered. Check the documented closure boundary.

Run the existing Rust/Python quality checks and verify dependency compatibility
with Rust 1.88. No changes to the MLIR grammar itself are needed for the two
existing shapes.


## Implementation notes

Serde's derived struct deserializer also accepts positional arrays. The shared
JSON reader therefore checks that the top-level registry and individual shape
records are objects, matching Pydantic. Typed deserialization runs first to
retain duplicate-field checks and normal line/column diagnostics. The extra
shape check only runs while loading these small configuration files.

The runnable fixture is [examples/cli/registry.json](../examples/cli/registry.json),
with [registered-shapes.mlir](../examples/cli/registered-shapes.mlir). The earlier
copy under `plans/examples` is only a design sample.
