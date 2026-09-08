# Changelog

## 0.0.5

Zirium 0.0.5 improves recovery and semantic inspection of custom-format MLIR.

- Resolve enclosing block arguments from nested regions.
- Decode quoted symbol-reference segments in Python's `symbol_value` accessor.
- Recover result declarations, operand uses, and trailing attribute dictionaries
  from unregistered custom-format operations.
- Suppress semantic diagnostics caused only by successful custom-operation
  recovery. The syntax diagnostic and `is_unparsed` identify each recovered op.
- Compose operation shapes with empty, core, proving, or declarative registries
  through `extend_operation_shapes`.
- Expose form-independent `symbol_name`, `signature`, and `callee` operation
  roles in Rust and Python.
- Add stable machine-readable categories to every Rust and Python semantic
  diagnostic.

## 0.0.4

Zirium 0.0.4 fixes function types used as attribute values in generic-form
MLIR and nested custom types.

- Parse and lower function types in attribute dictionaries and arrays.
- Preserve arrows inside opaque type parameters, including nested function
  types such as `!dialect.box<() -> ()>`.
- Verify normalized function identity and signatures across generic and custom
  forms, including StableHLO operations.

## 0.0.3

Zirium 0.0.3 extends semantic inspection for custom-format MLIR encountered in
production compiler dumps.

- Parse and lower `loc(unknown)` and fused locations with dictionary metadata.
- Represent type-suffixed decimal literals as integers when their value fits in
  `i128`, while retaining genuinely wide literals as wide numbers.
- Expose indexed array and dictionary elements through Python
  `SemanticAttribute` handles, including nested typed access and exact child
  spellings.

## 0.0.2

Zirium 0.0.2 improves best-effort inspection of real-world textual MLIR. The
API remains experimental, and this release makes no ABI stability guarantee.

- Handle standard `module` syntax and caller-provided func-like and call-like
  operation shapes.
- Recover unknown custom operations and nested regions as individual,
  diagnostic-bearing semantic operations.
- Expose richer Python inspection, including exact type and attribute
  spellings, scalar attributes, and stable value identity.
- Add a compact `zirium` CLI for selection and edit pipelines, with reusable
  program files.
- Parse and lower builtin `array<...>` dense-array attributes.
- Bound recursive alias expansion across type, attribute, affine, memref, and
  location aliases. Rust and Python callers can select the limit.
- Make `lower_with_dialect_registry` and
  `lower_with_dialect_registry_and_retention` the Rust lowering entry points.
  Remove the no-op shared registry and fixture lowering helpers.
- Publish version-specific wheels for CPython 3.11 through 3.14 on Linux x86_64
  and macOS arm64. ABI3 remains out of scope.

## 0.0.1

First public experimental release of Zirium.

- Lossless byte-oriented textual MLIR parsing and recovery.
- Separate semantic lowering, verification, editing, and output paths.
- Rust core crate and typed Python bindings.
- Python support for CPython 3.11 through 3.14, subject to the published
  artifact matrix.

The API is experimental and may change before 1.0.
