# Changelog

## Unreleased

- Add an opt-in release profiling test for query parsing and evaluation, with
  direct-scan comparisons and separate document-width and dependency-depth
  cases. Record the initial timings and the deep fixed-point scaling limit.

- Reengineer the query language around an implicit input selection and output.
  Empty programs print the input; `filter` replaces `select` and always tests
  the current selection. `input` explicitly returns to the whole document.
- Make `union`, `intersect`, and `except` infix operators on full selection
  queries, with grouping and pipe precedence. Remove predicate-based set stages.
- Make `closure` one dependency expansion step and add general
  `fixpoint(query)` repetition until unchanged, with cycle detection.
- Make `root` expand the outermost selected operations and their descendants.
  Use `input | emit` for whole-document output after an edit.
- Add `emit` as a pipeline tap, preserving intermediate output before later
  edits. Omitted final emission is implicit; a trailing explicit emit is not
  duplicated. The CLI buffers emissions until all inputs succeed.
- Add boolean literals, `#` comments, and source carets in query diagnostics.
  Rename string attribute equality to `string_attr_eq`.
- Replace the Rust query's single-result evaluation API with an emission
  callback. Query parsing exposes composable expressions rather than a
  distinguished initial predicate.

## 0.0.8

Zirium 0.0.8 fills gaps in generic attribute and operation-signature parsing.

- Parse trailing locations on every built-in declarative operation and preserve
  them through semantic lowering and source-preserving edits, including on named
  `builtin.module` operations.
- Keep quoted attribute strings containing `->` as strings, including inside
  arrays and nested dictionaries, while recognizing function types only from
  arrows outside quoted and nested syntax.
- Accept unit shorthand for every bare or quoted dictionary key in attributes,
  properties, aliases, and nested dictionaries. Comments after keys no longer
  become part of semantic names, and quoted keys are decoded consistently.

## 0.0.7

Zirium 0.0.7 adds shared registry configuration, an initial StableHLO preset,
and consistent handling for quoted symbols and aliased attribute values.

- Add a bundled `stablehlo` registry preset, available by name and through the
  JSON configuration's optional `presets` field. The initial subset parses and
  lowers binary elementwise custom forms and `stablehlo.return`.
- Load and combine JSON registry files with `zirium --registry FILE` and Python's
  `DialectRegistry.from_file`. Python's `from_config` accepts JSON-compatible
  dictionaries or the new Pydantic `RegistryConfig` / `OperationShapeConfig`
  models. All entry points share Serde deserialization and registry validation.

- Fall back to generic printing when built-in custom assembly cannot preserve
  operation structure, locations, properties, or supported attribute spellings.
  Malformed generic arithmetic operations no longer panic in custom printing.
- Accept named and nested `module` shorthand in proving and declarative
  registries containing `builtin.module`; remove the CLI's source rewrite.
- Reject CLI semantic lowering errors even when an unknown custom sibling
  requires best-effort recovery.
- Resolve quoted and nested symbol paths consistently in lookup, diagnostics,
  closure queries, and output.
- Preserve exact or canonical element spellings through attribute aliases.
- Expose dense-array element spellings in Python, accept general mappings for
  operation shapes, and report file-print validation failures as `ValueError`.

## 0.0.6

Zirium 0.0.6 raises the minimum supported Rust version to 1.88.

- Use Rust 2024 let chains across parsing, printing, and semantic processing.
- Test the minimum supported Rust version in CI.

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
