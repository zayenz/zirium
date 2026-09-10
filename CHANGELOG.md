# Changelog

## Unreleased

- Preserve both sides of StableHLO dot dimension clauses and accept `return`
  shorthand plus unparenthesized multi-operand return type lists.
- Warn when CLI recovery leaves semantic information incomplete; add `--strict`,
  repeatable `--preset`, `--list-presets`, `--help`, and `--version`. Options may
  also follow the query. Read input files one at a time while retaining atomic
  stdout across the whole invocation.
- Bound query work and stream size with configurable evaluation limits. Use a
  worklist for `fixpoint(closure)` and avoid repeated retained-subtree traversal.
  Queries emitting each iteration retain their step-by-step behavior.
- Reject undecodable attribute projections instead of silently dropping values;
  share string decoding with the semantic API.
- Add `names`, `operand_types`, `result_types`, `dialect`, and `result_type` for
  inspection, and include type arrays in operation JSON. Add indexed `defs` and
  `users`, plus SSA-only `slice` that stops at block arguments.
- Correct the query profiling fixture's expected use-site count and document
  recovery, compact reductions, slicing, and fixed-point limits.

## 0.0.12

Zirium 0.0.12 expands the registry catalog and makes configured coverage
inspectable.

- Add bundled presets covering selected custom forms across LLVM 22.1 dialects,
  including arithmetic, control flow, memory, tensors, vectors, GPU targets,
  parallel execution, lowering, transformation, and constraints.
- Rename `DialectRegistry.proving()` to `DialectRegistry.baseline()`, make the
  baseline assembly handlers operation-specific, and register `arith.addi`
  through its reusable declarative shape.
- Add the attribute-first optional typed-operand shape, accept `into` format
  separators and nested arrows, preserve operation boundaries around type lists
  and regions, and support bare opaque dialect types and attributes.
- Expose complete registry operation names and caller-supplied shape labels in
  Rust and Python, including entries loaded from bundled presets and JSON
  format descriptions.
- Store parser events in a four-byte transient representation and release the
  event buffer before trimming the completed CST, reducing peak parse memory.
- Model query results as ordered streams with explicit `unique`, add
  predicate-based ancestor lookup through `root(predicate)`, rename downward
  expansion to `subtree`, project attributes with `attr`, and emit streams as
  JSON with `json`.

## 0.0.11

Zirium 0.0.11 broadens structural custom-operation support for compiler and
machine-learning dialects. This support covers parsing and lowering; execution
semantics are outside its scope.

- Expand the StableHLO preset from 16 to 96 forms, including constants,
  attribute-heavy operations, compact reductions, and all region-bearing
  definitions reviewed in StableHLO 1.20.1.
- Add `tosa`, `scf`, and `linalg` presets. Their common operands, result types,
  attributes, regions, block arguments, and ownership relationships are
  available to Rust, Python, and binary queries.
- Add a core-only `dlti` preset for a dialect whose LLVM 22.1 surface consists
  of six opaque attributes and no operations or types.
- Add reusable operand-clause and region-clause operation shapes. Region
  clauses retain explicit block arguments and loop-style header bindings
  without implementing dialect-specific type inference.
- Add validated configurable operation formats for common operand and literal
  layouts, available through JSON registry files and Python configuration.
- Accept conversion type trailers and literal-attribute dictionaries, expose
  bundled preset names, and recover complete operations after registered-shape
  mismatches.
- Reuse lexer tokens while compacting the CST and extend the parser benchmark
  with whole-parser peak-memory measurements.
- Add a Zirium custom-format research skill grounded in the current user
  documentation.

## 0.0.10

Zirium 0.0.10 expands custom-operation support and improves Python inspection
of large or malformed MLIR inputs.

- Add unary-operand, variadic-operand, and literal-attribute operation shapes
  to Rust and Python registries. Binary operations now accept result types and
  bare or parenthesized function-type trailers.
- Recover cleanly from binary-operation operand-count mismatches so later
  operations remain available during best-effort lowering.
- Decode correctly sized hexadecimal IEEE bit patterns for `f16`, `bf16`,
  `f32`, and `f64` scalar attributes, including infinities and NaNs.
- Expose syntax diagnostic messages and semantic diagnostic kinds in Python.
  Add `File.line_column()` for converting byte offsets to one-based source
  locations.
- Expose the installed Python distribution version as `zirium.__version__`.
- Lazily index syntax operations for fast repeated `File.operation()` access.
  Add a stress benchmark modeled on production input and record its scaling baseline.

## 0.0.9

Zirium 0.0.9 revises the experimental query language for composable selection,
fragment editing, and intermediate output. Existing queries need updating;
see the [query language reference](docs/query-language.md).

- Start from an implicit input selection and emit the final result implicitly.
  Empty programs print the input. `filter` replaces `select` and always tests
  the current selection; `input` returns to the whole document.
- Combine complete selection queries with infix `union`, `intersect`, and
  `except`. Pipes bind more tightly than set operators; parentheses group
  queries. Predicate-based set stages are removed.
- Make `closure` one dependency expansion step. Use `fixpoint(query)` to repeat
  until unchanged, with cycle detection for nonconverging queries.
- Make `root` expand the outermost selected operations and their descendants.
  Use `input | emit` for whole-document output after an edit.
- Let `emit` print intermediate selections while passing them onward,
  including inside fixed points. Emissions capture earlier edits without
  being affected by later edits. The CLI buffers output until all inputs
  succeed and avoids duplicating a trailing explicit emission.
- Add boolean literals, `#` comments, and source carets in query diagnostics.
  Rename string attribute equality from `attr` to `string_attr_eq`.
- Replace the Rust query's single-result evaluation API with an emission
  callback and expose composable parsed expressions.
- Add executable CLI examples, a query-language reference, and a compact
  StableHLO decoder example.
- Add opt-in release profiling for parsing and evaluation, with direct-scan
  comparisons and width/depth scaling cases. Deep fixed-point closure still
  has approximately quadratic chain-depth cost; the
  [profiling baseline](docs/architecture/query-profiling.md) records this limit.

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
- Accept named and nested `module` shorthand in baseline and declarative
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
- Compose operation shapes with empty, core, baseline, or declarative registries
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
  `i128`, while retaining wider literals as wide numbers.
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
  platform and interpreter matrix.

The API is experimental and may change before 1.0.
