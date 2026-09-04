# Changelog

## 0.0.2

Zirium 0.0.2 improves best-effort inspection of real-world textual MLIR. The
API remains experimental, and this release makes no ABI stability guarantee.

- Handle standard `module` syntax and caller-provided func-like and call-like
  operation shapes.
- Recover unknown custom operations and nested regions as individual,
  diagnostic-bearing semantic operations.
- Expose richer Python inspection, including exact type and attribute
  spellings, scalar attributes, and stable value identity.
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
