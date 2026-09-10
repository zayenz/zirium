---
name: zirium-rust
description: Use Zirium's Rust crate to parse, inspect, verify, edit, and write textual MLIR. Use when Rust code depends on `zirium`, or a task mentions `ParsedFile`, `DialectRegistry`, semantic `Document`s, retention profiles, or Zirium's Rust API.
---

# Zirium Rust

## Goal

Use the Zirium crate through the API version selected by the repository. Keep
syntax work separate from semantic work, choose dialect support deliberately,
and use the least expensive retention profile that preserves the data needed by
the task.

Zirium evolves quickly. Inspect the repository's `Cargo.toml` and lockfile, then
use documentation matching that version. In a Zirium source checkout, prefer
the public rustdoc in `crates/zirium/src/lib.rs` and focused module docs. For a
published release, use its versioned [docs.rs documentation](https://docs.rs/zirium/).
Do not install, upgrade, or change the selected Zirium version unless asked.

## Start with the normal flow

```rust
use zirium::{
    dialect::DialectRegistry,
    parser::ParsedFile,
    printer::PrintLayout,
    semantic::{LoweringMode, RetentionProfile, lower_with_dialect_registry_and_retention},
};

let parsed = ParsedFile::parse(source.as_bytes())?;
let lowered = lower_with_dialect_registry_and_retention(
    &parsed,
    LoweringMode::Strict,
    RetentionProfile::SemanticOnly,
    &DialectRegistry::EMPTY,
);
let document = lowered.document.ok_or("semantic lowering failed")?;
document.validate_structure()?;
let output = document.canonical_bytes(PrintLayout::Pretty)?;
```

Use `ParsedFile` alone for lossless bytes, tokens, CST inspection, and syntax
diagnostics. Lower to a semantic `Document` for resolved SSA values, types,
symbols, verification, analysis, or edits. Preserve and report lowering
diagnostics instead of treating a missing strict document as an empty result.

## Make the important choices explicitly

- Use the empty registry for generic quoted operations. Use the same suitable
  registry for parsing, lowering, verification, editing, and custom printing
  when registered custom syntax is involved.
- Prefer `LoweringMode::Strict` when incomplete semantics would make a result
  misleading. Use best effort only when the caller can handle diagnostics and
  invalid sentinels.
- Use `SemanticOnly` for analysis and canonical output, `SyntaxOnly` when CST
  access must survive lowering, and `Hybrid` only for edits followed by
  source-preserving output.
- Choose original output from `ParsedFile`, canonical output from `Document`,
  or preserving output from a hybrid document according to the caller's actual
  contract. Canonical output may normalize formatting and SSA names.
- Treat operation and value IDs as document-owned, generation-checked handles.
  Do not reuse erased handles or pass handles between documents.

For caller-defined operation syntax, use the `zirium-custom-format` skill. For
query-language or command-line work, use `zirium-cli` instead.

## Verify proportionately

Run the repository's existing focused test or example first. In a Zirium source
checkout, `cargo test -p zirium` and `cargo doc -p zirium` are the broad checks;
do not add a large test suite for a small integration. Report the selected
registry, lowering mode, retention profile, diagnostics, and output contract
when they affect the result.
