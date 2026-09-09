# Zirium 0.0.10 usability findings

This plan covers notes 29–34 from the September 2026 evaluation. Each implementation part gets focused verification and its own commit.

The lossless CST remains part of Zirium's core design. Memory work will focus on the representation and transient construction costs while keeping syntax and hybrid retention intact.

## 1. Recover whole operations from shape mismatches (brief 32)

- Reproduce the fabricated `to` and attribute-key sibling operations.
- Add a parser checkpoint around shaped-operation parsing.
- When a shape leaves an invalid operation tail, restore the checkpoint and recover the original operation as one `UnparsedCustomOperation`.
- Report `ShapeMismatch` with the registered mnemonic and shape through Rust and Python diagnostics.
- Preserve successful shaped operations and existing unregistered recovery.

## 2. Accept `to` type trailers (brief 29)

- Extend operand shapes that accept function-type trailers to accept `: T to T`.
- Expose operand and result types through the existing syntax and semantic accessors.
- Cover representative shaped casts and the applicable built-in registry surface.
- Confirm that no sibling operation named `to` is produced.

## 3. Accept literal attribute dictionaries (brief 30)

- Let `LITERAL_ATTRIBUTE` consume an optional attribute dictionary between the literal type and result type.
- Keep dictionary-free literals unchanged.
- Verify named attribute access, result types, trailing locations, and complete source ranges.

## 4. Make registry capabilities discoverable (brief 33)

- Expose the valid preset names through the Python API and type stubs.
- Define a small typed capture program for operands, literals, dictionaries, types, and fixed tokens.
- Validate format descriptions when the registry is built.
- Emit the existing CST node kinds from captured parts and lower their declared semantic roles.
- Keep `OperationShape` as the concise built-in vocabulary.

This part follows the safe fallback and shared grammar work in parts 1–3. The
literal form uses the acceptance spelling:
``$value `:` type($value) attr-dict `:` type($result)``.

## 5. Reduce CST construction memory (brief 34)

- Add a reproducible parse peak-memory case with per-phase allocation counts.
- Measure lexer tokens, parser events, compaction scratch space, and retained CST arrays separately.
- Remove avoidable duplicate buffers and excess capacity during compaction.
- Keep the compact, lossless CST available for syntax and hybrid retention.
- Re-run parse time and retained-byte measurements after every memory change.

Memory changes are accepted when they produce a clear peak reduction without a material parsing slowdown.

## Verification

Each part runs its Rust and Python acceptance cases before commit. The final pass runs the workspace tests, Python tests, formatting, lint, type checks, the production-shaped Python benchmark, and the parser allocation benchmark.
