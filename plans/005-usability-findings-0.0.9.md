# Zirium 0.0.9 usability findings

This plan addresses the open findings in the September 2026 evaluation archive. Each part is independently verified and committed before the next dependent part is integrated.

## 1. Recover from binary-operand arity underflow (D12 / brief 24)

- Reproduce the Python `PanicException` for zero and one operand.
- Remove the unchecked operand access in semantic lowering.
- Make underflow produce diagnostics while preserving the rest of the document.
- Cover the reported arity table at the Rust seam and through Python.

## 2. Accept function-type trailers for binary operands (D13 / brief 25)

- Parse both `: T` and `: T -> T` / `: T1, T2 -> T3` forms.
- Keep the existing result-type form unchanged.
- Diagnose mismatched function-type arity without panicking.

Depends on part 1 so malformed trailers cannot reintroduce an underflow panic.

## 3. Decode hexadecimal scalar float attributes (D11 / brief 23)

- Reuse the existing bit-pattern decoding semantics used by supported float payloads.
- Support f16, bf16, f32, and f64 scalar attributes when the literal width matches.
- Preserve decimal floats, hexadecimal integers, dense payloads, and diagnostics for width mismatches.

## 4. Make diagnostics consistently reportable (D14 / brief 26)

- Give lexer/parser diagnostics a non-empty human-readable message.
- Expose `kind`, `message`, and `range` on both parse and semantic diagnostics while retaining semantic `code`.
- Tighten recovery ranges only where a focused regression demonstrates an oversized construct range.
- Provide a documented offset-to-line/column helper rather than adding a source-manager abstraction.

## 5. Extend the operation-shape vocabulary (D15 / brief 27)

- Add unary, variadic/N-ary, and literal-attribute shapes to Rust, Python, stubs, and serialized config.
- Accept both result-type and function-type trailers where applicable.
- Verify the eight representative spellings expose operands, result types, and inline attributes through existing accessors.
- Leave unregistered-operation recovery unchanged.

Depends on parts 1 and 2 for shared operand/trailer parsing behavior.

## 6. Expose the package version

- Export `zirium.__version__` from installed package metadata.
- Type it in the public stub and cover the public surface with one focused Python assertion.

## Verification

For every part: run its focused regression tests, then the relevant Rust and Python suites. At the end, run the full workspace tests, formatting, lint, and type checks available in the repository. The proprietary 268 MiB archive is not present, so its exact corpus counts cannot be rerun locally; the synthetic acceptance spellings from the briefs are the executable substitute.
