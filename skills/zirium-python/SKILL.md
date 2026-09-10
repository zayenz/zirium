---
name: zirium-python
description: Use Zirium's typed Python library to parse, inspect, verify, edit, and write textual MLIR. Use when Python code imports `zirium`, or a task mentions `parse_text`, `parse_bytes`, `parse_file`, `DialectRegistry`, semantic documents, or Zirium's Python API.
---

# Zirium Python

## Goal

Use the installed Zirium Python API without losing source bytes or confusing
syntax inspection with semantic processing. Follow the dependency version
selected by the project and make registry, lowering, retention, and output
choices explicit when they matter.

Zirium evolves quickly. Inspect `pyproject.toml`, the lockfile, or
`importlib.metadata.version("zirium")`, then consult matching documentation. In
a Zirium source checkout, `python/zirium/__init__.pyi` is the compact typed API
reference and `docs/getting-started.md` explains the main contracts. Do not
install or upgrade the package unless asked.

## Start with the normal flow

```python
import zirium

parsed = zirium.parse_text(source)
result = parsed.lower_strict("semantic")
if result.document is None:
    for diagnostic in result.diagnostics:
        line, column = parsed.line_column(diagnostic.range[0])
        print(line, column, diagnostic.kind, diagnostic.message)
    raise ValueError("semantic lowering failed")

document = result.document
document.validate_structure()
output = document.canonical_bytes()
```

Use `parse_text` for `str`, `parse_bytes` for arbitrary bytes, and `parse_file`
for paths. A parsed `File` owns the original bytes, CST, operations, and syntax
diagnostics. Lower only when the task needs resolved SSA values, types, symbols,
semantic verification, analysis, or edits.

## Make the important choices explicitly

- Pass a suitable `DialectRegistry` while parsing registered custom syntax;
  the parsed file and document retain it for later verification and printing.
- Prefer `lower_strict()` when incomplete semantics would make a result
  misleading. Use `lower_best_effort()` only when the caller can handle its
  diagnostics and incomplete document.
- Use `"semantic"` for analysis and canonical output, `"syntax"` when CST
  access must survive lowering, and `"hybrid"` only for edits followed by
  source-preserving output.
- Choose `original_bytes()` or `write_original()` for byte-for-byte source,
  canonical output for deterministic generic MLIR, and preserving output for
  supported edits on a hybrid document. Canonical output may normalize text.
- Apply edits with `with document.edit() as edit:` so they commit atomically.
  Treat semantic wrappers as document-owned handles; erased handles become
  stale, and handles from another document are foreign.
- Use packed syntax or operation tables only for bulk inspection. Prefer normal
  wrappers for small traversals because they are clearer and checked.

For caller-defined operation syntax, use the `zirium-custom-format` skill. For
query-language or command-line work, use `zirium-cli` instead.

## Verify proportionately

Run the project's existing focused Python test or example. In a Zirium source
checkout, use the locked environment and a narrow `uv run --locked --no-sync
pytest ...` invocation. Avoid inventing broad tests for a small integration.
Report diagnostics and note the registry, retention mode, and output contract
when they affect the result.
