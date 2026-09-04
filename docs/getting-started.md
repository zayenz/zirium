# Getting started with Zirium

This guide takes a source checkout through the two main Zirium workflows: using the Rust crate and building the Python extension. Both APIs follow the same sequence:

```text
source bytes -> parsed file -> semantic document -> verification/editing -> output
```

Keep the parsed file when exact source text matters. Lower to a semantic document when the task needs resolved SSA values, types, symbols, verification, or structural edits.

## Prerequisites

The Rust workspace requires:

- Rust 1.85 or newer;
- Cargo.

Python development also requires:

- CPython 3.11, 3.12, 3.13, or 3.14;
- [`uv`](https://docs.astral.sh/uv/), used below to create the environment;
- a working Rust toolchain so maturin can compile the extension.

CPython 3.14 free-threaded builds are experimental. Zirium builds version-specific extensions and does not use `abi3`.

## Check the source tree

Clone the repository, then run the workspace tests:

```sh
git clone https://github.com/zayenz/zirium.git
cd zirium
cargo test --workspace
```

If the checkout already exists, run the same test command from its root:

```sh
cargo test --workspace
```

This builds both workspace crates and runs the Rust tests. To generate and open the core API documentation:

```sh
cargo doc -p zirium --open
```

## Use Zirium from Rust

For a local consumer next to the Zirium checkout, point Cargo at the core crate:

```toml
[dependencies]
zirium = { path = "../zirium/crates/zirium" }
```

The following program parses generic MLIR, checks for syntax diagnostics, lowers it strictly, validates the semantic structure, and prints canonical generic MLIR:

```rust
use zirium::{
    dialect::DialectRegistry,
    parser::ParsedFile,
    printer::PrintLayout,
    semantic::{
        LoweringMode, RetentionProfile,
        lower_with_dialect_registry_and_retention,
    },
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let source = br#""builtin.module"() ({
  %value = "example.make"() : () -> i32
  "example.consume"(%value) : (i32) -> ()
}) : () -> ()
"#;

    let parsed = ParsedFile::parse(source.as_slice())?;

    for diagnostic in parsed.lexer_diagnostics() {
        eprintln!("lexer diagnostic: {diagnostic:?}");
    }
    for diagnostic in parsed.syntax().diagnostics() {
        eprintln!("syntax diagnostic: {diagnostic:?}");
    }

    let lowered = lower_with_dialect_registry_and_retention(
        &parsed,
        LoweringMode::Strict,
        RetentionProfile::SemanticOnly,
        &DialectRegistry::EMPTY,
    );

    for diagnostic in &lowered.diagnostics {
        eprintln!("{}: {}", diagnostic.range, diagnostic.message);
    }

    let document = lowered
        .document
        .ok_or("strict semantic lowering failed")?;
    document.validate_structure()?;

    let output = document.canonical_bytes(PrintLayout::Pretty)?;
    print!("{}", String::from_utf8(output)?);
    Ok(())
}
```

`ParsedFile::parse` uses the empty registry, which accepts generic quoted operation syntax. Use `ParsedFile::parse_with_registry` and pass the same registry to lowering and semantic verification when the input contains registered custom syntax.

## Build the Python package

Create a virtual environment and install the local extension:

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python maturin pytest ruff ty
VIRTUAL_ENV=.venv .venv/bin/maturin develop --uv
```

Replace `3.13` with another supported CPython version when needed. Confirm that the extension imports:

```sh
.venv/bin/python -c 'import zirium; print("zirium imported")'
```

For a useful smoke test instead of an import-only check:

```sh
.venv/bin/python - <<'PY'
import zirium

parsed = zirium.parse_text('"test"() : () -> ()')
assert not parsed.diagnostics
assert parsed.operation_count == 1
assert parsed.operation(0).text() == b'"test"() : () -> ()'
print("parsed one operation")
PY
```

Format the Python code with:

```sh
.venv/bin/ruff format python
```

Run the checks with:

```sh
.venv/bin/ruff format --check python
.venv/bin/ruff check python
.venv/bin/ty check python
.venv/bin/pytest
```

## Parse syntax without lowering

Syntax parsing is lossless even when the input is malformed. Diagnostics describe the problem, while the returned `File` still owns the original bytes and recoverable CST:

```python
import zirium

source = b'"broken"(  // unfinished operation\n'
parsed = zirium.parse_bytes(source)

assert parsed.original_bytes() == source

for diagnostic in parsed.diagnostics:
    print(diagnostic.kind, diagnostic.range)

for index in range(parsed.operation_count):
    operation = parsed.operation(index)
    print(operation.range, operation.has_error, operation.text())
```

For bulk syntax inspection, build one immutable packed snapshot and cast its
native-endian columns with `memoryview`:

```python
table = parsed.syntax_table()
node_kinds = memoryview(table.node_kind).cast("H")
node_starts = memoryview(table.node_start).cast("I")
subtree_ends = memoryview(table.node_subtree_end).cast("I")

for index in range(parsed.node_count):
    kind = table.node_kind_name(node_kinds[index])
    start = node_starts[index]
    end = subtree_ends[index]
    print(index, kind, None if start == 2**32 - 1 else start, end)

root = parsed.node(0)
children = memoryview(root.child_indices()).cast("I")
first_child = parsed.node(children[0])
```

The `H` and `I` casts match the native-endian `u16` and `u32` columns. Codes
and indexes are snapshot-local and belong only to the source `File`; use
`node_kind_code`/`node_kind_name` and `token_kind_code`/`token_kind_name` for
checked scalar conversion. `File.node()` and `File.token()` create wrappers
only when requested.

Use `parse_text` for Python strings, `parse_bytes` for arbitrary bytes, and `parse_file` for a path. `parse_text` encodes its input as UTF-8. The other two paths preserve raw bytes, including invalid UTF-8.

## Lower and verify semantic data

Strict lowering returns no document when semantic resolution is incomplete. Best-effort lowering can return a structurally valid document containing invalid sentinels and diagnostics. That document is useful for inspection, but verification, editing, and canonical output reject it until it is complete.

```python
import zirium

parsed = zirium.parse_text('''\
%value = "example.make"() : () -> i32
"example.consume"(%value) : (i32) -> ()
''')

result = parsed.lower_strict("semantic")
if result.document is None:
    for diagnostic in result.diagnostics:
        print(diagnostic.range, diagnostic.message)
    raise SystemExit(1)

document = result.document
document.validate_structure()

table = document.operation_table()
for index in range(table.count):
    operation = table.operation(index)  # wrappers are created only on demand
    print(operation.name, operation.source_range)
```

`OperationTable` is a frozen, self-contained snapshot. Its `name_code`,
`source_start`, and `source_end` columns are native-endian u32 bytes;
`root_flags` is one byte per row (bit 0 denotes a document root); and
`name_offsets` is a native-endian u32 offset table into `name_bytes`.
`0xffffffff` in both source columns denotes an absent range. Dense name codes
follow first encounter order. The packed columns remain valid after edits;
`operation(index)` checks the retained generation-checked ID and raises
`StaleHandleError` if that operation was erased.

`validate_structure()` checks ownership, IDs, parent-child links, and operand targets. `verify_semantics()` also runs the schemas and verifiers in the registry associated with the parsed file.

For the fixed Builtin, Func, Arith, and CF proving subset, construct a registry before parsing:

```python
import zirium

registry = zirium.DialectRegistry.declarative([
    "builtin.module",
    "func.func",
    "func.return",
])

parsed = zirium.parse_text(
    "builtin.module { func.func @main() { func.return } }",
    registry=registry,
)
result = parsed.lower_strict()
assert result.document is not None, result.diagnostics
result.document.verify_semantics()
print(result.document.custom_bytes().decode(), end="")
```

The registry is retained by the parsed file and semantic document, so it does not need to remain in a separate Python variable.

## Choose a retention profile

Lowering accepts three retention profiles:

| Python value | Rust value | Retained data | Use it for |
| --- | --- | --- | --- |
| `"semantic"` | `SemanticOnly` | Semantic storage | Analysis, canonical output, and lower retained memory. |
| `"syntax"` | `SyntaxOnly` | Semantic storage, source, and CST, without mappings | Semantic inspection that also needs the parsed syntax. |
| `"hybrid"` | `Hybrid` | Semantic storage, source, CST, and sparse mappings | Semantic edits followed by source-preserving output. |

The default is semantic-only. Hybrid retention costs more memory because the document keeps both representations.

## Write output

Original output comes from the parsed file and is byte-for-byte identical to its input:

```python
parsed.write_original("unchanged.mlir")
```

Canonical output comes entirely from semantic storage:

```python
document.write_canonical("canonical.mlir")
```

Canonical output normalizes formatting and names. It is intended for deterministic semantic output, not source fidelity.

Preserving output requires hybrid retention:

```python
result = parsed.lower_strict("hybrid")
assert result.document is not None, result.diagnostics
document = result.document

with document.edit() as edit:
    edit.compact_pools()

document.write_preserving("preserved.mlir")
```

An edit context buffers commands and commits them atomically when the context exits. If validation fails or the body raises an exception, the original document remains unchanged. Unedited source ranges are copied directly; dirty operations or blocks are regenerated.

## Resource limits

The Python parse functions accept keyword-only limits:

```python
parsed = zirium.parse_file(
    "input.mlir",
    max_file_bytes=64 * 1024 * 1024,
    max_tokens=4_000_000,
    max_delimiter_depth=256,
    max_payload_bytes=16 * 1024 * 1024,
    max_numeric_literal_bytes=4096,
    max_attribute_depth=64,
    max_alias_expansion_depth=64,
)
```

Exceeding `max_file_bytes` raises `ResourceLimitError` before lexing. Other syntax limits return a lossless parsed file with diagnostics where recovery is possible. Rust callers configure the same boundaries with `ParseLimits` and `ParsedFile::parse_with_limits`.
Alias expansion uses a shared budget across type, attribute, affine, memref, and
location aliases. When an alias chain reaches `max_alias_expansion_depth`,
lowering reports `alias expansion depth exceeds limit of 64`, with the selected
limit in place of 64. The default is 64.

## Where to go next

- [Compatibility and local wheel checks](compatibility.md) contains the full Rust quality gate and CPython test matrix.
- [Syntax representation baseline](architecture/representation-baseline.md) explains the flat immutable CST.
- [Processing benchmarks](architecture/processing-benchmarks.md) records the large-file measurement protocol and results.
- [`python/zirium/__init__.pyi`](../python/zirium/__init__.pyi) is the compact reference for the typed Python API.
- [`crates/zirium/src/lib.rs`](../crates/zirium/src/lib.rs) links the public Rust modules and contains a minimal doctest.
