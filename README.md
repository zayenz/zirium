# Zirium

Zirium is a Rust library and command-line tool for reading, inspecting, editing,
and writing textual MLIR, with typed Python bindings. It works without linking
LLVM. The parser preserves the original bytes, including comments, whitespace,
malformed syntax, and invalid UTF-8. A separate semantic representation supports
verification and structural edits.

Version 0.1.0 is experimental. It targets MLIR 22.1 textual syntax and supports
selected custom dialect forms. The API may change without a migration path;
bytecode and ODS/TableGen loading are unsupported.

## Installation

Install the Python package with:

```sh
python -m pip install zirium
```

Published wheels cover CPython 3.11 through 3.14 on Linux x86_64 and macOS
arm64. Wheels are specific to each CPython version. Other platforms require a
local source build and are unsupported.

### CLI binaries

CLI archives are produced by the release workflow for Linux x86_64 (static
musl) and macOS arm64 (macOS 11 or newer). They require neither Rust nor Python.
Binary distribution starts with 0.1.0.

Download the matching `zirium-VERSION-TARGET.tar.gz` archive and its `.sha256`
file from [GitHub Releases](https://github.com/zayenz/zirium/releases):

| Platform | Target |
| --- | --- |
| Linux x86_64, kernel 3.2 or newer | `x86_64-unknown-linux-musl` |
| macOS arm64, macOS 11 or newer | `aarch64-apple-darwin` |

In the download directory, substitute the version and target in these commands:

```sh
archive=zirium-VERSION-TARGET.tar.gz
shasum -a 256 -c "$archive.sha256"
tar -xzf "$archive"
mkdir -p "$HOME/.local/bin"
install -m 755 "${archive%.tar.gz}/zirium" "$HOME/.local/bin/zirium"
"$HOME/.local/bin/zirium" --version
```

Add `$HOME/.local/bin` to your `PATH` if it is not already there. On Linux,
`sha256sum -c "$archive.sha256"` can also verify the checksum.
The archive naming follows `cargo-binstall` conventions, so users with
[cargo-binstall](https://github.com/cargo-bins/cargo-binstall) can install a
release that provides binaries with `cargo binstall zirium`.

### CLI from source

Rust builds require Rust 1.88 or newer. Install the published CLI with:

```sh
cargo install zirium --locked
```

From a source checkout, use:

```sh
cargo install --path crates/zirium --locked
```

The Python wheel does not include the CLI. The
[getting-started guide](https://github.com/zayenz/zirium/blob/main/docs/getting-started.md)
covers Rust usage and building the Python extension locally.

## Command-line utility

The `zirium` binary queries and edits textual MLIR. Pass the query first,
followed by any input files. With no input files, it reads from standard input.
For example, this selects every `arith.addi` operation in `input.mlir`:

```sh
zirium 'filter(op("arith.addi"))' input.mlir
```

Use `zirium --help` for CLI options and `--list-presets` for dialect coverage.
For semantic queries on StableHLO, pass `--preset stablehlo --strict` to reject
unsupported custom forms rather than continuing with incomplete information.

To count each operation type within each function, save this as a query file
and run it with `zirium --preset stablehlo -f counts.zirium model.mlir`:

```zirium
functions = filter(op("func.func"));
functions | map_by(attr("sym_name"), children | subtree | names | tally) | json
```

Add `reachable |` before `names` to include supported referenced bodies,
counting each reachable operation once.

The [query language reference](https://github.com/zayenz/zirium/blob/main/docs/query-language.md)
lists every predicate and pipeline stage. The
[CLI examples](https://github.com/zayenz/zirium/blob/main/docs/cli-examples.md)
show complete queries, dependency slices, and edits.

You can also run the binary from a source checkout with Cargo:

```sh
cargo run --quiet --bin zirium -- \
  'filter(op("arith.addi"))' input.mlir
```

## Python example

```python
import zirium

source = """\
"builtin.module"() ({
  %value = "example.make"() : () -> i32
  "example.consume"(%value) : (i32) -> ()
}) : () -> ()
"""

parsed = zirium.parse_text(source)
assert parsed.original_bytes() == source.encode()

lowered = parsed.lower_strict("semantic")
if lowered.document is None:
    for diagnostic in lowered.diagnostics:
        print(diagnostic.range, diagnostic.message)
    raise SystemExit(1)

document = lowered.document
document.validate_structure()
table = document.operation_table()
print([table.operation(index).name for index in range(table.count)])
print(document.canonical_bytes().decode(), end="")
```

A `File` owns the original bytes, tokens, concrete syntax tree (CST), and
diagnostics. A `Document` owns resolved semantic data. Use the file to inspect
malformed source; use the document for semantic queries and structural edits.

## Structured queries

Rust and Python can build reusable queries without query source strings:

```python
from zirium.query import ops, op

consumers = ops().filter(op("example.make")).users().unique()
selected = document.query(consumers)
counts = document.query(consumers.names().tally())
```

Rust uses the same builders: `document.query(&consumers)?`. Queries return native
operation handles, strings, counts, and maps. The [structured query guide](https://github.com/zayenz/zirium/blob/main/docs/query-dsl.md)
covers predicates, nested queries, semantic projections, and evaluation limits.

## Output modes

Zirium has three output paths, each with a different contract:

| Output | Source object | Contract |
| --- | --- | --- |
| Original | Parsed file | Reproduces the input bytes exactly. |
| Canonical | Semantic document | Emits deterministic generic MLIR from semantic storage. |
| Preserving | Hybrid semantic document | Copies unchanged source and regenerates edited operations or blocks. |

Canonical output normalizes formatting and SSA names and does not preserve
comments or aliases. Use original output to reproduce the input, or hybrid
retention when edits should preserve unrelated source text.

## Dialect support

Generic quoted operations are handled without a dialect registry. Unknown
dialect types and attributes keep their balanced bodies as opaque values.

The baseline registry covers selected Builtin, Func, Arith, and CF operations.
Bundled presets add custom forms from other dialects. Unsupported forms use
best-effort recovery. Presets support structural queries, without full dialect
verification or execution.

The [custom-format guide](https://github.com/zayenz/zirium/blob/main/docs/custom-formats.md) explains registry configuration,
operation shapes, and use from Rust, Python, and the CLI. The
[preset reference](https://github.com/zayenz/zirium/blob/main/docs/registry-presets.md) groups the available dialects and
links to their exact registry definitions.

The [corpus notes](https://github.com/zayenz/zirium/blob/main/tests/corpus/mlir-22.1/README.md)
describe how syntax compatibility is checked against `llvmorg-22.1.0`.

## Agent skills

The repository includes optional skills for agents working with Zirium:

- [CLI](https://github.com/zayenz/zirium/blob/main/skills/zirium-cli/SKILL.md): queries, reports, dependency slices, and complete-document edits.
- [Python](https://github.com/zayenz/zirium/blob/main/skills/zirium-python/SKILL.md): typed API usage, diagnostics, retention, and transactions.
- [Rust](https://github.com/zayenz/zirium/blob/main/skills/zirium-rust/SKILL.md): library integration and document ownership.
- [Custom formats](https://github.com/zayenz/zirium/blob/main/skills/zirium-custom-format/SKILL.md): research a project's dialect and build a registry; explicitly invoked as `$zirium-custom-format`.

Install each skill folder independently through your agent's skill mechanism.
Skills are distributed in the repository, outside the Python wheel and Rust crate.

## Repository layout

```text
crates/zirium/          Core Rust library
crates/zirium-python/   PyO3 extension module
python/zirium/          Python package and type declarations
python/tests/           Python API tests
tests/corpus/           Versioned MLIR compatibility corpus
docs/                   Usage, compatibility, and architecture notes
skills/                 Agent guidance for CLI, libraries, and custom formats
fuzz/                   Lexer and parser fuzz targets
```

The architecture notes describe the [syntax representation](https://github.com/zayenz/zirium/blob/main/docs/architecture/syntax-representation.md)
and [processing benchmarks](https://github.com/zayenz/zirium/blob/main/docs/architecture/processing-benchmarks.md).

## Development

Run the Rust workspace tests from the repository root:

```sh
cargo test --workspace
```

The [compatibility guide](https://github.com/zayenz/zirium/blob/main/docs/compatibility.md)
lists the full CI checks, supported toolchains, and wheel checks. The
[getting-started guide](https://github.com/zayenz/zirium/blob/main/docs/getting-started.md)
covers Python development, and the
[release guide](https://github.com/zayenz/zirium/blob/main/docs/releasing.md)
describes packaging and publication.

## License

Zirium's original code is available under the
[MIT](https://github.com/zayenz/zirium/blob/main/LICENSE-MIT) or [Apache 2.0](https://github.com/zayenz/zirium/blob/main/LICENSE-APACHE) license. Some
MLIR-derived or adapted test material carries the source attribution and license
notices recorded in the [corpus manifest](https://github.com/zayenz/zirium/blob/main/tests/corpus/mlir-22.1/manifest.toml).

Release history is recorded in the
[changelog](https://github.com/zayenz/zirium/blob/main/CHANGELOG.md).
