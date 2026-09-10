# Zirium

Zirium is a Rust library and command-line tool for reading, inspecting, editing,
and writing textual MLIR, with typed Python bindings. It works without linking
LLVM. The parser preserves the original bytes, including comments, whitespace,
malformed syntax, and invalid UTF-8. A separate semantic representation supports
verification and structural edits.

Version 0.0.12 is experimental. It targets MLIR 22.1 textual syntax and supports
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

Rust builds require Rust 1.88 or newer. From a source checkout, install the CLI
with:

```sh
cargo install --path crates/zirium
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

A `File` owns the original bytes, tokens, concrete syntax tree (CST), and syntax
diagnostics. A `Document` owns resolved semantic data. Keeping them separate lets
you inspect malformed source without requiring a valid semantic model, and edit
semantic structure without manipulating syntax nodes.

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

The baseline registry covers a small set of Builtin, Func, Arith, and CF
operations. Bundled presets expose selected custom forms from other dialects
for structural queries. Unsupported forms use best-effort recovery; presets do
not implement full dialect verification or execution.

The [custom-format guide](docs/custom-formats.md) explains registry configuration,
operation shapes, and use from Rust, Python, and the CLI. The
[preset reference](docs/registry-presets.md) groups the available dialects and
links to their exact registry definitions.

The [corpus notes](https://github.com/zayenz/zirium/blob/main/tests/corpus/mlir-22.1/README.md)
describe how syntax compatibility is checked against `llvmorg-22.1.0`.

## Repository layout

```text
crates/zirium/          Core Rust library
crates/zirium-python/   PyO3 extension module
python/zirium/          Python package and type declarations
python/tests/           Python API tests
tests/corpus/           Versioned MLIR compatibility corpus
docs/                   Usage, compatibility, and architecture notes
fuzz/                   Lexer and parser fuzz targets
```

The architecture notes include the [syntax representation baseline](https://github.com/zayenz/zirium/blob/main/docs/architecture/representation-baseline.md)
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

Release notes are recorded in the
[changelog](https://github.com/zayenz/zirium/blob/main/CHANGELOG.md).
