# Zirium

Zirium 0.0.2 is an experimental release. The API may change without a
migration path.

Zirium is a Rust library for reading, inspecting, transforming, and writing textual MLIR. It also provides typed Python bindings. The parser keeps the original bytes, including comments, whitespace, malformed syntax, and invalid UTF-8. A separate semantic layer provides a compact representation for verification and editing.

The project is useful when a tool needs to understand MLIR without linking LLVM, and when source fidelity matters alongside semantic processing.

Zirium currently targets the textual syntax of MLIR 22.1. It has a
deliberately narrow dialect surface and is not a replacement for all of MLIR's
parser or ODS infrastructure.

The current release is intended for experimentation, tooling prototypes, and
evaluation of the lossless syntax/semantic split. It does not promise broad
dialect coverage, bytecode support, ODS/TableGen loading, or a stable API.

## What it provides

- A byte-oriented lexer and lossless concrete syntax tree (CST).
- Recovery from malformed input with ranged diagnostics and useful outer structure.
- Standard `module` handling and caller-provided func-like and call-like operation shapes.
- Diagnostic-bearing recovery of unknown custom operations, including nested regions.
- Strict and best-effort lowering into a separate semantic document.
- Operations, regions, blocks, SSA values, types, attributes, locations, symbols, and dominance queries.
- Python inspection of exact type and attribute spellings, scalar attributes, and stable value identity.
- Deterministic generic MLIR output and structural round-trip comparison.
- Buffered semantic edits that commit atomically.
- Conservative source-preserving output for documents lowered with hybrid retention.
- Rust and Python APIs over the same core implementation.
- Explicit limits for file size, token count, nesting, and large payloads.

## A small example

```python
import zirium

source = '''\
"builtin.module"() ({
  %value = "example.make"() : () -> i32
  "example.consume"(%value) : (i32) -> ()
}) : () -> ()
'''

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

Parsing and lowering are separate on purpose. A `File` owns the original bytes, tokens, CST, and syntax diagnostics. A `Document` owns resolved semantic data. This keeps malformed-source tooling from depending on a partly valid semantic model, while transformation code does not have to operate on syntax nodes.

## Getting started

Zirium requires Rust 1.85 or newer. The Python package supports conventional
CPython 3.11 through 3.14. Release artifacts are currently proven only for
Linux x86_64 and macOS arm64 by CI; other platforms require a local source
build with maturin and are not part of the platform promise. Wheels remain
specific to each supported CPython version rather than using the stable ABI.

The [getting-started guide](https://github.com/zayenz/zirium/blob/main/docs/getting-started.md) covers:

- building and testing the Rust workspace;
- installing the Python extension into a virtual environment;
- parsing, lowering, verifying, and writing MLIR;
- choosing a retention profile and output mode.

The [CLI examples](https://github.com/zayenz/zirium/blob/main/docs/cli-examples.md)
show how to query and edit small MLIR files with the `zirium` command.

For the shortest check from a fresh checkout:

```sh
cargo test --workspace
```

Install a published Python wheel with:

~~~sh
python -m pip install zirium
~~~

If a wheel is not available yet, build the extension locally using the
[getting-started guide](https://github.com/zayenz/zirium/blob/main/docs/getting-started.md).

## Output modes

Zirium has three output paths, each with a different contract:

| Output | Source object | Contract |
| --- | --- | --- |
| Original | Parsed file | Reproduces the input bytes exactly. |
| Canonical | Semantic document | Emits deterministic generic MLIR from semantic storage. |
| Preserving | Hybrid semantic document | Copies unchanged source and regenerates edited operations or blocks. |

Canonical output intentionally does not preserve comments, whitespace, aliases, or SSA spelling. Use original output when no semantic edit is needed, and hybrid retention when edits should leave unrelated source text alone.

## Dialect support

Generic quoted operations are handled without a dialect registry. Unknown dialect types and attributes keep their balanced bodies as opaque values.

Registered custom syntax is currently a proving surface rather than broad MLIR dialect coverage. The built-in proving registry covers a fixed subset of Builtin, Func, Arith, and CF operations. A declarative registry can select from that same fixed set. Zirium does not load LLVM dialect definitions, interpret arbitrary ODS/TableGen files, or run Python callbacks while parsing.

See [compatibility and local wheel checks](https://github.com/zayenz/zirium/blob/main/docs/compatibility.md) for the exact toolchain and CPython matrix. The [corpus notes](https://github.com/zayenz/zirium/blob/main/tests/corpus/mlir-22.1/README.md) explain how syntax compatibility is tied to `llvmorg-22.1.0`.

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

The architecture notes include the [syntax representation baseline](https://github.com/zayenz/zirium/blob/main/docs/architecture/representation-baseline.md) and [processing benchmarks](https://github.com/zayenz/zirium/blob/main/docs/architecture/processing-benchmarks.md).

## Development checks

The complete local quality gate, including the supported Rust and Python
versions, is documented in the [compatibility guide](https://github.com/zayenz/zirium/blob/main/docs/compatibility.md).
The CI workflow runs those same commands.

Python development details are documented in [the getting-started guide](https://github.com/zayenz/zirium/blob/main/docs/getting-started.md).
The [release guide](https://github.com/zayenz/zirium/blob/main/docs/releasing.md)
records the registry setup, package checks, and publication order.

## License

Zirium's original code is available under the [MIT](https://github.com/zayenz/zirium/blob/main/LICENSE-MIT) or
[Apache 2.0](https://github.com/zayenz/zirium/blob/main/LICENSE-APACHE) license. Some MLIR-derived or adapted test
material retains the provenance and licensing information recorded in the
[corpus manifest](https://github.com/zayenz/zirium/blob/main/tests/corpus/mlir-22.1/manifest.toml).

Release notes are recorded in the [changelog](https://github.com/zayenz/zirium/blob/main/CHANGELOG.md).
