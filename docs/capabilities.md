# Capability reference

This reference describes Zirium 0.1.0. Its textual syntax target is MLIR
22.1; bundled presets were checked against LLVM 22.1.0, except StableHLO,
which was checked against StableHLO 1.20.1. Preset names and coverage track
Zirium releases, not every form in an upstream release.

Generic quoted operations need no registry. A registered operation has a known
shape or custom grammar; registration does **not** mean that every assembly
form, optional clause, verifier, or printer for that operation is implemented.
Check the exact generated preset definition in the
[preset reference](registry-presets.md) and treat recovery diagnostics as
incomplete semantic information.

## Choose an interface

| Need | Rust library | Python package | CLI |
| --- | --- | --- | --- |
| Parse losslessly | Yes: `ParsedFile`, including invalid UTF-8 and recovered syntax. | Yes: `parse_bytes`, `parse_text`, and `parse_file`; use bytes for invalid UTF-8. | Yes, as the input stage of a query or edit; no CST API. |
| Verify | `validate_structure` checks document invariants; `verify_semantics(&registry)` also runs registered schemas and verifiers. | `validate_structure()` and `verify_semantics()`; the parsed file retains its registry. | No verification mode. `--strict` rejects incomplete lowering and unknown reachable references; it is not full dialect verification. |
| Query | Structured query builders and the parsed query language, with explicit registry overrides. | Typed structured query builders through `Document.query()`. | Full query-language interface for reports, selections, and dependency traversal. |
| Construct IR | Regionless `OperationSpec` insertion into a root or existing block; types and attributes may use arena-independent specs. | Regionless `OperationSpec` insertion; identity-bearing wrappers must belong to the document. | No operation, block, region, or signature construction. |
| Mutate | Transactional attribute edits, result-type replacement, successor-argument rewiring, and restricted insertion/erasure. | The same core edits through buffered `document.edit()` commands, with narrower construction inputs. | Only `set_attr` and `remove_attr` on selected operations. |
| Produce output | Original bytes, deterministic generic canonical output, custom printing with a registry, and eligible source-preserving output. | Original, canonical, built-in custom, and eligible source-preserving bytes/files. | Selected MLIR, complete edited MLIR, text, Markdown, JSON, or JSONL; stdout is buffered until success. |
| Supply dialect callbacks | Yes: static descriptors can provide parsing, lowering, verification, and printing callbacks. | No callback registration; load declarative registries and bundled presets. | No callback registration; load declarative registries and bundled presets. |
| Load registries and bundles | Presets, JSON text, one or more files, and file-relative import bundles. The caller passes the same registry at each relevant stage. | `DialectRegistry.from_name`, `.from_config`, and `.from_file`; parsed files retain the registry. | Repeatable `--preset` and `--registry`; imported paths resolve relative to the declaring file. |
| Bound untrusted work | All parse limits, query work/item limits, and registry-bundle load limits are configurable. | All parse and query limits, plus registry-bundle load limits on `from_file`. | File bytes, query work/items, and registry depth/files/edges/bytes are configurable; other parse limits use defaults. |

Use Rust when you need callbacks, the complete editing surface, or precise
control of registries and retention. Use Python for typed inspection and
transactional edits without writing Rust. Use the CLI for shell pipelines,
reports, dependency selections, or whole-document attribute edits. See the
[Rust and Python walkthrough](getting-started.md), [query builder guide](query-dsl.md),
and [CLI examples](cli-examples.md) for executable examples.

## Syntax and workflow limits

- Generic operation syntax is the broadest syntax path. Unknown dialect types
  and attributes retain balanced payloads as opaque values.
- Presets provide selected structural parsing and lowering. The linked JSON
  files in the [preset reference](registry-presets.md) are the release's exact
  operation/form inventory. Unsupported variants may recover instead of
  exposing their full semantic structure.
- Strict lowering and structural validation are distinct from semantic
  verification. Zirium does not provide full dialect verification, type
  inference, target validation, or execution.
- Bytecode, ODS/TableGen loading, VHLO and StableHLO portable artifacts are not
  supported. Presets do not promise every upstream operation or assembly form.
- Queries preserve source order and duplicates unless a stage says otherwise.
  Traversal of calls and references uses only registered structural semantics.
- Structural edits require a complete semantic document and are not general IR
  construction. Insertion and erasure discard hybrid source retention; consult
  the [editing matrix](structural-editing.md) before choosing an output path.
- Canonical output is deterministic generic MLIR, not the original spelling.
  Original output is byte-for-byte; preserving output is available only while
  the hybrid-retention contract remains satisfied.
- Limits are resource guards, not timeouts. Exact defaults and failure behavior
  are documented under [resource limits](compatibility.md#resource-limits) and
  [query limits](query-dsl.md#limits-and-errors).

For caller-defined syntax and the registry contract at every Rust stage, see
[custom formats](custom-formats.md#registry-contract-for-each-rust-stage). For
the compatibility corpus and its release tie, see the
[MLIR 22.1 corpus notes](../tests/corpus/mlir-22.1/README.md).
