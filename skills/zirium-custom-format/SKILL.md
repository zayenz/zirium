---
name: zirium-custom-format
description: Research MLIR dialects in the user's repository and create or refine custom registry formats for Zirium's Rust library, Python library, or CLI. Use only when the user explicitly invokes $zirium-custom-format in a repository that contains or uses custom MLIR dialects.
---

# Zirium custom formats

## Goal

Work in the repository where this skill was installed. Produce a small, working
Zirium registry configuration based on its dialect definitions and emitted MLIR.
Distinguish MLIR's custom assembly format from Zirium's `operation_formats` and
reusable operation shapes.

Do not require a Zirium source checkout or executable, inspect locally installed
Zirium files, install or upgrade Zirium, change Zirium itself, or copy files
into Zirium's repository unless the user explicitly changes the task's scope.

Zirium's format support is under active development. Before designing formats,
open the latest [custom-format documentation on
GitHub](https://github.com/zayenz/zirium/blob/main/docs/custom-formats.md).
Follow its links to current registry examples, APIs, or tests when useful. Use
the latest documentation as the design baseline rather than assumptions encoded
in this skill, but do not treat an omitted feature as a permanent exclusion.

## Start with the user

Inspect the repository to learn whether it uses Zirium through Rust, Python, the
CLI, or has not integrated it yet. Dependency manifests and existing code are
enough; do not inspect installed packages or executables.

Scan enough of the repository to ask concrete questions, then ask the user before choosing formats:

- Which dialects or operations matter first, and is the goal a representative subset or broad coverage?
- Where are representative `.mlir` files or commands that emit them, including important syntax variants?
- How will this repository use Zirium: Rust, Python, or CLI? What must work—parsing, semantic inspection, queries, editing, custom output, or round trips—and where should the registry live?

Ask follow-up questions when the source and examples disagree or an optional
form changes the appropriate registration. Do not silently choose a narrower
behavior than the user needs.

## Research the dialect

Search the whole repository, including generated build files when available.

- Find TableGen operation definitions and inspect `assemblyFormat`, `hasCustomAssemblyFormat`, traits, arguments, results, regions, successors, and attributes.
- For custom C++, find parser and printer implementations using terms such as `parse`, `print`, `OpAsmParser`, `OpAsmPrinter`, and the operation class name. Trace shared helper functions that materially affect syntax.
- Account for mixed definitions: TableGen may declare an operation while C++ implements its assembly, verification, or type inference.
- Search `*.mlir`, tests, examples, docs, FileCheck patterns, snapshots, and real compiler dumps. Look for more than one instance of each important operation so optional clauses and type spellings are visible.
- Record exact operation names and the observed order of operands, literals, attributes or properties, regions or successors, and type signatures. Note parser-only compatibility spellings and printer-canonical spellings separately.

Prefer actual checked-in definitions and representative emitted MLIR over
guesses from operation names. If examples are missing, ask the user for a sample
or for a safe command that produces one.

Useful searches include:

```sh
rg -n 'assemblyFormat|hasCustomAssemblyFormat|OpAsmParser|OpAsmPrinter|::parse\(|::print\(' PATH
rg -n --glob '*.mlir' --glob '*.td' 'DIALECT_PREFIX|OP_CLASS' PATH
```

## Choose a Zirium format

Use the latest GitHub documentation to map the dialect syntax to Zirium. If a
capability remains unclear, inspect the linked Zirium registry examples and
focused source or tests on GitHub. Do not inspect a local installation, and do
not assume Zirium accepts the full MLIR ODS `assemblyFormat` language.

For each operation, choose the smallest current mechanism that matches the required examples:

1. An existing preset or built-in implementation.
2. A reusable `operation_shapes` grammar.
3. A supported declarative `operation_formats` description.
4. A clearly reported compatibility gap that may require unknown-operation recovery, generic quoted MLIR, or a future Zirium version.

Do not force richer syntax into a similar-looking shape. Conversely, do not
declare a form unsupported merely because documentation omits it: inspect
current GitHub source or tests, or describe the uncertainty. Keep partial
coverage explicit, especially for regions, successors, optional groups, custom
directives, properties, and contextual type inference.

## Implement and check

Create or update the user-requested registry JSON in the current repository,
preserving unrelated entries. A typical starting point is:

```json
{"presets": [], "builtins": [], "operation_shapes": [], "operation_formats": []}
```

Validate through the repository's existing Zirium integration when one is already usable:

- Load the registry through the Rust, Python, or CLI path the repository uses.
- Parse representative examples and inspect diagnostics or errors.
- Confirm required operands, results, attributes, regions, or symbols through the relevant library API or CLI query.
- Test editing, custom output, or round trips only when they are part of the requested behavior.

Do not install Zirium merely to validate the result. If no usable integration
exists, check the JSON syntax and review it against the latest documentation,
then give the user a focused Rust, Python, or CLI example to run later.

Use a few high-value examples rather than manufacturing a large test suite. When
a form fails, reduce it to the smallest informative snippet, compare it with the
dialect definition and current GitHub documentation, and decide whether the
registry is wrong or Zirium currently has a compatibility gap.

## Report the result

Report the created file, Zirium integration path, operations covered,
representative checks run, and any syntax variants left unresolved. Phrase
limitations as gaps or uncertainties observed against the latest documentation,
not permanent boundaries, and ask the user whether to extend coverage when
meaningful cases remain.
