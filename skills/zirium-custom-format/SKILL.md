---
name: zirium-custom-format
description: Research MLIR dialects in the user's repository and create or refine registry formats for an installed Zirium binary. Use only when the user explicitly invokes $zirium-custom-format in a repository that contains or uses custom MLIR dialects.
---

# Zirium Custom Format

## Goal

Work in the repository where this skill was installed. Produce a small, working Zirium registry configuration grounded in that repository's dialect definitions and emitted MLIR. Distinguish MLIR's custom assembly format from Zirium's `operation_formats` and reusable operation shapes.

Use the installed `zirium` executable as the capability boundary. Do not require a Zirium source checkout, change Zirium itself, install or upgrade Zirium, or copy files into Zirium's repository unless the user explicitly changes the task's scope.

Zirium's format support is under active development. Treat behavior observed with the installed version as authoritative for this task; use documentation as guidance, not as a permanent list of exclusions.

## Start with the user

Run `zirium --version` and inspect `zirium --help` before designing the registry. If the executable is missing or its location is unclear, ask the user rather than installing it.

Do a light scan first when it will make the questions concrete, then ask the user before choosing formats:

- Which dialects or operations matter first, and is the goal a representative subset or broad coverage?
- Where are representative `.mlir` files or commands that emit them, including important syntax variants?
- What must work with the binary: parsing, queries, editing, selected-fragment output, or round trips? Where should the registry JSON live in this repository?

Ask follow-up questions when the source and examples disagree or an optional form changes the appropriate registration. Do not silently choose a narrower behavior than the user needs.

## Research the dialect

Search the whole repository, including generated build files when available.

- Find TableGen operation definitions and inspect `assemblyFormat`, `hasCustomAssemblyFormat`, traits, arguments, results, regions, successors, and attributes.
- For custom C++, find parser and printer implementations using terms such as `parse`, `print`, `OpAsmParser`, `OpAsmPrinter`, and the operation class name. Trace shared helper functions that materially affect syntax.
- Account for mixed definitions: TableGen may declare an operation while C++ implements its assembly, verification, or type inference.
- Search `*.mlir`, tests, examples, docs, FileCheck patterns, snapshots, and real compiler dumps. Look for more than one instance of each important operation so optional clauses and type spellings are visible.
- Record exact operation names and the observed order of operands, literals, attributes or properties, regions or successors, and type signatures. Note parser-only compatibility spellings and printer-canonical spellings separately.

Prefer actual checked-in definitions and representative emitted MLIR over guesses from operation names. If examples are missing, ask the user for a sample or for a safe command that produces one.

Useful searches include:

```sh
rg -n 'assemblyFormat|hasCustomAssemblyFormat|OpAsmParser|OpAsmPrinter|::parse\(|::print\(' PATH
rg -n --glob '*.mlir' --glob '*.td' 'DIALECT_PREFIX|OP_CLASS' PATH
```

## Map evidence to Zirium

Consult Zirium documentation matching the installed version when it is locally available or the user permits looking it up. Confirm uncertain capabilities by trying a minimal registry entry with the binary. Do not assume Zirium accepts the full MLIR ODS `assemblyFormat` language.

For each operation, choose the smallest current mechanism that matches the required examples:

1. An existing preset or built-in implementation.
2. A reusable `operation_shapes` grammar.
3. A supported declarative `operation_formats` description.
4. A clearly reported compatibility gap that may require unknown-operation recovery, generic quoted MLIR, or a future Zirium version.

Do not force richer syntax into a similar-looking shape. Conversely, do not declare a form unsupported merely because documentation omits it: try a minimal registry entry with the installed binary. Keep partial coverage explicit, especially for regions, successors, optional groups, custom directives, properties, and contextual type inference.

## Implement and check

Create or update the user-requested registry JSON in the current repository, preserving unrelated entries. A typical starting point is:

```json
{"presets": [], "builtins": [], "operation_shapes": [], "operation_formats": []}
```

Validate against representative real snippets with the installed binary:

- Pass the registry with `--registry` using the same query and inputs the user expects to run.
- Check the process status, diagnostics, and relevant query output.
- Confirm required operands, results, attributes, regions, or symbols through CLI queries when semantic access matters.
- Test editing, selected-fragment output, or round trips only when they are part of the requested behavior.

Use a few high-value examples rather than manufacturing a large test suite. When a form fails, reduce it to the smallest informative snippet, compare it with the dialect definition, and decide whether the registry or Zirium needs to change.

## Hand-off

Report the created file, installed Zirium version, operations covered, representative commands run, and any syntax variants left unresolved. Phrase limitations as observed gaps in that version, not permanent boundaries, and ask the user whether to extend coverage when meaningful cases remain.
