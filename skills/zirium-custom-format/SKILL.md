---
name: zirium-custom-format
description: Research a local or closed-source repository containing MLIR dialects and create or refine Zirium registry formats for their custom assembly syntax. Use only when the user explicitly invokes $zirium-custom-format.
---

# Zirium Custom Format

## Goal

Produce a small, working Zirium registry configuration grounded in the dialect's real definitions and emitted MLIR. Distinguish MLIR's custom assembly format from Zirium's `operation_formats` and reusable operation shapes.

Zirium's format support is under active development. Treat the current Zirium checkout, tests, and observed behavior as authoritative; use its documentation as guidance, not as a permanent list of exclusions.

## Start with the user

Identify both the dialect repository and the Zirium checkout. If their locations are not clear, ask for them.

Do a light scan first when it will make the questions concrete, then ask the user before choosing formats:

- Which dialects or operations matter first, and is the goal a representative subset or broad coverage?
- Where are representative `.mlir` files or commands that emit them, including important syntax variants?
- What must work: syntax parsing, semantic lowering and queries, rewriting, custom printing, or round trips? Should the result be registry JSON only, or may Zirium itself be extended?

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

Inspect the live Zirium facilities before writing the registry. Start with `docs/custom-formats.md`, `python/zirium/config.py`, `crates/zirium/src/dialect/format.rs`, existing registry JSON, and focused registry tests; locate renamed files with `rg` if needed.

For each operation, choose the smallest current mechanism that matches the required examples:

1. An existing preset or built-in implementation.
2. A reusable `operation_shapes` grammar.
3. A supported declarative `operation_formats` description.
4. A clearly reported gap requiring recovery, generic quoted syntax, or a Zirium change.

Do not force a richer syntax into a similar-looking shape. Conversely, do not declare a form unsupported merely because an older document omits it: inspect the implementation and try a minimal registry entry. Keep partial coverage explicit, especially for regions, successors, optional groups, custom directives, properties, and contextual type inference.

## Implement and check

Create or update the user-requested registry file, preserving unrelated entries. Prefer a direct JSON configuration unless the user asked for reusable Zirium source support.

Validate against representative real snippets with the current Zirium checkout or installed package:

- Load the registry through the same API or CLI the user will use.
- Parse the selected examples and inspect diagnostics.
- If semantic use matters, lower strictly and confirm the required operands, results, attributes, regions, or symbols are accessible.
- Test custom printing or round trips only when they are part of the requested behavior.

Use a few high-value examples rather than manufacturing a large test suite. When a form fails, reduce it to the smallest informative snippet, compare it with the dialect definition, and decide whether the registry or Zirium needs to change.

## Hand-off

Report the created file, operations covered, representative checks run, and any syntax variants left unresolved. Phrase limitations as current observed gaps, not permanent boundaries, and ask the user whether to extend coverage when meaningful cases remain.
