---
name: zirium-cli
description: Use Zirium's CLI to inspect, query, report on, slice, and edit textual MLIR safely and efficiently. Use when a task mentions the `zirium` command, `.zirium` query programs, command-line MLIR analysis or edits, Zirium presets or registries, or composing and debugging Zirium queries.
---

# Zirium CLI

## Goal

Use the installed `zirium` binary to inspect textual MLIR or make requested
structural edits. Keep queries short and report the command and dialect assumptions.

Check `zirium --help` and `zirium --version` against the installed release.
Follow the help's reference link when syntax or behavior is unclear. Do not
install, upgrade, or modify inputs unless the user asks.

## Start safely

1. Confirm `zirium` is available with `command -v zirium`.
2. Read `zirium --help`; use `zirium --list-presets` when custom dialect syntax
   may be present.
3. Inspect a small representative input or ask where the relevant `.mlir`
   files are. Never assume all files use the same dialects.
4. Begin with a diagnostic query such as `names | tally | json` or a narrow
   `filter(...) | json` before composing traversal or edits.

The CLI reads stdin when no input path is given and treats supplied files as
independent documents. It buffers output and never overwrites inputs. Use `--`
before an input path beginning with `-`.

## Choose dialect support deliberately

Without registry flags, Zirium uses its baseline registry. If stderr warns
about recovered custom operations, semantic data may be incomplete.

- Add repeatable `--preset NAME` flags for bundled dialects.
- Add repeatable `--registry FILE` flags for project-specific JSON registries.
- Use `--strict` in automation or when incomplete semantics would mislead.
  It rejects recovery and unknown reference semantics in `reachable`, without
  adding dialect support or full dialect verification.
- If the project needs a new registry format, consult the
  [custom-format guide](https://github.com/zayenz/zirium/blob/main/docs/custom-formats.md)
  for its Zirium version. The optional `zirium-custom-format` companion skill
  supports that research when the user explicitly invokes it.

## Compose queries from the data flow

A query transforms an ordered stream that initially contains every operation.
Keep these distinctions explicit:

- `filter(...)` keeps matching operations; `defs`, `users`, `parent`, and
  `children` replace the selection with related operations.
- Navigation can preserve duplicates. Add `unique` before counting distinct
  operations, but keep duplicates when they carry multiplicity for `tally`.
- `subtree` expands nested operations. `slice` follows transitive SSA
  definitions. `reachable` also follows supported bodies and references.
  `fixpoint(closure)` retains supported dependency context for a fragment.
- `union`, `intersect`, and `except` combine selections. Parenthesize a set
  expression before piping its combined result onward.
- `names`, `attr(...)`, `operand_types`, and `result_types` project values.
  Use `tally`, `map_by(...)`, object/array literals, `json`, or `markdown` for
  reports.

Read [query recipes](references/query-recipes.md) when choosing a traversal,
editing a document, producing a structured report, or diagnosing a failed
query.

## Edit without losing the document

`set_attr(...)` and `remove_attr(...)` edit the selected operations while
preserving the selection. Use `do EDIT; emit` to suppress the edited selection
and then print the complete edited document once. Write stdout to a new path,
inspect it, and replace an original only when the user asks.

Example:

```sh
zirium --strict \
  'do filter(op("arith.addi")) | set_attr("analysis.tag", "review"); emit' \
  input.mlir > tagged.mlir
```

Selected fragments may omit SSA definitions, users, or surrounding operations
and need not be valid standalone MLIR. Do not present a fragment as a complete
rewritten module unless a later statement starts from the full document, such
as the `emit` after `do` above, or the pipeline returns to `input`.

Printing may normalize formatting, operation spelling, or SSA names beyond the
requested edit. Verify the semantic change with a strict query and review the
full diff.

## Use program files for nontrivial work

Keep simple queries in single-quoted shell arguments. Put multiline queries,
bindings, reports, or complicated quoting in a `.zirium` file and run
`zirium -f analysis.zirium input.mlir`. A program file contains Zirium source,
not shell syntax.

Zirium concatenates statement outputs without adding separators. Add explicit
`print(...)` statements when a human-facing report needs headings or spacing.

When evaluation exceeds its deterministic safeguards, simplify accidental
duplicate growth first. Raise `--max-work` or `--max-items` only when the query
is intentional and the input size justifies it.

## Report the result

State the query or program file used, input scope, registry flags, whether
`--strict` passed, and what the output represents. Mention warnings, unsupported
custom forms, fragments that are not standalone, or edits not written back.
