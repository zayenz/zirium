# Semantic diff

Zirium compares two complete semantic MLIR documents with `--diff`:

```sh
zirium --diff before.mlir after.mlir
zirium --diff before.mlir after.mlir 'filter(changed("operands")) | json'
zirium --diff before.mlir after.mlir -f report.zirium
```

The paths are adjacent and ordered before, then after. Either may be `-`, but
not both. With no program Zirium prints a human report. A successful comparison
exits 0 whether or not changes exist.

## Comparison contract

The comparison ignores whitespace, comments, SSA names, block labels, and
operation-attached locations by default. `--diff-locations` includes the latter.
It compares operation fields, ordered SSA connections, successors, regions,
and relative sibling order. It does not prove computational equivalence or
recognize dialect-specific rewrites.

Reordering two distinguishable sibling operations is therefore a change even
when they are independent: both records are marked moved. Semantic diff
describes represented structure and dataflow, not freedom to reschedule
operations.

Each operation produces at most one `added`, `removed`, `modified`, or `moved`
record. A modified operation that moved stays `modified` with `moved = true`.
Records follow after-document structural preorder, then before-side removals.

Resource-backed attributes and trailing file metadata are rejected because
their external payload is not represented completely. Opaque and wide values
are compared byte-for-byte.

## Queries and output

Diff programs start with change records. `change("added")`,
`change("removed")`, `change("modified")`, and `change("moved")` select kinds.
`changed("attributes")`, `changed("operands")`, and the other documented field
names select direct changes. Ordinary predicates inspect the after endpoint
when present, otherwise the before endpoint.

`before` and `after` project records to operations in that document. Missing
endpoints are dropped. Projected selections use the normal MLIR printer. Diff
programs are read-only.

Diff `filter` and `root` use the shared predicate reader. A named argument and
`_` denote the existing change or side-owned operation subject:

```zirql
filter(c => changed("attributes") and length(c | names) > 0) |
after | root(p => op("builtin.module") and one(_ | names) == "builtin.module") | names
```

`root` tests each ancestor candidate in its projected document. Captured saved
bindings remain available, native leaves keep their change/before/after meanings,
and pure shared expressions use the same cumulative budget. Change predicates
refuse operation streams, including empty ones. Operations and changes do not
expose record or list fields; select native projections explicitly. This diff
surface has no function-declaration syntax or native capture clauses.

After projection, `users`, `defs`, `parent`, `children`, `root`, `subtree`,
`slice`, `closure`, and `reachable` inspect the selected complete document.
`fixpoint(...)` is supported on projected operations and must remain on the
same side. `where_exists(relation)` keeps original changes or projected
operations when a pure singleton-relative relation is nonempty. It preserves
order, duplicates and side ownership, and checks the relation even on empty
input. A relation must return its incoming native kind and projected side.
Explicit `input` still resets to the original changes; a relation that then
projects the other side refuses. Nested errors propagate.

```sh
zirium --diff examples/cli/diff-before.mlir examples/cli/diff-after.mlir \
  'after | where_exists(users) | names'
```

Rust callers use `changed_ops.where_exists(&diff_op_input().users())`.
`--strict` makes `reachable` reject operations whose reference
semantics are unknown.

Change and projected-operation streams support `unique`, bounds, reversal,
sets, `sort_by(...)`, keyed extrema, and `map_by(...)`. `names`, `attr`,
`result_types`, and `operand_types` produce value streams; those streams support
`unique`, `sort`, extrema, `tally`, JSON, and Markdown. Tally also accepts
change and operation streams, retaining comparison ownership and source side. Nested expressions start from the
current record or selection, while explicit `input` resets to all changes.

Native captures and `paths(...)` are unavailable on this diff surface, including
after `before` or `after`. A path request receives a located refusal; no traversal
implicitly crosses document sides.

Text programs and typed builders use one native diff host with Zirql's statement,
expression, predicate and collection execution. Nested fixed-point bodies,
selectors and navigation share the active work, item and nesting allowance.

Diff selections follow the shared [native set operators](https://zayenz.se/zirql/guides/library-native/#native-set-operators),
[collection stages](https://zayenz.se/zirql/guides/library-pipelines/) and
[ordering](https://zayenz.se/zirql/guides/library-ordering/). Dictionary sorting,
reversal and bounds retain dictionary kind in saved, nested and JSON results.
Navigation keeps its domain order. Native identity includes comparison ownership
and before/after side; ordering never derives it from rendered descriptors.

Diff dictionaries use Zirql's shared schema and entry admission. Retained
schemas are checked even when the dictionary is empty, and nested dictionaries
retain their native contents. Each native entry must belong to the current
comparison and agree with its before/after side. For example:

```zirql
a = after | head(1);
{a: {a: a}} | count
```

This retains the selected after operation as both keys and the inner value.
`count` returns `1`. Rendering remains a separate boundary.

[Fixed points](https://zayenz.se/zirql/guides/library-native/#fixpoint) must retain
the admitted side. The initial change stream must fit the item ceiling before
filtering. Flat pipelines and set chains do not add a nesting level per term.

Diff output follows its statement-end policy: `emit` passes the value through
without writing, while `json`, `markdown` and `print` transform the value.
A query statement writes its final value once. A binding or `do` statement
writes nothing. Ordinary operation programs emit at each written emitter.
The CLI stages all output until the complete program succeeds.

`markdown` on a change selection retains the native change table with its context columns.
Ordinary string, map and constructed data use Zirql's
[shared Markdown layouts](https://zayenz.se/zirql/guides/library-rendering/#render_report_markdown).
Diff output retains its existing table framing, including header-only empty
string and map tables. Native before/after operation descriptors and change
classification remain part of Zirium.

Bindings retain their stream side. The reserved scalar bindings
`before_document` and `after_document` contain the two CLI labels for
`print("{before_document} -> {after_document}")` and similar reports.

Change JSON uses schema `zirium.diff.v1`. Endpoints contain the operation name,
symbol context, structural path, and half-open byte range. `--jsonl` adds both
document labels and result-side attribution.

## Rust

```rust
use zirium::diff::{compare, ChangeField, DiffLimits, DiffOptions};
use zirium::query::{changed, changes};

let delta = compare(&before, &after, &registry,
    DiffOptions::default(), DiffLimits::default())?;
let after_ops = delta.query(
    &changes().filter(changed(ChangeField::Attributes)).after()
)?;
```

`Diff` borrows both immutable documents. Its handles are scoped to that
comparison and side, including empty projected selections. Surviving handles
remain native through bindings; strings and counts are evaluated values.

`DiffProgram::parse(source)` provides the same text program as a library API.
`evaluate` accepts a comparison, both document labels, `EvaluationLimits`,
`EvaluationOptions` and an emission callback receiving native `DiffProgramOutput` values.
`evaluate_with_budget` accepts a caller-owned `zirql::Budget`, including an
explicit `Budget::for_program` aggregate allowance. Binding copies, projections,
nested traversal and report construction spend that same allowance. The CLI
retains its legacy limits and rejects `--max-results` in diff mode. JSON conversion
and MLIR printing remain explicit rendering boundaries.

## Python

```python
import zirium
from zirium.query import changed, changes

delta = zirium.diff(before, after)
records = delta.query(changes().filter(changed("attributes")))
after_operations = delta.query(changes().after())
```

Python takes immutable snapshots of both documents. Snapshot-backed operation
wrappers remain valid if an original is edited or dropped, at the cost of
retaining semantic storage for both inputs.

Use `change_input()` and `diff_op_input()` for relative typed expressions such
as fixed-point bodies. They do not reset to the full comparison.

## Scaling checks

The deterministic test suite checks that comparison work stays linear when a
unique-operation chain or opaque payload doubles, and that oversized ambiguous
groups switch to the bounded fallback. Wall time is intentionally not a CI
assertion. Run the release driver at several adjacent sizes for local timing:

```sh
cargo run --release -p zirium --example diff_benchmark -- 64 128 256 512 1024
```

Each row reports its size, workload case, input bytes, changes, matches,
comparison work, fallback counters, and elapsed milliseconds. Lowering happens
before the timer.

Text and typed string queries accept the same unique operation. Full diff
programs keep actual tally keys until rendering and allow dictionary continuation:

```zirql
hist = after | tally;
hist | reverse | head(1) | sort_by(value) | json
```

A before-side key and an after-side key remain distinct even when their operation
names match. Use `names` explicitly when the intended identity is the spelling.
See the shared [unique and tally reference](https://zayenz.se/zirql/guides/library-pipelines/#unique).
