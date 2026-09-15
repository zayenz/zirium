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

After projection, `users`, `defs`, `parent`, `children`, `root`, `subtree`,
`slice`, `closure`, and `reachable` inspect the selected complete document.
`fixpoint(...)` is supported on projected operations and must remain on the
same side. `--strict` makes `reachable` reject operations whose reference
semantics are unknown.

Change and projected-operation streams support `unique`, bounds, reversal,
sets, `sort_by(...)`, keyed extrema, and `map_by(...)`. `names`, `attr`,
`result_types`, and `operand_types` produce value streams; those streams support
`sort`, extrema, `tally`, JSON, and Markdown. Nested expressions start from the
current record or selection, while explicit `input` resets to all changes.

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
comparison and side.

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
