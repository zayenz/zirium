# Zirium CLI query recipes

Use these recipes as starting points. Confirm available stages and current
semantics with `zirium --help` and its linked query-language reference.

## Inspect before specializing

Inventory operation kinds:

```sh
zirium --strict 'names | tally | json' input.mlir
```

Inspect one operation kind, including attributes and retained types:

```sh
zirium --strict 'filter(op("arith.addi")) | json' input.mlir
```

List distinct result types in a dialect:

```sh
zirium --preset stablehlo --strict \
  'filter(dialect("stablehlo")) | result_types | unique' model.mlir
```

## Select and traverse

Direct users of additions:

```zirium
filter(op("arith.addi")) | users | unique
```

Definitions contributing to the first returned value:

```zirium
filter(op("func.return")) | defs(0) | slice
```

Complete supported dependency context for calls:

```zirium
filter(op("func.call")) | fixpoint(closure)
```

Functions containing a matrix multiplication:

```zirium
filter(op("linalg.matmul")) | root(op("func.func")) | unique
```

Use `slice` for SSA analysis, `reachable` to count explicit and referenced bodies
once, and `fixpoint(closure)` to retain enclosing scopes and supported callees.
Unknown reference semantics stop `reachable` at a leaf unless `--strict` is set.
None of these queries guarantees a valid standalone module.

## Count and report

Count distinct direct users:

```zirium
filter(op("arith.addi")) | users | unique | count
```

Keep duplicates for a histogram of containing function names:

```zirium
filter(op("stablehlo.dot_general"))
| root(op("func.func"))
| attr("sym_name")
| tally
| json
```

Build a per-function operation histogram:

```zirium
functions = filter(op("func.func"));

functions
| map_by(
    attr("sym_name"),
    children | subtree | names | tally
  )
| json
```

Bindings restore saved results; they are not macros relative to the current
selection. Each binding and query statement starts from the full document
unless it consumes an incoming selection inside a nested query.

## Check a transformation

Require at least one matrix multiplication and no remaining tensor allocations:

```zirium
do filter(op("linalg.matmul")) | check("expected a lowered matmul");
do filter(op("bufferization.alloc_tensor"))
   | check(0, "tensor allocations must be eliminated");
```

Use `check(n)` for an exact cardinality. Both forms accept an optional message,
which is included in the failure diagnostic. A failed check gives the CLI a
nonzero exit status. `do` keeps individual successful checks silent; invoke a
complete check program with `zirium --silent -f checks.zirium input.mlir` to
suppress all results.

## Combine selections

Combine operation kinds with a predicate when both traverse identically:

```zirium
filter(op("arith.addi") or op("arith.muli"))
```

Use set operators when branches traverse differently:

```zirium
(filter(op("arith.addi")) | users
 union
 filter(op("arith.muli")) | defs)
| unique
| count
```

Pipes bind more tightly than set operators. Both operands receive the same
incoming selection. Group the set expression before applying a stage to its
combined result.

## Edit safely

Tag selected operations without emitting their fragment, then emit the complete
document:

```zirium
do filter(op("arith.addi"))
   | set_attr("analysis.tag", "review");
emit
```

Remove a string attribute from matching operations:

```zirium
do filter(string_attr_eq("analysis.tag", "old"))
   | remove_attr("analysis.tag");
emit
```

The CLI never overwrites inputs. Redirect to a new file and check stderr and the
exit status. For this edit, a strict negative check confirms that no addition
lacks the requested value:

```sh
zirium --strict \
  'filter(op("arith.addi") and not string_attr_eq("analysis.tag", "review")) | count' \
  tagged.mlir
```

Expect `0`. `do` suppresses only its query's implicit result; any explicit
emitter inside it still runs. The following `emit` starts from the complete,
edited document.

Review the full diff. Printing may normalize formatting, operation spelling,
or SSA names beyond the requested edit.

## Diagnose failures

- Query parse errors include a line, column, and caret. Move multiline or
  heavily quoted source into a `.zirium` file.
- Recovery warnings mean unregistered custom operations were accepted with
  incomplete semantics. Load the right preset or registry, then rerun with
  `--strict`.
- Empty results are often a wrong exact operation/type spelling. Inspect
  `names | tally | json`, `result_types | unique`, or a narrow operation `json`.
- Unexpectedly high counts often come from duplicate-preserving navigation or
  overlapping `subtree` expansion. Decide whether multiplicity is meaningful;
  otherwise add `unique` before `count`.
- A selected fragment can print enclosing syntax while its query selection
  still contains fewer operations. Use `subtree` when descendants must
  participate in filtering or counting.
- A work/item limit error can reveal non-converging duplicate growth such as
  `fixpoint(subtree)`. Prefer `fixpoint(subtree | unique)` when a set is meant.
- Semantic traversal errors on calls, successors, or unknown operations usually
  require appropriate registry support rather than a larger evaluation limit.
