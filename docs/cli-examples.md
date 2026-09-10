# Querying and editing MLIR from the command line

The `zirium` binary accepts a query as its first argument. It reads MLIR from
standard input when no input path follows the query, or reads each supplied
path as an independent document. From a source checkout, build it with:

```sh
cargo build --bin zirium
```

The [query language reference](query-language.md) lists every predicate,
pipeline stage, and output rule.

The input files and reusable `.zirium` programs used below are checked in under
[`examples/cli`](../examples/cli/). Each section links to its corresponding
files. Short queries are shown inline so you can read and change them directly.

The binary defaults to the baseline registry, which accepts ordinary, named,
and nested `module` shorthand. Use `--preset stablehlo` for a bundled dialect or
`--registry registry.json` to load presets
or caller-defined operations; repeat the flag to combine files. See
[custom formats](custom-formats.md) for configuration and output behavior.

`zirium --help` lists options, and `zirium --list-presets` lists available
dialects. Recovery warnings on stderr mean semantic queries may be incomplete;
add `--strict` to reject such input in scripts. Options can appear before or
after the query argument. Use `--` before input paths beginning with a dash.

The commands below assume `target/debug` is on `PATH`:

```sh
export PATH="$PWD/target/debug:$PATH"
```

## Find untagged arithmetic

Boolean predicates can combine operation names and attributes. This command
prints the untagged `arith.addi` from the sample and omits its tagged
`arith.muli`:

```sh
zirium \
  'filter((op("arith.addi") or op("arith.muli")) and not has_attr("analysis.tag"))' \
  examples/cli/arithmetic.mlir
```

The result contains the add and its enclosing module syntax. Its operand
definitions and users are omitted, so the fragment may not be valid on its own.
The same query is available as
[`untagged-arithmetic.zirium`](../examples/cli/untagged-arithmetic.zirium).

## Find direct consumers

`users` follows one step of SSA use relationships. The add in this sample has
one direct consumer, the multiply. Here is the complete interaction:

```console
$ cat examples/cli/arithmetic.mlir
module {
  %lhs = arith.constant 6 : i32
  %rhs = arith.constant 7 : i32
  %sum = arith.addi %lhs, %rhs : i32
  %product = "arith.muli"(%sum, %rhs) {analysis.tag = "old"} : (i32, i32) -> i32
}
$ zirium 'filter(op("arith.addi")) | users' examples/cli/arithmetic.mlir
builtin.module {
  %v3 = "arith.muli"(%v2, %v1) {analysis.tag = "old"} : (i32, i32) -> i32
}
```

The output contains the multiply. `users` stops after this one step.
The same query is available as
[`direct-consumers.zirium`](../examples/cli/direct-consumers.zirium).

## Combine arithmetic operation kinds

`union` combines the results of two queries. The result contains the add and
multiply in source order:

```sh
zirium \
  'filter(op("arith.addi")) union filter(op("arith.muli"))' \
  examples/cli/arithmetic.mlir
```

This is also a selected fragment, so its SSA dependencies are omitted.
The same query is available as
[`arithmetic-union.zirium`](../examples/cli/arithmetic-union.zirium).

## Extract a call dependency slice

`fixpoint(closure)` adds dependencies until the selection stops changing. For a
`func.call`, that includes the resolved callee and its body, and leaves
unrelated sibling functions out.

```sh
zirium 'filter(op("func.call")) | fixpoint(closure)' examples/cli/calls.mlir
```

The output contains `@caller` and `@answer`, and omits `@unrelated`. Closure
expands supported dependencies. The resulting slice may still need other parts
of the input to be valid MLIR.
The same query is available as
[`call-closure.zirium`](../examples/cli/call-closure.zirium).

## Count operations in a StableHLO decoder

The larger [`stablelm-decode.mlir`](../examples/cli/stablelm-decode.mlir)
example represents one token-generation step for a small two-layer StableLM.
It contains rotary position handling, KV-cache updates, grouped-query
attention, gated MLPs, and a language-model head.

The example is adapted from MLXcel's Apache-2.0 licensed [StableLM decode
program](https://github.com/lablup/mlxcel/blob/0accedd90ae9ea0679121bd087dafdd82882182a/src/lib/mlxcel-xla/assets/stablelm/decode.mlir).
Its large embedded rotary tables are replaced by zero splats; the operation
structure and tensor shapes are unchanged.

The first query below counts every operation. The second counts reductions:

```console
$ zirium --preset stablehlo --strict 'count' examples/cli/stablelm-decode.mlir
237
$ zirium --preset stablehlo --strict 'filter(op("stablehlo.reduce")) | count' examples/cli/stablelm-decode.mlir
14
```

The initial selection contains every operation, including the module and other
container operations.

For a query that will be reused, put the program in a file. The checked-in
[`stablehlo-matmul-count.zirium`](../examples/cli/stablehlo-matmul-count.zirium)
counts `stablehlo.dot_general`, the StableHLO operation used for the decoder's
matrix multiplications:

```console
$ cat examples/cli/stablehlo-matmul-count.zirium
filter(op("stablehlo.dot_general")) | count
$ zirium --preset stablehlo --strict --program-file examples/cli/stablehlo-matmul-count.zirium examples/cli/stablelm-decode.mlir
19
```

The nineteen operations comprise nine matrix multiplications in each decoder
layer and one final projection to logits. A program file contains only Zirium
source; surrounding whitespace and the final newline are ignored.

The count is structural: compact reductions using `applies stablehlo.add` do
not synthesize reducer-body operations. Use an explicit-region input if those
operations must participate in queries.

## Inspect shapes and operation kinds

Name and type projections make inventories easy to pipe into ordinary tools:

```sh
zirium --preset stablehlo --strict \
  'filter(dialect("stablehlo")) | names' examples/cli/stablelm-decode.mlir \
  | sort | uniq -c | sort -nr

zirium --preset stablehlo --strict \
  'filter(op("stablehlo.dot_general")) | result_types | unique' \
  examples/cli/stablelm-decode.mlir
```

Use `result_type("tensor<32xf32>")` to select that exact result type, or `json`
to inspect operand types, result types, and complete dimension clauses together.
Paired custom dimensions such as `[0] x [1]` remain intact in attribute projection.

## Inspect a single output computation

The decoder returns logits, a key cache, and a value cache. Select the first
return operand and follow its SSA definitions, stopping at function inputs:

```sh
zirium --preset stablehlo --strict \
  'filter(op("func.return")) | defs(0) | slice' \
  examples/cli/stablelm-decode.mlir
```

Change the index to `1` or `2` for either cache. `slice` does not expand symbols
or region bodies. At a multi-result operation it follows all explicit operands,
without inferring result-specific dependencies. The output remains an inspection
fragment. Use `fixpoint(closure)` when retaining scopes and supported callees is
more important than a narrow slice.

## Keep operations satisfying a relationship

Find matmuls that directly feed an addition by intersecting the matmul set with
the definitions used by additions:

```sh
zirium --preset stablehlo --strict \
  '(filter(op("stablehlo.dot_general")) intersect (filter(op("stablehlo.add")) | defs)) | json' \
  examples/cli/stablelm-decode.mlir
```

This finds ten matmuls in the decoder and retains the matmul selection. No
separate relationship-predicate syntax is needed for this case. Use `users(0)` to follow only the first result of a
multi-result operation.

## Tag selected operations

Edits preserve the current selection. Appending `input | emit`
returns to the whole edited document and prints all operations:

```sh
zirium \
  'filter(op("arith.addi")) | set_attr("analysis.tag", "review") | input | emit' \
  examples/cli/arithmetic.mlir
```

The output contains the complete document, with `analysis.tag = "review"`
added to `arith.addi`.
The same query is available as
[`tag-add.zirium`](../examples/cli/tag-add.zirium).

## Remove a tag while reading standard input

When no input path follows the query, the binary reads MLIR from standard
input:

```sh
zirium \
  'filter(string_attr_eq("analysis.tag", "old")) | remove_attr("analysis.tag") | input | emit' \
  < examples/cli/arithmetic.mlir
```

The output retains the complete document but removes `analysis.tag`
from `arith.muli`.
The same query is available as
[`remove-tag.zirium`](../examples/cli/remove-tag.zirium).

## Inspect an intermediate selection

`emit` prints the selection and passes it to the next stage. This prints the
add, then its direct users, separated by `// -----`:

```sh
zirium 'filter(op("arith.addi")) | emit | users' examples/cli/arithmetic.mlir
```

## Work inside a function fragment

`subtree` includes every descendant of the selected function. The following
filter therefore finds returns only inside `@caller`:

```sh
zirium \
  'filter(op("func.call")) | parent | subtree | filter(op("func.return"))' \
  examples/cli/calls.mlir
```

## Per-function operation inventories

Run the saved histogram query on the decoder:

```sh
zirium --preset stablehlo --strict \
  -f examples/cli/function-op-counts.zirium examples/cli/stablelm-decode.mlir
```

The result is a JSON object keyed by function name, containing counts keyed by
operation name. In this example, `main` contains 19 `stablehlo.dot_general`
operations. Add `reachable |` before `names` to include supported callees'
bodies, counting shared operations once. See
[bindings and aggregation](query-language.md#naming-intermediate-results) for
the evaluation rules and the distinction between body and reachable counts.
