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
files. The commands keep short queries inline so they can be read and changed
without opening another file.

The binary uses the proving registry and accepts ordinary, named, and nested
`module` shorthand. Use `--registry registry.json` to load caller-defined operation shapes; repeat
the flag to combine files. See [custom formats](custom-formats.md)
for the supported syntax and output boundaries.

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

The result is an intentional selected fragment: Zirium retains the module
shell needed to print the add, but does not add the add's operands or users.
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

Its selected-fragment output contains `arith.muli`, without recursively
following further relationships.
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
`func.call`, that includes the resolved callee and its body, while unrelated sibling functions are omitted.

```sh
zirium 'filter(op("func.call")) | fixpoint(closure)' examples/cli/calls.mlir
```

The output contains `@caller` and `@answer`, and omits `@unrelated`. Closure
expands the selection, but its output is still a slice of the input rather than
a promise that every selected fragment is independently valid MLIR.
The same query is available as
[`call-closure.zirium`](../examples/cli/call-closure.zirium).

## Count operations in a StableHLO decoder

The larger [`stablelm-decode.mlir`](../examples/cli/stablelm-decode.mlir)
example represents one token-generation step for a small two-layer StableLM.
It contains rotary position handling, KV-cache updates, grouped-query
attention, gated MLPs, and a language-model head.

The example is adapted from MLXcel's Apache-2.0 licensed
[StableLM decode program](https://github.com/lablup/mlxcel/blob/0accedd90ae9ea0679121bd087dafdd82882182a/src/lib/mlxcel-xla/assets/stablelm/decode.mlir).
Its large embedded rotary tables are replaced by zero splats; the operation
structure and tensor shapes are unchanged.

The first query below counts every operation. The second counts reductions:

```console
$ zirium 'count' examples/cli/stablelm-decode.mlir
237
$ zirium 'filter(op("stablehlo.reduce")) | count' examples/cli/stablelm-decode.mlir
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
$ zirium --program-file examples/cli/stablehlo-matmul-count.zirium examples/cli/stablelm-decode.mlir
19
```

The nineteen operations comprise nine matrix multiplications in each decoder
layer and one final projection to logits. A program file contains only Zirium
source; surrounding whitespace and the final newline are ignored.

## Tag selected operations

Mutations keep the selection as the pipeline value. Appending `input | emit`
returns to the whole edited document and prints all operations:

```sh
zirium \
  'filter(op("arith.addi")) | set_attr("analysis.tag", "review") | input | emit' \
  examples/cli/arithmetic.mlir
```

The output is the complete input document with
`analysis.tag = "review"` added to `arith.addi`. Unlike the earlier selected
fragments, this output contains the whole edited document.
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

`root` includes every descendant of the selected function. The following
filter therefore finds returns only inside `@caller`:

```sh
zirium \
  'filter(op("func.call")) | parent | root | filter(op("func.return"))' \
  examples/cli/calls.mlir
```
