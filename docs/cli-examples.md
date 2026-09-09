# Querying and editing MLIR from the command line

The `zirium` binary accepts a query as its first argument. It reads MLIR from
standard input when no input path follows the query, or reads each supplied
path as an independent document. From a source checkout, build it with:

```sh
cargo build --bin zirium
```

The [query language reference](query-language.md) lists every predicate,
pipeline stage, and output rule.

Every query and input used below is checked in under
[`examples/cli`](../examples/cli/). Each section links to its corresponding
`.zirium` program. The commands keep short queries inline so they can be read
and changed without opening another file.

The binary uses the proving registry and accepts ordinary, named, and nested
`module` shorthand. Use `--registry registry.json` to load caller-defined operation shapes; repeat
the flag to combine files. See [custom formats across Rust, Python, and the CLI](custom-formats.md)
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
  'select((op("arith.addi") or op("arith.muli")) and not has_attr("analysis.tag"))' \
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
$ zirium 'select(op("arith.addi")) | users' examples/cli/arithmetic.mlir
builtin.module {
  %v3 = "arith.muli"(%v2, %v1) {analysis.tag = "old"} : (i32, i32) -> i32
}
```

Its selected-fragment output contains `arith.muli`, without recursively
following further relationships.
The same query is available as
[`direct-consumers.zirium`](../examples/cli/direct-consumers.zirium).

## Combine arithmetic operation kinds

`union(predicate)` combines the current selection with all operations matching
another predicate. The result contains the add and multiply in source order:

```sh
zirium \
  'select(op("arith.addi")) | union(op("arith.muli"))' \
  examples/cli/arithmetic.mlir
```

This is also a selected fragment, so its SSA dependencies are omitted.
The same query is available as
[`arithmetic-union.zirium`](../examples/cli/arithmetic-union.zirium).

## Extract a call dependency slice

`closure` adds transitive dependencies. For a `func.call`, that includes the
resolved callee and its body, while unrelated sibling functions are omitted.
This example loads the query from
[`call-closure.zirium`](../examples/cli/call-closure.zirium):

```sh
zirium --program-file examples/cli/call-closure.zirium examples/cli/calls.mlir
```

The program file contains:

```zirium
select(op("func.call")) | closure
```

The output contains `@caller` and `@answer`, and omits `@unrelated`. Closure
expands the selection, but its output is still a slice of the input rather than
a promise that every selected fragment is independently valid MLIR.

A program file contains only Zirium source. Surrounding whitespace and the
final newline are ignored.

## Tag selected operations

Mutations keep the selection as the pipeline value. Appending `root` returns
to the whole edited document, validates it, and prints all operations:

```sh
zirium \
  'select(op("arith.addi")) | set_attr("analysis.tag", "review") | root' \
  examples/cli/arithmetic.mlir
```

The output is the complete input document with
`analysis.tag = "review"` added to `arith.addi`. Unlike the earlier selected
fragments, `root` output after a mutation is a validated whole document.
The same query is available as
[`tag-add.zirium`](../examples/cli/tag-add.zirium).

## Remove a tag while reading standard input

When no input path follows the query, the binary reads MLIR from standard
input:

```sh
zirium \
  'select(attr("analysis.tag", "old")) | remove_attr("analysis.tag") | root' \
  < examples/cli/arithmetic.mlir
```

The validated output retains the complete document but removes `analysis.tag`
from `arith.muli`.
The same query is available as
[`remove-tag.zirium`](../examples/cli/remove-tag.zirium).
