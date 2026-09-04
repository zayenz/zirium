# Querying and editing MLIR from the command line

The `zirium` binary accepts a query as its first argument. It reads MLIR from
standard input when no input path follows the query, or reads each supplied
path as an independent document. From a source checkout, build it with:

```sh
cargo build --bin zirium
```

The commands below assume `target/debug` is on `PATH`:

```sh
export PATH="$PWD/target/debug:$PATH"
```

## Find untagged arithmetic

Boolean predicates can combine operation names and attributes. This command
prints the untagged `arith.addi` from the sample and omits its tagged
`arith.muli`:

```sh
zirium 'select((op("arith.addi") or op("arith.muli")) and not has_attr("analysis.tag"))' examples/cli/arithmetic.mlir
```

The result is an intentional selected fragment: Zirium retains the module
shell needed to print the add, but does not add the add's operands or users.

## Find direct consumers

`users` follows one step of SSA use relationships. The add in this sample has
one direct consumer, the multiply:

```sh
zirium --program-file examples/cli/direct-consumers.zirium examples/cli/arithmetic.mlir
```

The checked-in program contains:

```zirium
select(op("arith.addi")) | users
```

Its selected-fragment output contains `arith.muli`, without recursively
following further relationships.

## Combine arithmetic operation kinds

`union(predicate)` combines the current selection with all operations matching
another predicate. The result contains the add and multiply in source order:

```sh
zirium --program-file examples/cli/arithmetic-union.zirium examples/cli/arithmetic.mlir
```

This is also a selected fragment, so its SSA dependencies are omitted.

## Extract a call dependency slice

`closure` adds transitive dependencies. For a `func.call`, that includes the
resolved callee and its body, while unrelated sibling functions are omitted:

```sh
zirium --program-file examples/cli/call-closure.zirium examples/cli/calls.mlir
```

The output contains `@caller` and `@answer`, and omits `@unrelated`. Closure
expands the selection, but its output is still a slice of the input rather than
a promise that every selected fragment is independently valid MLIR.

## Tag selected operations

Mutations keep the selection as the pipeline value. Appending `root` returns
to the whole edited document, validates it, and prints all operations:

```sh
zirium --program-file examples/cli/tag-add.zirium examples/cli/arithmetic.mlir
```

The output is the complete input document with
`analysis.tag = "review"` added to `arith.addi`. Unlike the earlier selected
fragments, `root` output after a mutation is a validated whole document.

## Remove a tag through standard input

The long option and positional input above are convenient for reusable
programs. Programs can also use `-f`, while MLIR comes from standard input:

```sh
zirium -f examples/cli/remove-tag.zirium < examples/cli/arithmetic.mlir
```

The whole validated output retains both arithmetic operations but removes
`analysis.tag` from `arith.muli`.

The inline untagged-arithmetic query is also checked in as
[`untagged-arithmetic.zirium`](../examples/cli/untagged-arithmetic.zirium).
A program file contains only Zirium source; surrounding whitespace and its
final newline are ignored.
