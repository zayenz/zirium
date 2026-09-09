# Zirium query language reference

A Zirium query starts with `select`, which tests every operation in the input
document. Pipeline stages then navigate, combine, edit, or summarize the
selection. Stages run from left to right.

```zirium
select(op("arith.addi") and not has_attr("analysis.tag")) | defs | count
```

Pass a query directly to the `zirium` binary or store it in a file and use
`--program-file` (or `-f`).

```sh
zirium 'select(op("arith.addi")) | count' input.mlir
zirium --program-file query.zirium input.mlir
```

See the [CLI examples](cli-examples.md) for complete queries and sample MLIR
files.

## Program structure

Every program has one initial selection and zero or more pipeline stages:

```text
program   = "select" "(" predicate ")" { "|" stage }

predicate = or-expression
or-expression  = and-expression { "or" and-expression }
and-expression = not-expression { "and" not-expression }
not-expression = { "not" } primary
primary   = operation-predicate
          | attribute-predicate
          | "(" predicate ")"

operation-predicate = "op" "(" string ")"
attribute-predicate = "has_attr" "(" string ")"
                    | "attr" "(" string "," string ")"

stage = "closure" | "defs" | "users" | "parent" | "children"
      | "union" "(" predicate ")"
      | "intersect" "(" predicate ")"
      | "except" "(" predicate ")"
      | "set_attr" "(" string "," string ")"
      | "remove_attr" "(" string ")"
      | "count" | "root"
```

Whitespace, including newlines, may appear between tokens. The language has no
comment syntax.

`not` binds more tightly than `and`, and `and` binds more tightly than `or`.
Parentheses override that order.

## Predicates

Predicates test one operation at a time.

| Predicate | Matches |
| --- | --- |
| `op("name")` | Operations whose full name is exactly `name`. |
| `has_attr("name")` | Operations with an attribute named `name`, regardless of its value. |
| `attr("name", "value")` | Operations whose named attribute is a string equal to `value`. |

`attr` matches only MLIR string attributes. It compares decoded string values,
so `attr("message", "say \"hi\"")` matches an MLIR attribute spelled
`message = "say \22hi\22"`.

Combine predicates with `not`, `and`, and `or`:

```zirium
select((op("arith.addi") or op("arith.muli")) and not has_attr("skip"))
```

Operation names must be non-empty. Attribute names use dotted ASCII
identifiers: each component starts with a letter or underscore and continues
with letters, digits, or underscores. Names such as `analysis.tag` and
`_zirium.state_2` are valid.

## Selection stages

These stages replace or combine the current selection. Results are
deduplicated and returned in source order.

| Stage | Resulting selection |
| --- | --- |
| `defs` | Operations that directly define operands of the selected operations. A block argument resolves to the operation that owns its region. |
| `users` | Operations that directly use results of the selected operations. |
| `parent` | The immediate enclosing operation of each selected operation. |
| `children` | Operations directly contained in the regions and blocks owned by the selected operations. |
| `closure` | The selection plus its transitive dependencies. |
| `union(predicate)` | The current selection plus every operation matching `predicate`. |
| `intersect(predicate)` | Selected operations that also match `predicate`. |
| `except(predicate)` | Selected operations that do not match `predicate`. |

`defs`, `users`, `parent`, and `children` move one step and do not retain the
input selection unless it also appears in the result.

`closure` follows SSA definitions. For block arguments, it retains the owning
operation and its contents. It also follows registered `func.call` targets and
the successors of `cf.br` and `cf.cond_br`. Closure fails when it encounters an
unregistered operation or a symbol or successor reference it does not support.
It does not guess how an unknown operation uses references.

The predicates in `union`, `intersect`, and `except` test all operations in the
current document, including edits made by earlier stages.

## Edit stages

Edit stages change every selected operation and keep the same selection for
the next stage.

| Stage | Effect |
| --- | --- |
| `set_attr("name", "value")` | Adds or replaces `name` with an MLIR string attribute. |
| `remove_attr("name")` | Removes `name`. Missing attributes are left unchanged. |

`set_attr` accepts the same dotted attribute names as the attribute predicates.
Its value may not contain control characters. Edits are buffered and committed
atomically for each stage.

The binary writes edited MLIR to standard output. It does not overwrite the
input files.

## Output stages

Without an output stage, the binary prints the final selection.

| Stage | Output |
| --- | --- |
| `count` | The number of selected operations, followed by a newline. |
| `root` | The complete current document after validation. |

`count` and `root` are terminal: no pipeline stage may follow either one.
`root` is useful after an edit when the output should contain the complete
document rather than a selected fragment.

Selection output retains the enclosing syntax needed to print the selected
operations. An operation that owns regions also retains their contents. The
result is an intentional fragment of the input, not a guarantee of standalone
valid MLIR. In particular, Zirium does not add SSA definitions or users unless
the query selects them or reaches them through a stage such as `defs`, `users`,
or `closure`.

## Strings and limits

Strings use double quotes. The query language supports two escapes:

| Escape | Value |
| --- | --- |
| `\"` | A double quote. |
| `\\` | A backslash. |

Other escapes are errors. The parser reports query errors at byte offsets and
rejects predicates that exceed its 64-level nesting limit. A query error or
evaluation error produces no MLIR output.
