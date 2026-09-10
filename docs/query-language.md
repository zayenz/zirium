# Zirium query language

A query transforms an ordered stream. It starts with every operation
in the input document. A pipe passes the current stream to the next stage:

```zirium
filter(op("arith.addi")) | users | filter(has_attr("analysis.tag"))
```

This finds `arith.addi` operations, follows their direct users, and keeps those
with an `analysis.tag` attribute.
`filter` always tests the current operation stream. Navigation replaces that
stream; edits preserve it. Navigation preserves order and duplicates. Set
operators and `closure` produce source-ordered sets; `unique` explicitly
removes duplicates from other streams.

`input` and a final `emit` are implicit. These programs print the same document:

```zirium
# An empty program is valid.
```

```zirium
input
```

```zirium
input | emit
```

The CLI accepts a query argument or a program file (`-f` / `--program-file`).
With no arguments, it runs the empty program on standard input. Use an empty
quoted argument to run the empty program on files:

```sh
zirium '' input.mlir
zirium 'filter(op("arith.addi")) | count' input.mlir
zirium -f query.zirium input.mlir
```

Input files are independent documents. `input` never combines different files.
The selected-fragment printer may change formatting even when printing the
whole input. Use the library's original-output API to reproduce input bytes.

## Predicates and filtering

Predicates test one operation. They appear inside `filter(...)`.

| Predicate | Matches |
| --- | --- |
| `true` / `false` | Every operation / no operation. |
| `op("name")` | An operation with exactly this full name. |
| `has_attr("name")` | An operation with this attribute, regardless of its value. |
| `string_attr_eq("name", "value")` | An operation whose named attribute is a string equal to this decoded value. |

Combine predicates with `not`, `and`, and `or`, in that precedence order.
Parentheses override precedence:

```zirium
filter((op("arith.addi") or op("arith.muli")) and not has_attr("skip"))
```

Use `filter(not predicate)` to exclude matches. `filter(true)` leaves the
selection unchanged. `count` alone counts every operation, including container
operations such as `builtin.module`.

String equality compares decoded values. For example,
`string_attr_eq("message", "say \"hi\"")` matches the MLIR string attribute
`message = "say \22hi\22"`. It does not compare numeric attributes or their
printed spellings.

Operation names must be non-empty. Attribute names use dotted ASCII
identifiers: each component starts with a letter or underscore, followed by
letters, digits, or underscores. `analysis.tag` and `_zirium.state_2` are valid.

## Input, navigation, and fragments

| Stage | Resulting selection |
| --- | --- |
| `input` | Every operation in the current document, including earlier edits. |
| `defs` | Operations directly defining operands of the selected operations. A block argument resolves to the operation owning its region. |
| `users` | Operations directly using results of the selected operations. |
| `parent` | Each selected operation's immediate enclosing operation. |
| `children` | Operations directly contained in the selected operations' regions and blocks. |
| `root(predicate)` | The nearest operation on each selected operation's ancestor chain that matches the predicate. |
| `subtree` | Each selected operation and all its descendants. |
| `closure` | The selection plus one step of supported dependency expansion. |

`defs`, `users`, `parent`, and `children` move one step and replace the input
selection. They do not automatically keep it. `parent` drops operations with no
enclosing operation. `defs` and `users` are not inverse relationships: a block
argument belongs to an enclosing operation rather than an operation result.

`root(predicate)` tests each selected operation, then walks toward the document
root until it finds a match. It returns at most one match for each input item.
Repeated matches remain repeated, so `unique` commonly follows `root`:

```zirium
filter(op("linalg.matmul"))
| root(op("func.func"))
| unique
```

`subtree` expands each selected operation independently. Overlapping subtrees
therefore contain duplicates. An empty stream stays empty.

For example, find the function containing a call, expand its body, and tag its returns:

```zirium
filter(op("func.call")) | parent
| subtree
| filter(op("func.return"))
| set_attr("analysis.tag", "review")
```

Returns in sibling functions are unaffected. Append `input` to
print the complete edited document.

Selection and printing are distinct. Selecting a function counts as one
operation, but printing it includes its body. Printing a nested operation also
retains the enclosing syntax needed to represent it. Neither behavior adds
those operations to the query selection. Use `subtree` when descendants must
participate in filtering, counting, or editing.

Fragments may omit SSA definitions and users, so they are not guaranteed to
be standalone valid MLIR. Select the needed dependencies explicitly.

## Combining queries

`union`, `intersect`, and `except` are infix operators between selection
queries. Both operands receive the same incoming selection.

```zirium
filter(op("arith.addi")) union filter(op("arith.muli"))
```

For this example, `filter(op("arith.addi") or op("arith.muli"))` is shorter.
Set operators are useful when the operands navigate differently:

```zirium
filter(op("arith.addi")) | (defs union users)
```

This combines the definitions and users of the selected adds. It does not run
`users` on the output of `defs`.

| Operator | Result |
| --- | --- |
| `left union right` | Operations in either result. |
| `left intersect right` | Operations in both results. |
| `left except right` | Operations in the left result but not the right. |

Pipes bind more tightly than set operators. All set operators have equal
precedence and associate left to right. Parenthesize a set expression before
applying a stage to its combined result:

```zirium
(filter(op("arith.addi")) | users union filter(op("arith.muli")) | defs)
| count
```

An operand searches the complete document only when it receives the initial
selection or explicitly uses `input`. This query adds every tagged operation
in the document to the current selection:

```zirium
filter(op("arith.addi")) | (filter(true) union (input | filter(has_attr("tag"))))
```

Set operands cannot edit or count. Apply edits and counting after grouping the
set expression. Operands may use `emit` for inspection; emissions occur from
left to right. Both operands still see the same document and incoming selection.

## Closure and fixed points

`closure` retains the selection and expands dependencies once. It adds direct
SSA definitions. Block arguments add their owning operation and its subtree;
`func.call` adds its resolved callee and subtree; `cf.br` and `cf.cond_br` add
the successor region's owning operation and subtree. These subtree expansions
retain the corresponding scope, but dependencies of newly added operations
are followed on subsequent applications.

Use `fixpoint(closure)` for the complete supported dependency slice:

```zirium
filter(op("func.call")) | fixpoint(closure)
```

Closure requires registered operations with supported reference semantics.
It fails on an unregistered operation or an unsupported symbol or successor
reference. It does not infer how unknown operations use references.

`fixpoint(query)` repeatedly replaces the selection with the query's result
until the selection is unchanged. Its body can contain navigation, filters,
set expressions, grouping, and nested fixed points. It cannot edit or count.
Use `fixpoint(closure | emit)` to inspect each iteration, including the final
unchanged result. If selections cycle without becoming unchanged, evaluation
fails. Emission does not affect convergence.

Fixed points do not implicitly accumulate intermediate selections:

```zirium
# Retain the seed and add users until no more are found.
filter(op("arith.addi")) | fixpoint(filter(true) union users)

# Follow definitions until unchanged; on an acyclic chain this becomes empty.
filter(op("arith.addi")) | fixpoint(defs)
```

The same rule applies to `parent`: repeatedly moving beyond the document's
roots eventually gives an empty selection. Use `root(predicate)` to find a
matching ancestor and `subtree` to expand a selected fragment.

## Projection and uniqueness

`attr("name")` replaces each operation with its named attribute value and drops
operations without that attribute. It decodes string and symbol attributes.
For other attribute kinds, it returns the MLIR spelling. Operation-only stages,
including navigation, filtering, and edits, reject value streams. `emit` and
the implicit final emission print one projected value per line.

`unique` keeps the first copy of each operation or value. It preserves stream
order. For example, this prints the names of functions that contain a matrix
multiplication:

```zirium
filter(op("linalg.matmul"))
| root(op("func.func"))
| unique
| attr("sym_name")
| json
```

## Edits and emission

`set_attr("name", "value")` adds or replaces a string attribute on every
selected operation. `remove_attr("name")` removes an attribute; missing
attributes are ignored. Both preserve the selection. Later filters and
`input` see the edits. Attribute values passed to `set_attr` cannot contain
control characters. Each edit stage commits atomically.

`emit` prints the current fragment and passes the same selection onward:

```zirium
filter(op("arith.addi")) | emit | users
```

This prints the adds, then their users. Each explicit emission captures the
document at that point, before later edits. An implicit final emission prints
the result unless the program ends with an explicit `emit`, including inside
a final group. `emit | emit` therefore prints twice, while `emit` prints once.
Nested queries do not implicitly reset to `input` or emit their results. A fixed-point
body ending in `emit` emits each iteration; the enclosing program still emits
its final result unless it ends with an explicit `emit`.

`count` prints the stream's size followed by a newline. It is terminal;
no stage may follow it. To emit a fragment and then count it, use `emit | count`.

`json` emits the current stream as a JSON array and passes the stream onward.
For value streams, the array contains strings. For operation streams, each
entry contains the operation name and an object of attribute spellings. This
format favors inspection and interchange; Zirium cannot read it back as MLIR.
Like `emit`, a final `json` suppresses the implicit final emission.

Consecutive fragment outputs are separated by `// -----`. Counts are plain
lines. The CLI buffers all emissions across all input files until processing
succeeds: a query, evaluation, or printing error produces no standard output.
Input files are never overwritten. Rust callers use the `Query::evaluate`
emission callback and can choose their own buffering policy.

## Grammar and diagnostics

```text
program    = [ query ]
query      = pipeline { ("union" | "intersect" | "except") pipeline }
pipeline   = stage { "|" stage }
stage      = "input" | "filter" "(" predicate ")"
           | "defs" | "users" | "parent" | "children" | "closure"
           | "root" "(" predicate ")" | "subtree" | "unique"
           | "attr" "(" string ")" | "json"
           | "fixpoint" "(" query ")" | "(" query ")"
           | "set_attr" "(" string "," string ")"
           | "remove_attr" "(" string ")" | "emit" | "count"
predicate  = and-expr { "or" and-expr }
and-expr   = not-expr { "and" not-expr }
not-expr   = { "not" } primary
primary    = "true" | "false" | "op" "(" string ")"
           | "has_attr" "(" string ")"
           | "string_attr_eq" "(" string "," string ")"
           | "(" predicate ")"
```

Whitespace is insignificant between tokens. `#` begins a comment extending to
the end of the line; inside a string it is a literal character. Strings use
double quotes and support `\"` and `\\`. Other escapes are errors.

Query and predicate nesting share a 64-level limit. Long flat boolean chains,
pipelines, and set chains do not require corresponding recursive nesting.
The CLI reports query errors with a byte offset, line, column, and source
caret. Program-file positions include leading whitespace and comments.

For implementation timing and scaling measurements, see
[query profiling](architecture/query-profiling.md).
