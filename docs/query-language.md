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
operators on operations, `closure`, `slice`, and `reachable` produce source-ordered sets; `unique` explicitly
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
With no arguments, it runs the empty program on standard input. An input path
of `-` also reads standard input, so `zirium 'count' -` works in a pipeline.
Use an empty
quoted argument to run the empty program on files:

```sh
zirium '' input.mlir
zirium 'filter(op("arith.addi")) | count' input.mlir
zirium -f query.zirium input.mlir
zirium --preset stablehlo --strict 'filter(op("stablehlo.dot_general")) | count' model.mlir
```

Use `zirium --help` for CLI usage and `zirium --list-presets` for bundled dialects.
`--preset NAME` and `--registry FILE` are repeatable and combine their registries.
Options can appear before or after the query; `--` ends option processing.
`-f` replaces the inline query rather than adding another program.

Without a registry option, Zirium uses the baseline registry. Unsupported custom
operations may still support name and structural queries, but their operands,
attributes, and types can be incomplete. The CLI warns on stderr when it recovers
these operations. Use `--strict` to reject recovery with no stdout, especially in
scripts. Strict mode requires supported parsing; it does not enable full dialect
verification or reconstruct operations implicit in custom assembly.

Input files are independent documents. `input` never combines different files.
Output is buffered until all inputs and statements succeed; a later error
leaves stdout empty. Multiple results are concatenated, so multiple `json`
outputs are separate JSON values rather than one combined JSON document.
Closing an output pipe early (for example with `head`) exits quietly.
The selected-fragment printer may change formatting even when printing the
whole input. Use the library's original-output API to reproduce input bytes.

## Naming intermediate results

Bindings make a longer query easier to read and let several parts reuse the same
selection. Each binding is evaluated once for each input document:

```zirium
adds = filter(op("stablehlo.add"));
matmuls = filter(op("stablehlo.dot_general"));

(adds union matmuls) | names | tally | json
```

A binding saves its result. It is not a query macro: using `adds` later restores
the saved stream, regardless of the current selection. Each right-hand side
starts with all operations in the document and may refer to earlier bindings.
The final expression also starts with all operations.

Names use ASCII letters, digits, and underscores, starting with a letter or
underscore. Stage names, predicate names, and language keywords are reserved.
Bindings cannot be reassigned, refer forward, or refer to themselves. Each
binding needs a trailing semicolon. Query statements may follow bindings or
appear between them.

Bindings preserve order and duplicates. Saved operations refer to the live
document, so later edits are visible through those operations. Projected strings
and aggregates retain their saved values. Bindings cannot edit or emit; put
those stages in a query statement. Use `union` to combine saved selections;
`or` combines predicates inside `filter`.

## Counting values and building maps

`tally` counts how many times each value occurs and returns a map from strings
to counts. For example, `names | tally` counts operations by their full names.
An empty value stream produces `{}`. Maps print as JSON objects, with keys
sorted lexically. `json` can also emit a map explicitly.

To map function names to their number of matmuls:

```zirium
matmuls = filter(op("stablehlo.dot_general"));

matmuls | root(op("func.func")) | attr("sym_name") | tally | json
```

Each matmul contributes its enclosing function's name once. Keep the duplicates:
adding `unique` would lose the information needed for the count. Functions
without matmuls are absent. On `examples/cli/stablelm-decode.mlir`, this produces
`{"main": 19}` when run with `--preset stablehlo`.

For a separate analysis of each function, use `map_by(key_query, value_query)`:

```zirium
functions = filter(op("func.func"));

functions
| map_by(
    attr("sym_name"),
    children | subtree | names | tally
  )
| json
```

Both queries run independently on each single input operation. Here the key is
the function's name, and the value is a histogram of its body. `children |
subtree` includes explicitly represented nested regions and excludes the
function operation itself. An empty body produces an empty histogram.

The key must produce exactly one string. Missing keys and duplicate keys are
errors. If different symbol scopes contain the same function name, select a
scope before building the map, or use an attribute with a unique identifier.
By contrast, `tally` intentionally combines equal strings across scopes.

The value may be a count, a map, or a stream. Streams become JSON arrays, using
the same representation as `json`. For example, replace `names | tally` with
`count` for a total body size, or with `result_types | tally` for a type
histogram. Maps can be nested. Key and value queries cannot edit or emit.
Bindings used inside them still restore their saved results; they do not
become queries relative to the current function.

## Ordering, bounds, and extrema

`sort` orders a value stream lexically. `reverse` reverses an operation or
value stream. `head(n)` keeps the first `n` items, while `tail(n)` keeps the
last `n`; both preserve the retained items' order.

Use `sort_by(query)` to order operations by a derived key. The selector runs
once for each operation and must return exactly one string or count. Sorting is
stable, so operations with equal keys retain their previous order:

```zirium
filter(op("func.func")) | sort_by(attr("sym_name")) | head(10)
```

`min` and `max` return the lexical minimum or maximum of a non-empty value
stream. `min_by(query)` and `max_by(query)` return one operation using the same
selector rules as `sort_by`; the first operation wins a tie. The corresponding
`min_all`, `max_all`, `min_all_by(query)`, and `max_all_by(query)` stages retain
every tied extreme in its previous order. All extrema stages report an error on
an empty stream. Use `reverse` after `sort` or `sort_by` for descending order.

Maps remain sorted lexically by key. Ordering stages do not rank `tally` or
`map_by` maps.

## Building JSON structures

Object and array literals combine saved results into a larger report:

```zirium
functions = filter(op("func.func"));
n = functions | count;
counts = functions
  | map_by(attr("sym_name"), children | subtree | names | tally);

{
  "title": "Report for {n} functions",
  "function_count": n,
  "operation_counts": counts,
  "metadata": {"schema_version": 1, "static_counts": true},
  "notes": ["Shared callees count once when using reachable", null]
} | json
```

A bare binding inserts its saved result as structured data. A count stays a
number, a map stays an object, and a stream becomes an array. Operation streams
use the same inspection objects as `json`. Inserted data is copied into the
new structure; subsequent edits do not change it.

Strings and object keys use the same `{name}` interpolation as `print`.
For example, `"count": n` stores a number, while `"count": "{n}"` stores a
string. Interpolation accepts counts and single strings, including a literal
array containing exactly one string. Use `{{` and `}}` for literal braces.
Quotes, newlines, and other characters in inserted strings are escaped when
JSON is emitted; inserted text is never parsed as JSON.

Objects require quoted keys and reject duplicate keys, including duplicates
created by interpolation. Values may be objects, arrays, strings, JSON numbers,
`true`, `false`, `null`, or earlier bindings. Trailing commas and forward
references are errors. Values cannot contain query pipelines; bind the result
of a query first. All bindings retain their saved meaning inside literals.

A literal is a pipeline stage that replaces the current result. It must start
with `{` or `[`; scalars occur inside objects and arrays. Literals may be saved
in bindings or embedded within one another:

```zirium
totals = names | tally;
report = {"totals": totals, "complete": true};
[report, {"note": "Counts include container operations"}] | json
```

Objects and arrays also print as JSON through implicit output or `emit`.
`count` counts outer object keys or array elements. `markdown` accepts literal
objects with its supported table shapes and arrays of scalars; deeper structures
produce its usual shape diagnostic. Literal construction is not a selection
query, so it cannot be used directly as a set operand or fixed-point body.

The shared nesting, item, and work limits apply to literal construction.
Inserted copies count toward the resulting structure's size, and empty
containers count too. Numbers use the JSON representation's integer and
floating-point ranges; use strings for numbers requiring arbitrary precision.

## Writing reports with Markdown and text

Use `markdown` to emit a table from a histogram or a map of histograms.
Use `print("text")` for headings and explanatory text between results:

```zirium
functions = filter(op("func.func"));
n = functions | count;

print("# Function operation report");
print("");
print("Functions inspected: {n}");
functions
| map_by(attr("sym_name"), children | subtree | names | tally)
| markdown;
print("Counts describe explicitly represented operations.");
```

Each semicolon ends a statement. Statements run in order, and each query
statement starts with every operation in the current document. Its result is
implicitly emitted unless it ends with an explicit emitter. Bindings may appear
between query statements; they are evaluated where written and never emit.
The final query statement may omit its semicolon. A program containing bindings
still needs a query statement after its last binding.

Prefix a query with `do` to evaluate it while suppressing its implicit result.
This makes edit-only statements explicit and lets a later statement choose what
to print:

```zirium
do filter(op("arith.addi"))
   | set_attr("analysis.tag", "review");

emit
```

The edit remains visible to later statements, and the final `emit` prints the
complete edited document once. A `do` statement requires its trailing
semicolon, including at the end of a program. Explicit emitters inside its query
still emit; `do` suppresses only the implicit result.

`print` appends a newline. It leaves the current stream unchanged, so it also
works inside a pipeline. A final `print` suppresses implicit output, just like
a final `json` or `markdown`. Use `print("")` for a blank line. Template
strings can span actual lines; the string escape rules below still apply.

Interpolation uses `{name}`, where `name` is an earlier binding containing a
count or exactly one string. Empty or multiple strings, maps, and operation
selections produce a diagnostic: project or count them first. Interpolation
does not evaluate expressions or traverse map fields. Write `{{` and `}}`
for literal braces. Inserted values are plain text, with no Markdown escaping
and no further interpolation.

The Markdown emitter supports these shapes:

| Input | Output |
| --- | --- |
| String stream or scalar array | A one-column table headed `Value`, preserving order and duplicates. |
| Map of scalars | A `Key` / `Value` table. |
| Map of maps of scalars | Outer keys become rows; the union of inner keys becomes columns. |
| Saved count | The number as a paragraph. |
| Empty stream or map | The paragraph *No entries.* |

For example, `{"a": {"add": 2}, "b": {"add": 1, "mul": 3}}` renders as:

| Key | add | mul |
| --- | --- | --- |
| a | 2 | |
| b | 1 | 3 |

Map rows and columns are sorted lexically. Missing cells are blank, not inferred
to be zero. Scalars are strings, numbers, booleans, or null; null also produces
a blank cell. An inner map with no keys contributes a row with blank cells.
If all inner maps are empty, the table has only its `Key` column.

Operation streams, mixed scalar/map rows, arrays inside maps, and deeper maps
produce a shape diagnostic. Use `names`, `tally`, or `map_by` to form a
table, or use `json` for deeper data. The diagnostic identifies the offending
entry or cell when possible.

`markdown` preserves the current result, as `json` does. It surrounds its
output with blank lines, escapes Markdown and HTML characters in cells, and
renders embedded newlines as `<br>`. The table uses GitHub-flavored Markdown.
The expanded table, including missing cells and headers, must fit
`--max-items`; emitted text also consumes `--max-work`. A late shape or
interpolation error leaves CLI stdout empty, including earlier `print` output.

## Counting reachable operations

Adding `reachable` includes supported dependencies and referenced bodies:

```zirium
functions = filter(op("func.func"));

functions
| map_by(
    attr("sym_name"),
    children | subtree | reachable | names | tally
  )
| json
```

Suppose `main` calls `helper` twice, and `helper` contains one add and one
return. The body histogram for `main` includes its two calls and its own
return. The reachable histogram also includes one add and the helper's return.
The helper's body is counted once, even though two calls reach it. If the helper
calls itself, traversal still terminates and counts each static operation once.
These counts describe the represented code, not execution frequency.

`reachable` includes seeds, their explicit nested bodies, and transitive SSA
definitions. Block arguments are boundaries. Direct `func.call` and registered
`call_like` operations resolve their `callee` symbol and include the target's
body. The target declaration itself is not added merely because it is called.
Registered `func_like` operations participate in symbol lookup. Supported
`cf.br` and `cf.cond_br` edges include their target blocks without widening
the selection to the enclosing function. The result is a source-ordered set.

A lambda represented by a named function and a direct call works with these
rules. Indirect calls through function values, captures with dialect-specific
semantics, and other unsupported reference kinds require additional dialect
support. Unknown operations, unresolved callees, external callees without
bodies, and declared unsupported references cause errors. Load the appropriate
registry; a shape or format registration describes the supported structure,
not arbitrary dialect behavior.

As with body counts, compact assembly can omit implicit operations from the
represented structure. `reachable` does not synthesize those operations.
Use `closure` when retaining enclosing scopes for a printable fragment;
`reachable` deliberately stops at block arguments for analysis.

## Predicates and filtering

Predicates test one operation. They appear inside `filter(...)`.

| Predicate | Matches |
| --- | --- |
| `true` / `false` | Every operation / no operation. |
| `op("name")` | An operation with exactly this full name. |
| `dialect("name")` | An operation whose name starts with exactly this dialect followed by a dot. |
| `result_type("type")` | An operation with at least one result whose retained type spelling equals this string. |
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

For example, `filter(dialect("stablehlo") and result_type("tensor<32xf32>"))`
finds StableHLO operations producing that tensor type. Type matching is exact
spelling matching; use `result_types` or operation JSON to inspect shapes and
element types when more involved analysis is needed.

Operation names must be non-empty. Attribute names use dotted ASCII
identifiers: each component starts with a letter or underscore, followed by
letters, digits, or underscores. `analysis.tag` and `_zirium.state_2` are valid.

## Input, navigation, and fragments

| Stage | Resulting selection |
| --- | --- |
| `input` | Every operation in the current document, including earlier edits. |
| `defs` | Operations directly defining operands of the selected operations. A block argument resolves to the operation owning its region. |
| `users` | Operations directly using results of the selected operations. |
| `defs(index)` | The definition of one zero-based operand of each selected operation. |
| `users(index)` | Uses of one zero-based result of each selected operation. |
| `parent` | Each selected operation's immediate enclosing operation. |
| `children` | Operations directly contained in the selected operations' regions and blocks. |
| `root(predicate)` | The nearest operation on each selected operation's ancestor chain that matches the predicate. |
| `subtree` | Each selected operation and all its descendants. |
| `closure` | The selection plus one step of supported dependency expansion. |
| `slice` | The selection and its transitive SSA definitions, stopping at block arguments. |
| `reachable` | The selection, explicit bodies, SSA definitions, and supported referenced bodies, each operation once. |

`defs`, `users`, `parent`, and `children` move one step and replace the input
selection. They do not automatically keep it. `parent` drops operations with no
enclosing operation. `defs` and `users` are not inverse relationships: a block
argument belongs to an enclosing operation rather than an operation result.
An out-of-range navigation index drops that input item. Indexed navigation
preserves duplicates just like its unindexed form.

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

Use `slice` to inspect a computation without expanding the whole function when
an operand is a function input:

```zirium
# Follow the first returned value, for example logits rather than cache outputs.
filter(op("func.return")) | defs(0) | slice
```

`slice` follows explicit SSA operands only. It does not expand callees, successors,
or region bodies, and does not infer which operands affect individual results of
a multi-result operation. Generic quoted operations need no registration for
this traversal; recovered unparsed operations and invalid operands are errors.
Use `closure` when retaining complete scopes and supported symbol dependencies
is the desired behavior.

`defs(index)` still maps a block argument directly to its owning operation. If
the selected return operand is itself a function argument, that navigation
selects the function; `slice` does not undo that selection. Starting `slice` at
the return instead retains the return and stops at its argument operands.

Compact StableHLO reductions using `applies stablehlo.add` are represented as a
single operation. Their implicit reducer body is not synthesized, so `children`
is empty and operation counts differ from the explicit-region spelling. Counts
describe the parsed structural representation, not normalized MLIR.

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

Set operators also accept value streams. Value union keeps first appearance
from left then right; intersection and difference preserve left-side order.
Both operands must produce the same stream kind.

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

Set operands must return operation or value streams; they cannot edit, count,
or construct maps. Apply edits and counting after grouping the
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
set expressions, grouping, and nested fixed points. It cannot edit, count, or construct maps.
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

Duplicate-preserving expansion need not converge: `fixpoint(subtree)` keeps
adding copies when an ancestor and its descendant are selected. Use
`fixpoint(subtree | unique)` when a set is intended. Cycle detection catches
repeating selections; evaluation limits also stop streams that keep growing.

By default, each evaluation permits 10,000,000 work units and 1,000,000 items per
stream. Work counts stage input items (at least one per stage), fixed-point
iterations, and dependency, ancestor, and subtree visits. These are deterministic safeguards,
not a time or byte-memory limit. CLI callers can set `--max-work N` and
`--max-items N`; Rust callers can use `Query::evaluate_with_limits` and
`EvaluationLimits`. A limit error follows the usual no-stdout CLI contract.

The common `fixpoint(closure)` query uses a worklist and visits dependencies once.
A body with additional stages, including `emit` or `json`, runs step by step and
retains its per-iteration output. Deep slices with per-iteration output can still
be expensive and may need a larger work limit.

## Projection and uniqueness

`attr("name")` replaces each operation with its named attribute value and drops
operations without that attribute. It decodes string and symbol attributes.
For other attribute kinds, it returns the MLIR spelling. Operation-only stages,
including navigation, filtering, and edits, reject value streams. `emit` and
the implicit final emission print one projected value per line.

An attribute string that cannot be decoded as UTF-8 produces an error rather
than disappearing from the stream. Use operation `json` to inspect its escaped
spelling. Paired StableHLO dot dimensions preserve both sides: for example,
`attr("contracting_dims")` returns `[0] x [1]`. Generic fragment printing
represents this custom clause as nested arrays `[[0], [1]]`.

`names` projects operation names. `result_types` and `operand_types` project
retained type spellings in result or operand order, flattening across selected
operations. These stages produce value streams and preserve duplicates. For
example, `filter(dialect("stablehlo")) | names | unique` lists the StableHLO
operation kinds present in the input.

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

Use a `do` statement when an edit should not itself produce output. A following
query starts from the complete edited document, so `do EDIT; emit` prints one
whole document without first printing the edited selection.

`emit` prints the current fragment and passes the same selection onward:

```zirium
filter(op("arith.addi")) | emit | users
```

This prints the adds, then their users. Each explicit emission captures the
document at that point, before later edits. An implicit final emission prints
the result unless the query statement ends with an explicit emitter, including inside
a final group. `emit | emit` therefore prints twice, while `emit` prints once.
Nested queries do not implicitly reset to `input` or emit their results. A fixed-point
body ending in `emit` emits each iteration; the enclosing program still emits
its final result unless the statement ends with an explicit emitter.

`count` prints the stream's size followed by a newline. It is terminal;
no stage may follow it. To emit a fragment and then count it, use `emit | count`.

`json` emits streams as JSON arrays and maps as JSON objects, then passes the result onward.
For value streams, the array contains strings. For operation streams, each
entry contains the operation name, an object of attribute spellings, and
`operand_types` and `result_types` arrays. Unknown type
information can appear as a placeholder or JSON null on incompletely understood
input. This format favors inspection and interchange; Zirium cannot read it back as MLIR.
Like `emit`, a final `json` suppresses the implicit final emission.

Statement outputs and results from multiple input files are concatenated
without an automatic separator. Use `print` when a report needs headings,
spacing, or a delimiter. Counts are plain lines. The CLI buffers all emissions
across all input files until processing succeeds: a query, evaluation, or
printing error produces no standard output. Input files are never overwritten.
Rust callers use the `Query::evaluate` emission callback and can choose their
own buffering policy.

## Grammar and diagnostics

```text
program    = empty | { statement } final
statement  = binding | do-statement | query ";"
final      = query [ ";" ] | do-statement
binding    = identifier "=" query ";"
do-statement = "do" query ";"
query      = pipeline { ("union" | "intersect" | "except") pipeline }
pipeline   = stage { "|" stage }
stage      = object | array | identifier | "markdown" | "print" "(" string ")"
           | "tally" | "map_by" "(" query "," query ")"
           | "sort" | "sort_by" "(" query ")" | "reverse"
           | ("head" | "tail") "(" integer ")"
           | "min" | "min_all" | ("min_by" | "min_all_by") "(" query ")"
           | "max" | "max_all" | ("max_by" | "max_all_by") "(" query ")"
           | "input" | "filter" "(" predicate ")"
           | ("defs" | "users") [ "(" integer ")" ]
           | "parent" | "children" | "closure" | "slice" | "reachable"
           | "root" "(" predicate ")" | "subtree" | "unique"
           | "attr" "(" string ")" | "names" | "result_types" | "operand_types" | "json"
           | "fixpoint" "(" query ")" | "(" query ")"
           | "set_attr" "(" string "," string ")"
           | "remove_attr" "(" string ")" | "emit" | "count"
predicate  = and-expr { "or" and-expr }
and-expr   = not-expr { "and" not-expr }
not-expr   = { "not" } primary
primary    = "true" | "false" | "op" "(" string ")"
           | "dialect" "(" string ")" | "result_type" "(" string ")"
           | "has_attr" "(" string ")"
           | "string_attr_eq" "(" string "," string ")"
           | "(" predicate ")"
object     = "{" [ string ":" value { "," string ":" value } ] "}"
array      = "[" [ value { "," value } ] "]"
value      = object | array | string | number | "true" | "false" | "null" | identifier
integer    = digit { digit }
number     = a JSON number
```

Whitespace is insignificant between tokens. `#` begins a comment extending to
the end of the line; inside a string it is a literal character. Strings use
double quotes and support JSON escapes: `\"`, `\\`, `\/`, `\b`, `\f`, `\n`,
`\r`, `\t`, and `\uXXXX` (including valid surrogate pairs). Actual newlines
and tabs are also allowed in query strings. Other escapes are errors.

Query and predicate nesting share a 64-level limit. Materialized objects and arrays also have
a 64-level nesting limit, including structures built through successive bindings.
Nested contents count toward `--max-items`, and copying saved results counts
toward `--max-work`. Long flat boolean chains,
pipelines, and set chains do not require corresponding recursive nesting.
The CLI reports query errors with a byte offset, line, column, and source
caret. Program-file positions include leading whitespace and comments.

For implementation timing and scaling measurements, see
[query profiling](architecture/query-profiling.md).
