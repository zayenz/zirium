# Zirium query language

A query transforms an ordered stream. It starts with every operation
in the input document. A pipe passes the current stream to the next stage:

```zirium
filter(op("arith.addi")) | users | filter(has_attr("analysis.tag"))
```

This finds `arith.addi` operations, follows their direct users, and keeps users
with an `analysis.tag` attribute. Navigation replaces the stream and preserves
duplicates; edits keep the selection. Operation set operators, `closure`,
`backward_slice`, `forward_slice`, and `reachable` produce source-ordered sets.
`slice` is an alias for `backward_slice`. Use `unique` to remove duplicates
from other streams.

For Rust and Python builders with native results, see the
[structured query guide](query-dsl.md).

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
Use an empty quoted argument to run the empty program on files:

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
`--silent` evaluates the complete program but suppresses all query results on
stdout; warnings and errors still appear on stderr, and failures still return a
nonzero exit status.
Value-taking long options accept both separated and equals forms, such as
`--preset stablehlo` and `--preset=stablehlo`. `-f -` is a literal query
filename, and standard input may be supplied as an MLIR input only once.

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
The reserved `document` binding holds the input path, or `stdin`. Use it in
`print("{document}")` and interpolated JSON strings and keys. Redefining it is
an error. Library callers opt into this context with
`Query::parse_with_document_context` and
`evaluate_with_context_options_and_limits`; ordinary `Query::parse` reports
`{document}` as unavailable.
Closing an output pipe early (for example with `head`) exits quietly.
The selected-fragment printer may change formatting even when printing the
whole input. Use the library's original-output API to reproduce input bytes.

## Required query parameters

Declare caller inputs before using them:

```zirium
param operation_name: string;
param minimum_ops: int;
filter(op(operation_name)) | head(minimum_ops) | names | json
```

Pass each required value with a repeated `--param name=value` option. Inline
programs and program files use the same admission rules:

```sh
zirium --param operation_name=arith.addi --param minimum_ops=2 'param operation_name: string; param minimum_ops: int; filter(op(operation_name)) | head(minimum_ops) | names | json' examples/cli/function-report.mlir
zirium --param operation_name=arith.addi --param minimum_ops=2 -f examples/cli/function-report-parameters.zirql examples/cli/function-report.mlir
```

Types are exactly `int`, `string` and `bool`. Integers are signed 64-bit decimal
values with an optional `+` or `-`; whitespace, fractions and overflow fail.
Booleans are exactly `true` or `false`. A string is every character after the
first `=`, including an empty value, whitespace, quotes, backslashes or further
`=` characters. Shell quoting protects the argument; query string syntax is
not applied to its contents.

Every declaration requires one input. Duplicate declarations or inputs,
missing inputs, extra names and type mismatches fail before operations are
selected or edited. Declarations have no defaults. Names use the existing
binding identifier rules, cannot shadow reserved names or earlier bindings,
and become visible at their written declaration. A comprehension-local name
in an earlier statement may precede a later declaration of the same name.
Forward references and shadowing visible declarations fail at the name.
`param` remains contextual: an existing binding such as `param = names; param`
keeps its meaning.

Parameters are immutable shared scalar values. Pure expressions and checked
primitive arguments use the same scope and request allowance. String predicates
(`op`, `dialect`, `has_attr`, `result_type`, `string_attr_eq`), `attr`, integer
`head`/`tail`/indexed `defs`/`users`, and `check` counts accept checked expressions.
Their native name, nonnegative count and index rules still apply. Known wrong
types fail even in an empty stream or unchosen predicate branch. Inputs are
never inserted into query source. Diff queries reject supplied parameters.

The [parameterized function report](../examples/cli/function-report-parameters.zirql)
keeps the literal report's `double` and `mix` rows, total operations `5` and total
matches `2` for the values above. The [literal program](../examples/cli/function-report.zirql)
and shorter commands remain usable without declarations or `--param`.

## Naming intermediate results

Bindings make a longer query easier to read and let several parts reuse the same
selection. Each binding is evaluated once for each input document:

```zirium
adds = filter(op("stablehlo.add"));
matmuls = filter(op("stablehlo.dot_general"));

(adds union matmuls) | names | tally | json
```

A binding saves a result. Using `adds` later restores that stream regardless
of the current selection. Each right-hand side starts from the whole document
and may use earlier bindings. The final expression also starts from the whole
document.

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

The key must produce exactly one string; missing or duplicate keys are errors.
If function names repeat across scopes, select one scope or use a unique
attribute as the key. `tally` combines equal strings across scopes.

The value may be a count, a map, or a stream. Streams become JSON arrays, using
the same representation as `json`. For example, replace `names | tally` with
`count` for a total body size, or with `result_types | tally` for a type
histogram. Maps can be nested. Key and value queries cannot edit or emit.
Bindings used inside them still restore their saved results; they do not
become queries relative to the current function.

## Ordering, bounds, and extrema

`sort` orders a value stream lexically. `reverse` reverses an operation stream,
value stream, or explicitly ranked map. `head(n)` keeps the first `n` items,
while `tail(n)` keeps the last `n`; both preserve the retained items' order.
These stages truncate an already computed stream. They do not avoid parsing the
whole file or evaluating earlier stages.

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

Unranked maps remain sorted lexically by key. Use `sort_by(value)` on a `tally`
or `map_by` map whose values are comparable scalars of one type. The result is
an explicitly ranked map, and `reverse`, `head`, and `tail` preserve that order
through bindings, JSON, and Markdown output. Equal values use lexical key order;
`reverse` reverses the whole order, including ties. Nested map values are not
implicitly ranked: project a scalar first.
Here `value` is a map-entry selector, not a general pipeline stage.

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

## Positional comprehensions

Use `[expression for name in source if condition]` to construct a report from
projected values. The filter is optional. For this complete input:

```mlir
"alpha"() : () -> ()
"beta"() : () -> ()
```

this query projects operation names, constructs records, then emits JSON:

```zirium
labels = names;
[{"name": n, "label": "data:" ++ n} for n in labels] | json
```

The output is:

```json
[
  {"label": "data:alpha", "name": "alpha"},
  {"label": "data:beta", "name": "beta"}
]
```

Repeated `for` clauses can depend on earlier bindings. The rightmost generator
varies fastest; each `if` filters its own clause before later clauses run:

```zirium
labels = names;
[{"name": n, "label": "data:" ++ s}
 for n in labels
 for s in [n, n ++ "!"] if s != "beta!"] | json
```

The rows have labels `data:alpha`, `data:alpha!`, and `data:beta`, in that order.
A generator sees earlier names in its source, then installs its binder for its
filter and the head. Duplicate binders, shadowing, forward references, and
undefined names fail at their query locations. `_` discards a value and cannot
be read. Nested comprehensions retain their own array shape.

Inside a comprehension, object labels are static quoted strings and bare names
are values. `{"n": n}` uses the label `n` and the bound value of `n`.
Strings are decoded literals: `"{n}"` contains braces and the letter `n`.
Outside comprehensions, plain JSON-shaped literals keep their existing binding
and interpolation rules. `for` and `if` remain ordinary binding names elsewhere,
including a literal such as `[for]` when `for` is bound.

The supported expression forms are:

- Boolean, signed integer, finite decimal float, and string literals; earlier
  bindings and generator names; grouping with parentheses.
- Positional arrays, records with quoted labels, and nested array comprehensions.
- Numeric sign, `not`, checked `+`, `-`, `*`, `/`, integer `div` and `mod`;
  numeric `<`, `<=`, `>`, `>=`; typed `==` (also `=`) and `!=`.
- Boolean `and` and `or`; string or positional-array `++`; record access
  `r.field`, array access `a[i]`, and inclusive integer ranges `lo..hi`.
- `length`, `sum`, `forall`, `exists`, `require`, `group_by`, and `one` calls;
  explicitly projected native queries and correlated `project` rows.

Array positions start at one. Integer arithmetic uses signed 64-bit values and
refuses overflow or division by zero. Floats use finite binary64 values;
nonzero literals or arithmetic results rounded to zero fail. Mixed numeric
operations require exact integer-to-float conversion. Strings receive no numeric
coercion. Boolean operands short-circuit, so `false and (1 div 0 == 1)` skips the
runtime division. Every branch still receives scope and type validation before
execution; a skipped branch cannot hide incompatible types or an unsupported
call. Other expression calls and unlisted forms are refused.

Use an explicit projection such as `names`, `attr`, or `count` before constructing
scalar reports from operations. Native operation, type, and attribute handles
retain their identity and cannot become comprehension values through their
printed spelling. Saved JSON arrays and objects can supply scalar or structured
values; `null` and integers outside `i64` fail when projected into this subset.

Rust callbacks receive `QueryOutput::Native(NativeValue::Constructed(report))`
before rendering. `report.value()` borrows the ordered Zirql value, including
record field order. `Clone` shares this immutable result. Evaluator binding reads
and emission snapshots use charged copies. `report.render_json_with_budget()`
and `report.render_ndjson_with_budget()` render under the active allowance;
`Query::evaluate_with_budget()` supplies that allowance to callbacks. The CLI
prints constructed values as JSON and supports the usual `--jsonl` envelope.
`json` and `emit` pass the constructed value onward. MLIR and collection stages
that do not support this report kind fail before execution.

Comprehension requests use one budget for bounded parsing/lowering, validation,
source projections, candidates before filters, nested construction, copies,
snapshots, and JSON rendering. The default aggregate allowance is 1,000,000
produced or copied entries; `--max-results` changes it. `--max-work` and
`--max-items` retain their work and per-stream bounds. Reusing a parsed Rust
query charges its retained parse spending on each evaluation. Rust callers can
bound parsing with `Query::parse_with_program_limits()` and choose execution
bounds with `Query::evaluate_with_program_limits()`; callbacks that render can
use `Query::evaluate_with_budget()` with `Budget::for_program()`.

## Stable record groups

`group_by(rows, "field")` accepts a positional record array and a literal string
selector. Keys must be homogeneous integers, strings or nominal members of one
type. It returns records with `key` then `items`, preserving first-seen group
order and every row's order and duplicates. Empty input returns an empty array;
other record fields may differ. Keyed arrays need an explicit positional
conversion. Keys retain typed identity and receive no string coercion.

For the [function fixture](../examples/cli/function-report.mlir), this short
program groups functions by their represented `arith.addi` count:

```zirium
let rows = filter(op("func.func")) | project(f, {
  "name": f | attr("sym_name") | one,
  "matches": f | children | subtree | filter(op("arith.addi")) | count
});
group_by(rows, "matches") | json;
```

The first group has key `1` and items `double`, `mix`; the second has key `0`
and item `plain`. The [CLI example](cli-examples.md#group-functions-by-match-count)
executes the same inline and saved forms. [Typed report builders](query-dsl.md#constructed-report-builders)
return the native ordered data before explicit rendering.

Preflight checks known key types even in skipped branches. Missing fields fail
when executed; missing or incompatible keys point to the selector, while
non-array input and non-record rows point to the call. Child failures retain
their query or source locations. Grouping evaluates rows once and charges scans,
comparisons, nominal metadata, new containers/slots and copied keys/labels under
the active budget. Rows move into items without another deep copy. Snapshots,
Python conversion and rendering retain that allowance and existing completeness
rules. A grouping error leaves CLI stdout empty after earlier reports too.

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
Use it for an edit followed by whole-document output:

```zirium
do filter(op("arith.addi"))
   | set_attr("analysis.tag", "review");

emit
```

The edit remains visible to later statements, and the final `emit` prints the
complete edited document once. A `do` statement requires its trailing
semicolon, including at the end of a program. Explicit emitters inside its query
still emit; `do` suppresses only the implicit result.

## Checking query results

`check` fails evaluation when the current stream is empty. `check(n)` instead
requires exactly `n` items, including zero. Either form accepts a message as
`check("message")` or `check(n, "message")`; a failure includes that message
after the expected and actual cardinalities. A successful check leaves the
stream unchanged, so it can appear anywhere in a pipeline. Prefix a check
statement with `do` when only the exit status matters:

```zirium
do filter(op("linalg.matmul")) | check("expected a lowered matmul");
do filter(op("bufferization.alloc_tensor"))
   | check(0, "tensor allocations must be eliminated");
```

This makes a Zirium program usable as a structural MLIR test: the first check
requires a matmul, the second rejects any remaining tensor allocation, and the
CLI exits unsuccessfully if either condition is false. As with other evaluation
errors, output is buffered and stdout stays empty when any check fails. Use
`check(n)` after `unique` when duplicate navigation paths should not count as
separate matches.

For a check program whose statements need not use `do`, suppress every result
at the CLI boundary:

```sh
zirium --silent --strict -f checks.zirium input.mlir
```

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

Map rows and columns are sorted lexically, except that ranked maps retain their
row order. Missing cells and null values are blank. Other scalars are strings,
numbers, or booleans. Empty inner maps contribute blank rows; if every inner
map is empty, only the `Key` column remains.

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

If `main` calls `helper` twice, its body histogram counts both calls. The
reachable histogram also counts the helper's body once. Recursive calls still
terminate because each static operation is counted once. These counts describe
the represented code, not execution frequency.

`reachable` includes seeds, their explicit nested bodies, and transitive SSA
definitions. Block arguments are boundaries. Direct `func.call` and registered
`call_like` operations resolve their `callee` symbol and include the target's
body. The target declaration itself is not added merely because it is called.
Registered `func_like` operations participate in symbol lookup. Supported
`cf.br` and `cf.cond_br` edges include their target blocks without widening
the selection to the enclosing function. The result is a source-ordered set.

Named functions and direct calls can represent lambdas under these rules.
Indirect calls and dialect-specific captures need additional reference support.
An operation with unknown reference semantics is retained as a leaf: traversal
skips its regions, operands, successors, and symbols. Already-selected operations
inside its regions remain selected. Registering its edges may expand the result.

`--strict` rejects unknown reference semantics by operation name. Unresolved or
external callees, malformed known calls, and declared unsupported references
are errors in either mode.

As with body counts, compact assembly can omit implicit operations from the
represented structure. `reachable` does not synthesize those operations.
Use `closure` when retaining enclosing scopes for a printable fragment;
`reachable` deliberately stops at block arguments for analysis. `closure`,
including inside `fixpoint`, remains strict about unknown reference semantics.

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
| `slice`, `backward_slice` | The selection and its transitive SSA definitions, stopping at block arguments. |
| `forward_slice` | The selection and its transitive SSA users. |
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

Fragments may omit required SSA definitions. Select dependencies explicitly;
printing a fragment does not verify that it is valid standalone MLIR.

Use `backward_slice` to inspect a computation without expanding the whole
function when an operand is a function input. The shorter `slice` spelling is
an alias:

```zirium
# Follow the first returned value, for example logits rather than cache outputs.
filter(op("func.return")) | defs(0) | backward_slice
```

`backward_slice` follows explicit SSA operands only. It does not expand callees,
successors, or region bodies, and does not infer which operands affect individual
results of a multi-result operation. Generic quoted operations need no
registration for this traversal; recovered unparsed operations and invalid
operands are errors. Use `closure` when retaining complete scopes and supported
symbol dependencies is the desired behavior.

`forward_slice` follows uses of every result of the selected operations, then
uses of every operation it reaches, until no new users remain. It retains the
seed and returns each operation once in source order:

```zirium
filter(op("stablehlo.dot_general")) | forward_slice
```

Like `users`, it follows explicit SSA uses, including successor arguments. It
does not infer control-flow, symbol, region, memory, or effect dependencies.

`defs(index)` still maps a block argument directly to its owning operation. If
the selected return operand is itself a function argument, that navigation
selects the function; `backward_slice` does not undo that selection. Starting
`backward_slice` at the return instead retains the return and stops at its
argument operands.

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

Closure requires registered reference semantics. Unregistered operations and
unsupported symbol or successor references produce errors.

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
iterations, and dependency, ancestor, and subtree visits. These limits do not
bound time or memory bytes. CLI callers can set `--max-work N` and
`--max-items N`; Rust callers can use `Query::evaluate_with_limits` and
`EvaluationLimits`. A limit error follows the usual no-stdout CLI contract.
The initial stream contains every operation, so it must fit `max_items` even
when the first stage filters down to a small selection. The work limit does not
bound MLIR parsing, printing, or the total bytes buffered across emissions.

`--max-results N` adds a whole-program allowance for produced and copied native
entries, binding reads, emission snapshots, nested JSON containers/fields and
rendered report storage. It is separate from the per-stream `--max-items` ceiling.
With this option, copied strings and encoded JSON/Markdown also consume work;
escaped Markdown reserves a byte bound before construction. Rust callers can use
`Query::evaluate_with_program_limits` with `zirium::query::ProgramLimits`. Existing
`EvaluationLimits` APIs keep their established work and stream limits for
programs without comprehensions. Comprehension requests also use the default
aggregate report allowance described above. An
aggregate failure carries a query range and follows the same no-stdout CLI rule.
Native MLIR printing remains a caller-owned emission boundary.

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

This prints the adds, then their users. Each emission captures the document
before later edits. A statement emits its final result implicitly unless it
ends in an explicit emitter, including inside a final group. Thus `emit | emit`
prints twice and `emit` prints once.

`format` applies source-aware layout to the current fragment and also passes
the selection onward. By default it retains the source assembly form for each
operation, including custom operation syntax, SSA names, aliases, and comments
when retained syntax is available:

```zirium
format
```

Formatting options use named arguments. `width` controls the preferred line
width and `indent` controls spaces per indentation level. Both must be positive;
`indent` may not exceed 256.
Use MLIR's quoted generic assembly form explicitly when source spelling should
be replaced:

```zirium
format(assembly = "generic", width = 100, indent = 2)
```

For edited hybrid documents, unchanged source is reused and dirty operations
prefer registered custom printers. A fragment or structurally edited document
without reusable source prefers registered custom syntax and falls back to
generic syntax. Unknown recovered custom operations remain verbatim in source
mode.

Nested queries neither reset to `input` nor emit implicitly. A fixed-point body
ending in `emit` prints each iteration; the enclosing statement still follows
the final-emission rule above.

`count` prints the stream's size followed by a newline. It is terminal;
no stage may follow it. To emit a fragment and then count it, use `emit | count`.

Selected fragments normally retain every attribute on the ancestor operations
needed as enclosing shells. CLI `--fragment-scope minimal` instead prints those
shell-only ancestors in generic syntax, retaining properties and structural
identity/signature attributes while omitting incidental attributes and
locations. Selected operations and their bodies keep all metadata. The option
affects selection output only.

Minimal scope does not supply missing SSA definitions, callees, or terminators.
The fragment may still be incomplete.

`json` emits streams as arrays and maps as objects, then passes the result onward.
For value streams, the array contains strings. For operation streams, each
entry contains the operation name, an object of attribute spellings, and
`operand_types` and `result_types` arrays. Incomplete type information may appear
as a placeholder or null. Zirium cannot read this inspection format back as MLIR.
Like `emit`, a final `json` suppresses the implicit final emission.

Statement outputs and results from multiple input files are concatenated
without an automatic separator. Use `print` when a report needs headings,
spacing, or a delimiter. Counts are plain lines. The CLI buffers all emissions
across all input files until processing succeeds: a query, evaluation, or
printing error produces no standard output. Input files are never overwritten.
Rust callers use the `Query::evaluate` emission callback and can choose their
own buffering policy. Each edit stage commits its own transaction. A later
library evaluation failure preserves earlier committed edits and callbacks;
saved scalar results retain their values, while retained native handles observe
the edited document.

Pass `--jsonl` to emit one compact JSON object per result in input and
statement order. Each record has `document` and `result` fields. A `json`
emitter supplies the JSON value in `result`. Text and Markdown are strings with
escaped newlines, counts are numbers, and operation selections are MLIR strings.
The same buffering rule applies: any later failure leaves stdout empty.
`--ndjson` is an alias.

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
           | "format" [ "(" format-option { "," format-option } ")" ]
           | "tally" | "map_by" "(" query "," query ")"
           | "sort" | "sort_by" "(" query ")" | "value" | "reverse"
           | ("head" | "tail") "(" integer ")"
           | "min" | "min_all" | ("min_by" | "min_all_by") "(" query ")"
           | "max" | "max_all" | ("max_by" | "max_all_by") "(" query ")"
           | "input" | "filter" "(" predicate ")"
           | ("defs" | "users") [ "(" integer ")" ]
           | "parent" | "children" | "closure" | "slice"
           | "backward_slice" | "forward_slice" | "reachable"
           | "root" "(" predicate ")" | "subtree" | "unique"
           | "attr" "(" string ")" | "names" | "result_types" | "operand_types" | "json"
           | "fixpoint" "(" query ")" | "(" query ")"
           | "set_attr" "(" string "," string ")"
           | "remove_attr" "(" string ")" | "emit" | "count"
format-option = "assembly" "=" ("source" | "generic")
              | ("width" | "indent") "=" integer
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

Query and predicate nesting share a 64-level limit. Materialized objects and
arrays have a separate 64-level limit, including structures built through bindings.
Nested contents count toward `--max-items`, and copying saved results counts
toward `--max-work`. Flat boolean chains, pipelines, and set chains do not
consume a nesting level per term.
The CLI reports query errors with a byte offset, line, column, and source
caret. Program-file positions include leading whitespace and comments.

Shared parser errors use Zirql's categories for the existing work-limit,
result-limit and unsupported-null wording. All other shared errors retain their
full message. Their diagnostic range remains the zero-length range at the
responsible shared range's start; `QueryError::position` uses that same byte
offset. Native runtime errors retain their existing messages and locations.

For implementation timing and scaling measurements, see
[query profiling](architecture/query-profiling.md).

## Semantic diff input

`--diff BEFORE AFTER` changes the initial stream from operations to change
records. `change("added")` selects a primary kind and `changed("operands")`
selects a direct changed field. Ordinary predicates inspect the after endpoint,
or the before endpoint for removals. Use `before` or `after` before operation
navigation. The two sides remain distinct even for an empty selection, and diff
programs reject edits. Nested expressions keep their current input; explicit
`input` resets to every change record. Projected operations support the ordinary
graph traversals and same-side fixed points. Those bodies and traversals share
the active evaluation allowance. Diff `emit` is a no-op; `json`, `markdown` and
`print` transform the value written once at the end of a query statement.
Bindings and `do` suppress that statement output. Diff mode retains its legacy
work and per-stream limits and rejects `--max-results`. Change streams support keyed
ordering and `map_by`, and the document labels are available as
`before_document` and `after_document` for interpolation. See
[semantic diff](semantic-diff.md) for the full policy.

## Correlated constructed reports

`project(subject, {fields})` evaluates ordered fields for each native input
operation or capture record. The subject is local to the body: a singleton
operation selection or one capture record. Rows follow
input order, and duplicate subjects or equal projected names remain separate
rows. Native handles require explicit scalar projections such as `names`,
`attr` or `count`.

A small projection fits in the existing inline query argument:

```sh
zirium 'filter(op("func.func")) | project(f, {"name": f | attr("sym_name") | one, "n": f | children | count}) | json' examples/cli/function-report.mlir
```

`one` extracts exactly one projected string; zero or multiple values fail at
that stage. A projected string sequence remains an array when `one` is omitted.
Native counts remain unsigned counts in ordinary queries and become checked
signed integers at the constructed boundary.

`let name = expression;` adds immutable scalar, constructed or native bindings.
`report expression;` requests the established result emission. A final `| json`
suppresses duplicate implicit output and remains pass-through. Bare saved names
`let`, `report`, `project`, `sum` and `one` keep their legacy meaning. Explicit
new value bodies use shared scalar rules; ordinary JSON literals retain their
interpolation and null behavior.

This complete program uses local literal arguments:

```text
let operation_name = "arith.addi";
let minimum_ops = 2;
let threshold = require(minimum_ops >= 0, minimum_ops, "minimum_ops must be nonnegative");
let functions = input | filter(op("func.func"));
let rows = functions | project(f, {
  "name": f | attr("sym_name") | one,
  "operations": f | children | subtree | count,
  "matches": f | children | subtree | filter(op(operation_name)) | count,
  "kinds": f | children | subtree | names | unique
});
let selected = [r for r in rows if r.operations >= threshold];
report {"functions": selected,
        "total_operations": sum([r.operations for r in selected]),
        "total_matches": sum([r.matches for r in selected])};
```

The [function fixture](../examples/cli/function-report.mlir) produces `double`
and `mix`, with body counts 2 and 3, one matching addition each, and operation
kinds in first-seen order. The totals are 5 and 2. `minimum_ops = 4` produces an
empty function array and zero totals. Run the
[saved program](../examples/cli/function-report.zirql) with `-f`, or pass the
same text inline. Both modes use original query locations and one request
allowance.

`length`, `sum`, `forall` and `exists` reduce materialized arrays. Integer sum
checks overflow; empty identities are 0, 0, true and false.
`require(condition, value, "message")` preflights both operands, then evaluates
its value only when the condition is true. On false it fails with the message.
An unchosen value still receives scope, known-type and purity validation.

Projection over an empty selection validates every field before returning an
empty array. Known record schemas survive empty results, so later invalid sums
or field operations are refused before native execution. Effects and emitters
are forbidden in projection fields. Parsing, nested native queries, value
generators, copies and rendering retain one work/result/nesting allowance.
CLI output remains buffered until success; library callbacks retain their
existing explicit buffering policy.

## Bounded SSA-use paths

Find one shortest native witness from each selected operation to each matching
sink within an explicit edge bound:

```sh
zirium 'filter(op("arith.muli")) | paths(users, to=op("func.return"), max_depth=2) | json' examples/cli/function-report.mlir
zirium -f examples/cli/bounded-paths.zirql examples/cli/function-report.mlir
```

The [saved program](../examples/cli/bounded-paths.zirql) uses the same short
expression. Both commands return the `mix` witness with steps
`[arith.muli, arith.addi, func.return]`. A bound of one gives no witness for this
pair. `users` is one direct SSA-use connection, following result indices and
retained use-site order; operand and successor-argument uses both select their
direct user operation.

The required `max_depth` is a nonnegative integer literal measured in edges.
`to` uses ordinary operation predicates, including parameters and shared
`not`/`and`/`or` control. It does not extend the native match guard language.
Zirql owns the breadth-first traversal. Distinct roots keep input identity order,
sinks keep discovery order, and neighbor order chooses equal-length ties.
Repeated roots and cycles cannot duplicate a source/sink pair. Searching
continues through matching sinks.

Each native capture has `source`, `sink` and `steps`. Endpoints are singleton
operation selections; steps are an ordered native operation sequence containing
both endpoints. A matching source has a zero-edge witness whose steps remain
`[source]`, including in JSON and Python. Existing singleton captures retain
their behavior even when a user names a capture `steps`.

Field selection and `project` retain native handles until explicit scalar
projection, for example `p.steps | names`. Handles keep their document owner
and source coordinates; owner/stale checks still apply. Paths are read-only and
unavailable on the current diff surface. No endpoint implicitly crosses sides.

Path preflight covers all target branches and types, including empty input.
The same aggregate program budget pays for scans, identity comparisons,
traversal slots, native copies, witnesses and rendering. Use `--max-work`,
`--max-items` and `--max-results` to bound a request. Exhaustion fails the whole
request. No witness within this bound does not prove global disconnection.
See [Rust and Python builders](query-dsl.md#bounded-native-path-builders).

## Native SSA captures

The development Git API supports ordered native captures with `match`. This
program selects additions whose two operands are the same SSA value, then
projects a scalar report:

```text
let matches = match {
  from add in (input | filter(op("arith.addi")));
  bind lhs = operand(add, 0);
  bind rhs = operand(add, 1);
  where same_value(lhs, rhs);
};
let rows = matches | project(m, {
  "function": m.add | root(op("func.func")) | attr("sym_name") | one,
  "operation": m.add | names | one
});
report {"matches": rows, "count": length(rows)};
```

With the original [function fixture](../examples/cli/function-report.mlir), the
result is `{"matches":[{"function":"double","operation":"arith.addi"}],"count":1}`.
Run the [saved capture program](../examples/cli/native-captures.zirql) with
`zirium --program-file examples/cli/native-captures.zirql examples/cli/function-report.mlir`,
or pass the same text inline.

For native descriptions alone, the short form needs no declarations:

```sh
zirium 'match { from add in (input | filter(op("arith.addi"))); bind lhs = operand(add, 0); bind rhs = operand(add, 1); where same_value(lhs, rhs); } | json' examples/cli/function-report.mlir
```

`from` enumerates a finite native selection in order. Later clauses see earlier
captures; a name is absent from its own source or binding expression. Captures
cannot shadow an ambient binding or repeat a name. Dependent `from` clauses
enumerate in depth-first order, with the rightmost source varying fastest.
`bind` requires exactly one native value. Records retain all capture fields in
written order, including operations and exact SSA handles.

`operand(operation, index)` requires one operation and a zero-based operand
index. It refuses an absent, unresolved or invalid operand. `same_value(left,
right)` compares the complete SSA identities, including result or block-argument
position. Two results from the same defining operation are different values.
Names, source spelling and rendered descriptions do not determine equality.
Guards combine with the usual Boolean operators; unavailable required operands
and unknown final truth fail at their query location.

`m.add` selects the native operation field of a capture record. Use explicit
projections such as `names`, `attr` and `one` before constructing scalar reports.
An empty match retains its field schema, so an invalid field, type or effect is
still refused. Every clause and Boolean branch receives scope, type and purity
validation before enumeration. Candidate work is charged before filtering;
matching, field copies, projection and rendering share the request allowance.

Rust retains native capture rows and SSA handles; Python returns dictionaries
containing the existing semantic wrappers. The explicit JSON boundary describes
SSA values with `kind`, zero-based `index` and optional `type`, alongside the
existing operation description. See [native capture builders](query-dsl.md#native-capture-builders).
This matching surface is read-only and does not support diff queries or rewrites.
