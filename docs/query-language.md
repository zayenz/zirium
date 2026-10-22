# Zirium query language

These examples use Zirium 0.6.1. Repository-relative inputs are in the
[`v0.6.1` checkout](https://github.com/zayenz/zirium/tree/v0.6.1);
see [getting started](getting-started.md) for installation.

A query transforms an ordered stream. It starts with every operation
in the input document. A pipe passes the current stream to the next stage:

```zirql
filter(op("arith.addi")) | users | filter(has_attr("analysis.tag"))
```

This finds `arith.addi` operations, follows their direct users, and keeps users
with an `analysis.tag` attribute. Navigation replaces the stream and preserves
duplicates; edits keep the selection. `closure`, `backward_slice`,
`forward_slice`, and `reachable` produce source-ordered sets. `slice` is an alias
for `backward_slice`. Use `unique` to remove duplicates from other streams.
Set union keeps first appearance from the left operand, then the right.
Intersection and difference deduplicate in left-operand order.

Zirium supplies operation selection, SSA navigation, dialect semantics and edits.
Its scalar expressions, constructed collections, patterns and pure functions use
[Zirql](https://zayenz.se/zirql/guides/library-reference/). The library reference
owns their shared signatures and semantics. This guide supplies MLIR recipes,
native projections, accepted syntax and output/edit policy. Start with
[calculated values](#calculate-with-shared-values), then use the local sections on
[comprehensions](#positional-comprehensions),
[optional fields](#optional-fields-and-lazy-defaults) and
[functions](#typed-recursive-functions) as needed. The Zirql reference provides
further explanation of the shared evaluator and its use in other hosts.

The development checkout also uses Zirql's common program reader. It collects
function declarations before reading their bodies and introduces parameters and
`let` names at their written positions. Zirium supplies native expressions and
types, legacy `name = query` bindings, `do`, and emission policy. Parsing only
builds syntax; edits run during evaluation. The same request budget covers
reading and execution.

For Rust and Python builders with native results, see the
[structured query guide](query-dsl.md).

`input` and a final `emit` are implicit. These programs print the same document:

```zirql
# An empty program is valid.
```

```zirql
input
```

```zirql
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

```zirql
param operation_name: string;
param minimum_ops: int;
filter(op(operation_name)) | head(minimum_ops) | names | json
```

Pass each required value with a repeated `--param name=value` option. Inline
programs and program files use the same admission rules:

```sh
zirium --param operation_name=arith.addi --param minimum_ops=2 \
  'param operation_name: string;
   param minimum_ops: int;
   filter(op(operation_name)) | head(minimum_ops) | names | json' \
  examples/cli/function-report.mlir

zirium --param operation_name=arith.addi --param minimum_ops=2 \
  -f examples/cli/function-report-parameters.zirql \
  examples/cli/function-report.mlir
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

```zirql
let adds = filter(op("stablehlo.add"));
let matmuls = filter(op("stablehlo.dot_general"));

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

## Calculate with shared values

Bindings can also hold ordinary values. Save this program as `counts.zirql`:

```zirql title="counts.zirql"
let samples = [
  {"label": "pear", "count": 2},
  {"label": "apple", "count": 1},
  {"label": "ash", "count": 2},
  {"label": "birch", "count": 3}
];
let selected = [row for row in samples if row.count > 1];
sort_by({row.label: row.count for row in selected}, value, descending=true)
```

Run it against the small example document:

```sh
zirium -f counts.zirql examples/cli/function-report.mlir
```

```json
{"birch":3,"pear":2,"ash":2}
```

`selected` contains pear, ash and birch in their input order. The dictionary uses
their labels as string keys and their counts as values. `descending=true` puts
the largest count first. Pear and ash both have count 2, so their incoming order
stays pear, then ash. Sorting retains the dictionary kind. Zirium prints this
JSON with indentation.

Constructed lists, tuples, sets and dictionaries can also be dictionary keys
or set members. For example, `[{"ash", "apple"}, {"apple", "ash"}] | tally`
produces one set key with count 2. Set identity ignores member order.
Zirium renders a dictionary with non-string keys as
`{"kind":"dictionary","entries":[{"key":["ash","apple"],"value":2}]}`.
Keys and values are converted recursively; sets and tuples render as arrays.
The language keeps the original set key for lookup and equality.

The shared [collection reference](https://zayenz.se/zirql/guides/library-collections/),
[operators](https://zayenz.se/zirql/guides/library-operators/) and
[presence helpers](https://zayenz.se/zirql/guides/library-presence/)
define iteration, access, arithmetic and missing-value behavior. Zirium spells
absence `null`. Strings in shared expressions are literal: `"{row}"` keeps its
braces. The separate [JSON-template stages](#building-json-structures) support
interpolation.

This program calculates only from `samples`; it does not inspect the document.
The [shared collection rules](https://zayenz.se/zirql/guides/values/#dictionaries-and-collection-order)
provide more detail on dictionary order and stable sorting. The same expressions can consume
other hosts' data; the [CSV guide](https://zayenz.se/zirql/guides/csv/) demonstrates
filtering and reporting over rows. MLIR operation projections remain specific
to Zirium.

Use native selections to supply values from MLIR. For example,
`filter(op("func.func")) | attr("sym_name")` projects names, while a saved
operation selection retains its identities and source coordinates. The
[projection section](#correlated-constructed-reports) constructs a report from those
operations without a text or JSON conversion.

## Counting values and building maps

Use [tally](https://zayenz.se/zirql/guides/library-pipelines/#tally) to count
values or native subjects. Project `names` first to count operation spellings;
unprojected operations retain document identity. String-key histograms emit
ordered JSON objects, and `json` can emit a map explicitly.

To map function names to their number of matmuls:

```zirql
let matmuls = filter(op("stablehlo.dot_general"));

matmuls | root(op("func.func")) | attr("sym_name") | tally | json
```

Each matmul contributes its enclosing function's name once. Keep the duplicates:
adding `unique` would lose the information needed for the count. Functions
without matmuls are absent. On `examples/cli/stablelm-decode.mlir`, this produces
`{"main": 19}` when run with `--preset stablehlo`.

For a separate analysis of each function, use `map_by(key_query, value_query)`:

```zirql
let functions = filter(op("func.func"));

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

The shared [ordering reference](https://zayenz.se/zirql/guides/library-ordering/)
and [collection stages](https://zayenz.se/zirql/guides/library-pipelines/)
define sorting, extrema, bounds and reversal for values and native inputs.
Bounds truncate an already computed stream; they do not avoid parsing the file
or evaluating earlier stages.

Zirium's native `sort`, `min` and `max` use the intrinsic string projection on
value streams; dictionaries sort by key. Operations need an explicit selector:

```zirql
filter(op("func.func")) | sort_by(attr("sym_name")) | head(10)
```

Native string selectors must return exactly one value. Ordinary constructed and
JSON data can supply compound, Boolean or absent keys; counts retain their
unsigned range. Native extrema keep selected handles, their owner and result
kind, including empty selections. The `_all` forms retain every tied native
subject. Use `reverse` after a native sort for descending order.

`map_by` and `tally` dictionaries support `sort_by(value)`, bounds and reversal
through saved results, nested reports, JSON and Markdown. Bare `value` selects
the whole dictionary value. Navigation retains its domain order; dictionary
operations follow the shared contracts linked above.

## Building JSON structures

Zirium has two ways to build report data. A plain object or array used as a
pipeline stage is a JSON template: its strings and keys interpolate saved
bindings. A shared value expression, selected explicitly with `report`, treats
strings literally and supports arithmetic, computed keys and comprehensions.
For example:

```zirql
let n = count;
{"label": "count={n}"} | json;
report {"label": "count={n}"} | json;
```

On a document containing one operation, the first result has label `count=1`;
the second has label `count={n}`. To store the count as a number with shared
value rules, write `report {"count": n} | json`.

Use a JSON template to assemble saved query results with interpolated labels.
Use `report expression`, a value initializer such as `let x = 1 + 2;`, or a
comprehension when calculating new values. Projection fields and function
bodies also use shared expressions. In particular, `let x = {"label": "{n}"};`
stores the literal string `{n}`. The JSON-template rules below apply to plain
object and array stages; the following sections cover shared values.

Object and array literals combine saved results into a larger report:

```zirql
let functions = filter(op("func.func"));
let n = functions | count;
let counts = functions
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
with `{` or `[`; scalars occur inside objects and arrays. Objects and arrays
may be nested, and saved results can supply their values:

```zirql
let totals = names | tally;
let report = {"totals": totals, "complete": true};
[report, {"note": "Counts include container operations"}] | json
```

Objects and arrays also print as JSON through implicit output or `emit`.
`count` counts outer object keys or array elements. `markdown` accepts literal
objects and arrays of record rows as tables; nested cells use compact JSON. Literal construction is not a selection
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

```zirql
let labels = names;
[{"name": n, "label": "data:" ++ n} for n in labels] | json
```

The output is:

```json
[
  {
    "name": "alpha",
    "label": "data:alpha"
  },
  {
    "name": "beta",
    "label": "data:beta"
  }
]
```

Repeated `for` clauses can depend on earlier bindings. The rightmost generator
varies fastest; each `if` filters its own clause before later clauses run:

```zirql
let labels = names;
[{"name": n, "label": "data:" ++ s}
 for n in labels
 for s in [n, n ++ "!"] if s != "beta!"] | json
```

The MiniZinc spelling uses the same generators and evaluator. For example,
`[(a, b) for (a, b) in [(1, 2), (3, 4)] if a < b]` and
`[(a, b) | (a, b) in [(1, 2), (3, 4)] where a < b]` produce the same tuples.
Both forms accept shared patterns and dependent generators.

The rows have labels `data:alpha`, `data:alpha!`, and `data:beta`, in that order.
A generator sees earlier names in its source, then installs its binder for its
filter and the head. Duplicate binders, shadowing, forward references, and
undefined names fail at their query locations. `_` discards a value and cannot
be read. Nested comprehensions retain their own array shape.

Inside a shared value expression, dictionary keys are expressions.
`{"n": n}` uses the literal key `n` and the bound value of `n`; `{n: 1}`
uses the bound string `n` as its key.
Strings are decoded literals: `"{n}"` contains braces and the letter `n`.
Plain object and array stages use the [JSON-template rules](#building-json-structures);
`report`, shared value initializers and function bodies use literal strings.
`for` and `if` remain ordinary binding names elsewhere, including a literal
such as `[for]` when `for` is bound.

A scalar expression can be a query result directly. An immutable binding lets
later expressions reuse the value:

```sh
zirium '1+2' examples/cli/arithmetic.mlir
zirium 'let n=1; n+2' examples/cli/arithmetic.mlir
```

Both commands print `3`. The same expressions work in program files and through
`Query::parse`; no function declaration or `report` prefix is required.

Equality uses `==` in both bare expressions and `report` expressions:

```sh
zirium 'let n=1; n == 1' examples/cli/arithmetic.mlir
```

This command prints `true`. Legacy bindings use a single `=`, as in
`name = expression;`.

The [library reference](https://zayenz.se/zirql/guides/library-reference/)
organizes the shared expression forms and their contracts. Use its
[scalar and string entries](https://zayenz.se/zirql/guides/library-strings/)
for exact matching and conversion, and [operators](https://zayenz.se/zirql/guides/library-operators/)
for numeric limits, type compatibility and lazy Boolean evaluation.

Project a domain name first, then apply the helpers to that string:

```zirql
let labels = filter(op("func.func")) | attr("sym_name");
[n for n in labels if starts_with(n, "m") and ends_with(n, "x")] | json;
```

The native `op(...)` and `string_attr_eq(...)` selectors retain their existing
host semantics. These helpers also work in typed function bodies and native
expression arguments. Inline and file programs use the same evaluator.

Use an explicit projection such as `names`, `attr`, or `count` before constructing
scalar reports from operations. Native operation, type, and attribute handles
retain their identity and cannot become comprehension values through their
printed spelling. Saved JSON arrays and objects can supply scalar or structured
values; JSON `null` projects to absence, while integers outside `i64` fail.

### Sets and computed dictionary keys

A colon at a brace body's own level selects a dictionary; otherwise a
nonempty body is a set. `{3, 1, 3}` keeps `3, 1`, and
`{n mod 3 for n in 1..5}` keeps `1, 2, 0`. Repeated `for` clauses and filters
can supply dictionary keys as well as values:

```zirql
{k: {n mod 3 for n in 1..5}
 for w in ["pear", "apple"]
 for k in [w, w ++ "!"] if ends_with(k, "!")}
```

The dictionary keeps `pear!`, then `apple!`. Constructed values, including
lists, tuples, sets and dictionaries, can be keys or values. A structurally
duplicate key fails before evaluating its value. `.field` uses a static string
label and `[key]` accepts a computed key. Strict access fails on a missing key,
while `lookup` is optional.

The [collection entries](https://zayenz.se/zirql/guides/library-collections/#sets)
define empty sets and dictionaries, structural identity and insertion order.
[Infix operators](https://zayenz.se/zirql/guides/library-operators/#equality)
retain their compatible-type and numeric admission through ordinary native
wrappers. Opaque MLIR leaves retain exact document, source and diff-side identity.
Empty nominal sets need host evidence. Ordinary native sets and general native
dictionaries are not available in Zirium's current host representation.

The [saved collection report](../examples/cli/brace-collections.zirql) combines
these forms with strict access, nesting and sorting:

```sh
zirium --program-file examples/cli/brace-collections.zirql examples/cli/arithmetic.mlir
```

Inline programs use the same parser and evaluator. Collections inside a
function or control block keep their own braces, for example
`if true { {3, 1, 3} } else { set<int>() }`. Plain JSON objects and arrays
retain their established interpolation and native-result insertion behavior.
Selected shared collection forms use the aggregate request budget, including
parsing, candidates before filters, construction, copying and rendering.

### Optional fields and lazy defaults

Zirium spells absence `null` and uses the shared
[presence, lookup and lazy-default contracts](https://zayenz.se/zirql/guides/library-presence/).
A stored null remains distinct from omitted storage:

```zirql
let info = {"id": 4, "stored": null};
report {"value": lookup(info, "missing") default 8,
        "stored": has_key(info, "stored"),
        "missing": has_key(info, "missing"), "info": info};
```

The result has `value: 8`, `stored: true` and `missing: false`; `info` contains
`id` and `stored` only. Stored absence renders as JSON null. The reusable
`examples/cli/optional-report.zirql` also includes positional lookup, a native
count and a skipped division error. Run it against the existing input:

```sh
zirium --program-file examples/cli/optional-report.zirql examples/cli/function-report.mlir
```

Inline programs, file programs and the [Rust/Python report builders](query-dsl.md#optional-report-values)
use the same evaluator. Legacy plain JSON literals keep their output shape.
External inputs remain required int/string/bool; null is a report value, not an
optional parameter transport. Native operation identities stay document-owned. Nominal members and keyed
arrays use Zirql's tagged JSON fallback to retain names, axes and missing cells.

Rust callbacks receive `QueryOutput::Native(NativeValue::Constructed(report))`
before rendering. `report.value()` borrows the ordered Zirql value, including
record field order. `Clone` shares this immutable result. Prepared evaluator reads share immutable values when available;
owned materialization and emission snapshots use charged copies. `report.render_json_with_budget()`
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

Shared value calls work in both binding initializers and final expressions.
For example, these commands print `7` and `2`:

```sh
zirium 'let numbers=[3,1,3]; sum(numbers)' examples/cli/function-report.mlir
zirium 'let numbers=[3,1,3]; numbers | sort | unique | count' examples/cli/function-report.mlir
```

Saved constructed collections and function results use the shared
[collection stages](https://zayenz.se/zirql/guides/library-pipelines/).
String arrays also accept `filter(starts_with("prefix"))`. Native selectors such
as `sort_by(names)` retain their domain projections; constructed
`sort_by(rows, "score")` uses a literal field selector. Zirql distinguishes them
by a comma at the call's own argument level. Nested expressions, strings and
comments keep their normal meaning.
For example, this uses both routes with the existing function-report fixture:

```sh
cargo run --locked --bin zirium -- '
let ordered = sort_by([{"score":2},{"score":1}], "score");
report ordered | count;
sort_by((names # comma, and closing delimiter ) stay in the comment
)) | names
' examples/cli/function-report.mlir
```

The development adapter uses Zirql's common syntax recognition before reading
expressions and suffixes. Plain legacy JSON/interpolation keeps its established
parsing allowance; common words inside strings do not trigger value-program
allocation charges. Shared parsing and checked lookahead use the active request
budget. Native stage precedence, emission/edit boundaries, MLIR and diff meaning
remain Zirium rules.

Names depend on their position. Saved names such as `for`, `if`, `report`,
`project`, `lookup` and `default` keep their contextual uses. Native operations,
input/document bindings and literal/operator words retain their existing
exclusions. Function declarations also reject Zirql's common operation names.
The shared reader's quoted identities remain available where the frontend
admits quoted names; its ASCII spelling check does not replace host admission.

## Transform positional reports

These forms require the development checkout, which pins Zirql Git revision
[`49c30dac58ed2614fb9d6ba97dafe1432fa7a9cc`](https://github.com/zayenz/zirql/commit/49c30dac58ed2614fb9d6ba97dafe1432fa7a9cc).
The published 0.6.1 integration predates them. Inline and `--program-file`
programs use the same reader and evaluator.

```sh
cargo run --locked --bin zirium -- '
{
  "pairs": [n for case present(n) in [null, 1, 2]] | map(n => n + 1) | windows(2),
  "sizes": [{"id": 1}, {"id": 1}, {"id": 2}] | runs_by("id") | map(length(_.items)),
  "changed": {"id": 1, "amount": 2} with {amount: 3},
  "keys": align([{"id": 1}], [{"id": 2}], by="id") | map(_.key)
} | json' examples/cli/function-report.mlir
```

The shared entries describe [map and flat_map](https://zayenz.se/zirql/guides/library-pipelines/#map),
[selective generators](https://zayenz.se/zirql/guides/library-collections/#comprehensions),
[windows, runs and alignment](https://zayenz.se/zirql/guides/library-grouping/)
and [record updates](https://zayenz.se/zirql/guides/library-operators/#record-update).

These are operations on positional report data. Native selections retain their
host kinds and require an explicit projection or declared positional view.
Zirql owns expression traversal, normalization structure and evaluation;
Zirium supplies native stages, document ownership, source coordinates, stale
handle checks and effect admission. See the
[builder examples](query-dsl.md#transform-report-data) for Rust and Python.

## Sort constructed rows and select extrema

Constructed calls use the shared [ordering and extrema contract](https://zayenz.se/zirql/guides/library-ordering/),
including literal field selectors, descending ties, structural key order and
optional single extrema. In this frontend, empty constructed extrema are
`null`; use `min([]) default 0` for an explicit fallback. Keyed inputs require an
explicit positional projection before reordering cells.

The [report example](../examples/cli/report-ordering.zirql) uses scores
`[2,1,2,1]` on rows A, B, C and D. Ascending IDs are B, D, A, C;
descending IDs are A, C, B, D. Min selects B and max selects A:

```sh
zirium -f examples/cli/report-ordering.zirql examples/cli/function-report.mlir
```

Native operation and dictionary stages use the same key order while retaining
their native selections, ownership and default projections described above.
CLI output stays buffered until the whole program, including rendering, succeeds.

## Stable record groups

Use [group_by](https://zayenz.se/zirql/guides/library-grouping/#group_by) to
collect positional report rows by a field. Project native subjects into those
rows explicitly; keyed arrays need an explicit positional view.

For the [function fixture](../examples/cli/function-report.mlir), this short
program groups functions by their represented `arith.addi` count:

```zirql
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

Grouping uses the request's continuing budget, including snapshots, Python
conversion and rendering. Zirium retains native owners and completeness rules;
a grouping error leaves stdout empty after earlier reports too.

<a id="writing-reports-with-markdown-and-text"></a>

## Markdown reports

Use `markdown` to emit a table from a histogram or a map of histograms.
Use `print("text")` for headings and explanatory text between results:

```zirql
let functions = filter(op("func.func"));
let n = functions | count;

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

```zirql
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

```zirql
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

Ordinary print and JSON strings and keys delegate fragment expansion to Zirql.
Computed object entries use its shared key-then-value constructor, which rejects
a duplicate key before evaluating that entry's value. Zirium retains binding
admission, saved JSON snapshots, JSON numbers and null, escaping and emission.
Expansion bytes, object/array storage and nested construction share the active
request limits. A limit refusal returns no partial result and keeps the enclosing
print or literal location.

Zirium delegates ordinary data formatting to Zirql's
[Markdown renderer](https://zayenz.se/zirql/guides/library-rendering/#render_markdown).
Scalar counts use a paragraph; scalar collections use a `Value` table. Records
and dictionaries use `Key` / `Value` columns, or a leading `Key` column followed
by record fields when every value is a record. Sequences of records use the
first-seen union of fields as columns. Rows retain input order; missing and
absent cells are blank. Empty collections display `_No entries._`.
Nested cells and structural dictionary keys use compact ordinary JSON with
Markdown escaping. Nominal members and explicit keyed arrays retain their
shared tagged shape inside those cells.

For a native-subject report, save this as `subjects.zirql`:

```zirql
match { from subject in (input | filter(op("func.func"))); } | markdown;
```

Run it inline or from a file:

```sh
printf 'module { func.func @f() { func.return } }\n' | cargo run --locked -p zirium -- \
  'match { from subject in (input | filter(op("func.func"))); } | markdown;'
printf 'module { func.func @f() { func.return } }\n' | cargo run --locked -p zirium -- -f subjects.zirql
```

The `subject` column contains each operation's ordinary descriptor. Capture
rows retain native handles until rendering. Raw operation streams still need
an explicit row or scalar projection; computational queues and identity sets
cannot be rendered. [Library use](query-dsl.md#markdown-reports) runs the same
query through `Query::parse`.

`markdown` preserves the current result, as `json` does. It surrounds its
output with blank lines, escapes Markdown and HTML characters in cells, and
renders embedded newlines as `<br>`. The table uses GitHub-flavored Markdown.
The expanded table, including missing cells and headers, must fit
`--max-items`; emitted text also consumes `--max-work`. A late evaluation, rendering or
interpolation error leaves CLI stdout empty, including earlier `print` output.

## Counting reachable operations

Adding `reachable` includes supported dependencies and referenced bodies:

```zirql
let functions = filter(op("func.func"));

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

A predicate can name its subject or use `_`. Shared expressions, Boolean
operators and permitted pure function calls compose with the native leaves:

```zirql
filter(p => one(p | names) == "arith.addi" and not has_attr("skip"))
filter(one(_ | names) == "arith.addi" and not has_attr("skip"))
```

A constructed filter tests each element and preserves its input order:

```zirql
let pairs = [(1, 2), (3, 4)];
pairs | filter((a, b) => a + b > 4) | json
```

This selects the same tuples as `[pair for pair in pairs if pair.1 + pair.2 > 4]`.
A predicate pattern may destructure tuples, records and positional lists. `_`
always denotes the whole subject, even when the pattern binds its parts.
Nested predicates give `_` a new subject; an explicit outer name stays visible:

```zirql
[1, 2, 4] | filter(outer => length([2, 3] | filter(_ > outer)) > 0) | json
```

The result is `[1, 2]`. `_` is available only inside predicate arguments. It
still discards a component when written in a pattern.

Native `filter` tests one operation, `root` tests each operation and then its
ancestors, and `paths(..., to=...)` tests each candidate target. A native match
`where` tests the original incoming input; its named clause captures remain
separate bindings. All these positions accept the same named and implicit
predicate bodies. Native handles retain their owners and remain opaque; use
projections such as `one(p | names)` rather than invented operation fields.
Invalid bodies and effects are checked before selecting subjects, including
on typed empty streams. Mismatched argument patterns fail at their location.

The native predicate leaves are:

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

```zirql
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

```zirql
filter(op("linalg.matmul"))
| root(op("func.func"))
| unique
```

`subtree` expands each selected operation independently. Overlapping subtrees
therefore contain duplicates. An empty stream stays empty.

For example, find the function containing a call, expand its body, and tag its returns:

```zirql
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

```zirql
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

```zirql
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

```zirql
filter(op("arith.addi")) union filter(op("arith.muli"))
```

For this example, `filter(op("arith.addi") or op("arith.muli"))` is shorter.
Set operators are useful when the operands navigate differently:

```zirql
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

```zirql
(filter(op("arith.addi")) | users union filter(op("arith.muli")) | defs)
| count
```

An operand searches the complete document only when it receives the initial
selection or explicitly uses `input`. This query adds every tagged operation
in the document to the current selection:

```zirql
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

```zirql
filter(op("func.call")) | fixpoint(closure)
```

Closure requires registered reference semantics. Unregistered operations and
unsupported symbol or successor references produce errors.

`where_exists(query)` keeps each original operation whose pure relation returns
at least one operation. The relation receives one operation and retains saved
bindings, parameters and function locals. Order and duplicates survive. A complete
empty relation drops the subject; unresolved navigation or a nested error fails
the request. Kinds and effects are checked even for empty input.

```sh
zirium 'where_exists(users) | names' examples/cli/arithmetic.mlir
```

The Rust and Python `where_exists` builders use the same filtering algorithm.
Relations cannot edit, render or emit. Native handles stay native until an
explicit output stage.

`fixpoint(query)` repeatedly replaces the selection with the query's result
until its ordered values, including duplicates, are unchanged. Zirql owns this
replacement and cycle policy, retaining one checkpoint. Its body can contain navigation, filters,
set expressions, grouping, and nested fixed points. It cannot edit, count, or construct maps.
Use `fixpoint(closure | emit)` to inspect each iteration, including the final
unchanged result. If selections cycle without becoming unchanged, evaluation
fails. Emission does not affect convergence.

Fixed points do not implicitly accumulate intermediate selections:

```zirql
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
Native kind and diff-side checks apply even to empty selections. Pure function
bodies use the same reader and runner while retaining parameters, locals and
saved selections. Domain closure and the optimized `fixpoint(closure)` worklist
remain host operations.

By default, each evaluation permits 10,000,000 work units and 1,000,000 items per
stream. Work counts scope and type preparation, stage input items (at least one
per stage), fixed-point iterations, ordered comparisons, retained checkpoint and
body-input copies, and dependency, ancestor, and subtree visits. These limits do not
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

```zirql
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

```zirql
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

```zirql
format
```

Formatting options use named arguments. `width` controls the preferred line
width and `indent` controls spaces per indentation level. Both must be positive;
`indent` may not exceed 256.
Use MLIR's quoted generic assembly form explicitly when source spelling should
be replaced:

```zirql
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

This sketch covers native pipeline composition. `shared-expression` uses the
value syntax described above; `pattern` uses the shared destructuring syntax.
The `object`, `array` and `value` rules describe JSON templates, not the full
shared value grammar. Function and parameter signatures have their own sections.
Declarations do not replace the final query; bindings and parameters require
their semicolons.

```text
program    = empty | { declaration | statement } final
statement  = binding | do-statement | query ";"
declaration = function | "param" identifier ":" ("int" | "string" | "bool") ";"
function   = "fn" identifier "(" parameters ")" "->" type "{" function-body "}"
final      = query [ ";" ] | do-statement
binding    = [ "let" ] identifier "=" query ";"
           | "let" pattern "=" shared-expression ";"
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
           | "where_exists" "(" query ")"
           | "fixpoint" "(" query ")" | "(" query ")"
           | "set_attr" "(" string "," string ")"
           | "remove_attr" "(" string ")" | "emit" | "count"
format-option = "assembly" "=" ("source" | "generic")
              | ("width" | "indent") "=" integer
predicate  = [ pattern "=>" ] predicate-body
predicate-body = and-expr { "or" and-expr }
and-expr   = not-expr { "and" not-expr }
not-expr   = { "not" } primary
primary    = shared-boolean-expression | native-leaf | "(" predicate-body ")"
native-leaf = "op" "(" shared-expression ")"
            | "dialect" "(" shared-expression ")"
            | "result_type" "(" shared-expression ")"
            | "has_attr" "(" shared-expression ")"
            | "string_attr_eq" "(" shared-expression "," shared-expression ")"
            | "same_value" "(" query "," query ")"
object     = "{" [ string ":" value { "," string ":" value } ] "}"
array      = "[" [ value { "," value } ] "]"
value      = object | array | string | number | "true" | "false" | "null" | identifier
integer    = digit { digit }
number     = a JSON number
```

A predicate body must produce a Boolean. `true` and `false` are ordinary shared
Boolean expressions. Native leaves keep their argument types; `same_value`
is available only in native match guards.

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
ordering and `map_by`. Textual and typed diff `unique` accept the same change, operation and string
streams. Tally retains native keys and their source side. The document labels are available as
`before_document` and `after_document` for interpolation. See
[semantic diff](semantic-diff.md) for the full policy.

## Correlated constructed reports

`project(subject, {fields})` evaluates ordered fields for each native input
operation or capture record. The subject is local to the body: a singleton
operation selection or one capture record. Rows follow
input order, and duplicate subjects or equal projected names remain separate
rows. Native handles require explicit scalar projections such as `names`,
`attr` or `count`.

Pass a projection as the inline query argument:

```sh
zirium 'filter(op("func.func"))
  | project(f, {
      "name": f | attr("sym_name") | one,
      "n": f | children | count
    })
  | json' examples/cli/function-report.mlir
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

Common value expressions use `index_set(a)` for the keys of a positional array
or a rank-one keyed array, and `tuple.1` for a positive one-based tuple
position. Record fields, tuple positions and array lookups compose from left
to right: `({"rows": [(2, 5)]}, 0).1.rows[1].2` returns `5`. These accesses
bind more tightly than arithmetic and ranges.

This [saved program](../examples/cli/index-sets-and-tuples.zirql) pairs each
projected function name with its array position:

```zirql
let labels = filter(op("func.func")) | attr("sym_name");
report [{"position": i, "name": (labels[i], i).1}
        for i in index_set(labels)] | json;
```

Run it inline or with `zirium --program-file examples/cli/index-sets-and-tuples.zirql
examples/cli/function-report.mlir`. It returns positions `1`, `2` and `3` for
`double`, `mix` and `plain`. Empty positional input has an empty integer index
set. A keyed index set preserves its axis's integer or nominal key identity,
including typed empty axes, and uses the existing set order. Non-array or
higher-rank input and missing tuple positions fail with query locations.

An indexed array associates cells with integer keys instead of positional
indices. `[5: 50, 2: 20]` infers the keys `2`, `5` in numeric order.
To retain unused keys or choose their order, write
`[2: 20, 5: 50] axes([5, 2, 9])`. The declared key domain is called an axis;
key `9` belongs to the domain without storing a cell. `has_key(a, 9)` is
therefore false, and `a[9]` fails.

Tuple keys give each dimension its own ordered domain:

```zirql
let sparse = [(2,1):21, (5,8):58, (2,8):28] axes([5,2,9], [8,1,4]);
let generated = [(i,j):i*10+j for i in [5,2] for j in [8,1]
                 if i==2 or j==8] axes([5,2,9], [8,1,4]);
report {"same": sparse==generated, "cells": entries(sparse)} | json;
```

Both arrays store three cells, ordered by the row keys and then the column
keys. Missing cells stay missing. Declared domains evaluate once, left to
right, in the surrounding scope before cells or comprehension generators.
Keys must be unique integers or members of one supplied nominal type per
dimension; all stored keys must belong to their domains. Zirium's text
examples use integers.

`[]` remains positional. `[] axes(set<int>(), [8,1])` is an empty rank-two
indexed array with a typed empty row domain and two column keys.
`[] axes([5,2,9])` retains all three keys without storing cells. An untyped
empty domain `[] axes([])` is refused because it does not establish a key
type. The `axes(...)` suffix belongs directly after an indexed bracket,
before access or a pipeline; it is not a standalone value call.

Rust library results retain the keyed array and its domains. Direct JSON output
uses Zirql's tagged array representation, including complete declared axes and
only present cells. For example, `zirium '[2:20] axes([2,9]) | json' input.mlir`
keeps key `9` in `axes` without inventing a cell. Ordinary projections such as
`entries`, `index_set`, access and `has_key` remain available. This
[saved program](../examples/cli/indexed-arrays.zirql) shows unused keys and
ordered stored cells:

```sh
zirium --program-file examples/cli/indexed-arrays.zirql examples/cli/function-report.mlir
zirium 'let counts = [2:20, 5:50] axes([5,2,9]);
  report {
    "keys": index_set(counts),
    "cells": entries(counts),
    "second": counts[2],
    "has_nine": has_key(counts,9)
  } | json;' examples/cli/function-report.mlir
```

Both commands return keys `[5,2,9]`, cells for `5` and `2`, the value `20`
for key `2`, and `false` for `has_nine`.

This complete program uses local literal arguments:

```zirql
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

Value sources, constructed records and `project` fields lower to Zirql's common
`Value` and `Projection` steps. Each request prepares those expressions once
through its shared function unit before executing statements. Rows and generator
values use the current local scope; selected native operations run against the
current document. Preparation checks every written branch, while Boolean
branches and default fallbacks keep their existing runtime demand. Reusing a
parsed query prepares a fresh invocation for its new input. See the
[shared projection contract](https://zayenz.se/zirql/guides/host-integration/#correlated-native-projection)
for field scope and empty-schema rules.

Immutable constructed binding reads can share their retained value inside that
invocation. Inspection and slot sharing charge their actual work without copying
payload bytes. Owned report fields, emission snapshots, Python conversion and
rendering still pay their materialization costs. Preparation metadata, selected
new values and native work spend the same cumulative allowance. Defaults and
nesting ceilings are unchanged; lower repeated analysis/copy costs can let a
request near an old limit complete, or move the location where it exhausts work.

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
zirium 'filter(op("arith.muli"))
  | paths(users, to=op("func.return"), max_depth=2)
  | json' examples/cli/function-report.mlir
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

Zirium supports ordered native captures with `match`. This
program selects additions whose two operands are the same SSA value, then
projects a scalar report:

```zirql
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
zirium 'match {
  from add in (input | filter(op("arith.addi")));
  bind lhs = operand(add, 0);
  bind rhs = operand(add, 1);
  where same_value(lhs, rhs);
} | json' examples/cli/function-report.mlir
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

## Typed recursive functions

Declare a function with `fn name(parameter: type) -> result_type { value }`
and call it as `name(argument)`. Its body ends with the result expression;
earlier local `let` bindings end with semicolons. A conditional value uses
`if condition { value } else { value }`. Functions can calculate scalar values,
construct reports, or accept checked MLIR handles and call Zirium's native
functions. The examples below cover these uses. The
[Zirql function guide](https://zayenz.se/zirql/guides/functions-and-control-flow/)
adds further scalar examples and explains the shared execution rules.

Calls follow Zirql's [shared scope and admission rules](https://zayenz.se/zirql/guides/functions-and-control-flow/).
Function declarations may appear after a call or refer to later functions.
Saved bindings and parameters remain sequential: they are available only after
their declaration. All inputs to functions are explicit; a function cannot
capture saved query bindings, caller parameters, a projection subject or implicit
`input`. Every written body is
prepared before execution, including skipped branches. Native functions use the
same ordinary call path as declared functions:

| Signature | Result and domain rule |
| --- | --- |
| `direct_children(op: Operation) -> sequence<Operation>` | Region, block and operation containment order, including empty children. |
| `cost_annotation(op: Operation) -> opt int` | Decoded `cost` attribute; absence is allowed, invalid or out-of-range integers fail. |
| `same_operation(a: Operation, b: Operation) -> bool` | Equality of checked document-owned operation handles. |
| `user_info(op: Operation) -> record{targets: sequence<Operation>}` | Complete SSA adjacency in result/use order; incomplete documents fail. |

Each argument must contain one operation in the current document. Admission
checks owners and stale handles; implementations retain their domain checks.
The [embedding guide](https://zayenz.se/zirql/guides/embedding-programs/#define-a-native-function)
shows the common definition contract.

```zirql
fn subtree_cost(op: Operation) -> int {
  (cost_annotation(op) default 1)
    + sum([subtree_cost(child) for child in direct_children(op)])
}
let selected = filter(op("example.root"));
report subtree_cost(selected) | json
```

The [saved program](../examples/cli/subtree-cost.zirql) returns `8` for the
[containment tree](../examples/cli/subtree-cost.mlir): one unannotated root and
exactly two leaves with integer costs `3` and `4`. `direct_children` follows
region, block and operation order. It follows containment, including zero
children, without consulting SSA users. Its result is `sequence<Operation>`.
`cost_annotation` reads the document's decoded `cost` attribute and returns
`opt int`. Missing cost is absent and selects `default 1`; a non-integer,
malformed integer or value outside i64 fails at the annotation read.

Calls accept zero or more comma-separated arguments. They work in reports,
projection fields, computed native arguments and native query sources. For
example, an `identity(op: Operation) -> Operation` call can feed
`identity(selected) | children | count`. A native call such as
`direct_children(identity(selected))` also retains the native argument.
An explicit call source receives its written arguments rather than the
surrounding pipeline selection.

A function can return projected records. Fields inside its projection can read
the function's parameters and earlier locals; each field's relative input is
the singleton subject. Nested projections keep that lexical scope. For example:

```sh
zirium 'fn summarize(root: Operation) -> sequence<record {"name": string, "children": int}> {
  root | project(row, {"name": one(row | names), "children": children | count})
} summarize(input | head(1)) | json' examples/cli/function-report.mlir
```

Every written field is checked before subjects are selected. Empty projections
retain their record schema, and native handles still need a scalar projection
such as `names` or `count` before they can become report fields.

Signatures accept `value`, `int`, `float`, `bool`, `string`, `opt T`,
`sequence<T>`, `tuple<T, ...>`, `record {field: T, ...}`, `dict<K,V>`, `set<T>`
and `keyed<T>`. `value` admits ordinary constructed values; `set<value>` admits
general set elements. Scalar set declarations, including `set<string>`, retain
their element contract. Nested declarations use the same grammar.
Keyed shape and present cells remain runtime facts. Zirium's native
names are `Operation` and `SSAValue`. `SSAValue` means one exact checked
`ValueId`, including its result or block-argument position. Operation admission
requires exactly one valid current-document operation. An explicit native
operation sequence retains its sequence shape at length zero or one.
Native capture records and maps check their declared fields and nested handles.
There is no native-handle transport through external `param` declarations;
those inputs remain int, string or bool.

Dictionary admission checks retained key and value schemas before entries, including
empty and nested dictionaries. Each native key or value still needs a valid
current-document handle. For example, a nested dictionary can retain operations
at both levels:

```zirql
fn nested(root: Operation) -> dict<Operation, dict<Operation, Operation>> {
  {root: {root: root}}
}
nested(input | head(1))
```

No operation is converted to a name for admission. A retained empty
`dict<string, int>` is incompatible with `dict<int, int>` despite having no keys.

Function arguments and results are admitted against their whole declarations.
Wrong arity, duplicate names, unknown calls, unavailable names and effects in
any written body or branch fail before execution, even when unused. Runtime
failures keep their query range and bounded active call context. Optional
absence remains distinct from unknown type information and execution failure.
`default` handles absence; it does not catch a type, arithmetic or budget error.
Functions cannot edit, emit or invoke an impure nested query. Textual functions
are unavailable in diff programs.

Conditional values require a final `else`:

```zirql
fn accumulate(n: int, total: int) -> int {
  if n == 0 { total } else { accumulate(n - 1, total + n) }
}
report accumulate(1000, 0) | json
```

`else if` adds ordered conditions. Each branch is a lexical block with ordered
local `let` declarations and one final value. Compatible self and mutual tail
calls reuse an activation. Pending arithmetic, incompatible return declarations
and native callbacks keep their non-tail or host-reentry obligations.

Zirium defaults to 64 active user-function calls. Rust callers can set
`FunctionLimits { active_depth: 1 }` with `Query::with_function_limits` or the
corresponding typed root method; the accumulator still returns `500500`.
Expression nesting and host reentry have separate finite guards. Every call,
native inspection and result spends the same request allowance, so infinite
tail recursion still fails normally. Function-bearing programs use aggregate
program budgeting even when they contain no comprehension. CLI controls remain
`--max-work`, `--max-items` and `--max-results`.

Declarations and written entries prepare once per submitted request. Projection
rows reuse those entries with current bindings and the current document;
immutable native results retain owners across released function frames.
Rendering remains explicit. The CLI buffers every emission until success;
Rust's emission callback can observe an earlier event before a later failure.

## Destructuring and guarded value matches

Patterns bind parts of a value in `let` declarations, function parameters,
predicate arguments, comprehension generators and value-match arms. A tuple uses exact arity, such as `(label, cost)`. A record
uses exact labels, such as `{label: label, cost: cost}`; add `..` to ignore
other fields. A list pattern `[first, second]` requires exactly two positional
elements; `[first, second, ..]` requires at least two and discards the rest.
`[]` matches an empty list and `[..]` accepts any positional list. Lists remain
distinct from tuples, sets and indexed arrays. `_` discards a part. `absent`, `present(pattern)` and scalar
literals test presence or a value without introducing a name.

```zirql
fn add_cost((label, {cost: (_, cost), ..}):
    tuple<string, record {cost: tuple<string, int>, ignored: int}>)
    -> record {label: string, total: int} {
  if cost >= 0 {
    let answer = cost + 1;
    {"label": label, "total": answer}
  } else { {"label": label, "total": 0} }
}
let (label, details) = ("alpha", {"cost": ("estimate", 4), "ignored": 99});
report add_cost((label, details)) | json;
```

The initializer runs before any of its pattern's names become available. A
failed binding introduces no names and fails the request. Names cannot repeat
within a pattern or shadow an enclosing binding. A function admits every whole
argument against its declaration before destructuring it. Structural patterns
can inspect constructed records and explicit native capture records; native
operation and SSA handles remain opaque checked values.

Value matches try their arms in order. The subject runs once, and a false
guard continues to the next arm. Guards require known Boolean values. Include
an unguarded final `_` arm, even when earlier arms appear exhaustive:

```zirql
let row = {"cost": 4};
report match (lookup(row, "cost")) {
  absent => {"kind": "absent", "cost": 0},
  present(cost) if cost > 10 => {"kind": "large", "cost": cost},
  present(cost) => {
    let answer = cost + 1;
    {"kind": "small", "cost": answer}
  },
  _ => {"kind": "fallback", "cost": 0}
} | json;
```

Every written arm is checked for scope, calls and effects before execution.
Only the selected arm's value runs. A failing guard, stale or foreign native
handle, type error or exhausted budget fails the request; none becomes a
pattern mismatch. Blocks and selected match arms retain compatible tail-call
behavior. Native capture enumeration keeps its separate `match { from ... }`
syntax.

The [destructured report](../examples/cli/destructured-report.zirql) and
[guarded match](../examples/cli/guarded-match.zirql) are complete saved programs.
Their [typed builders](query-dsl.md#destructuring-reports) construct the same
values without parsing generated source.

## Explicit loops and native worklists

A loop names its immutable state and ends each body path with `continue` or
`break`. Initializers run in the outer scope. A continue computes every next
value against the old state, from left to right, then replaces all slots
together. Each slot retains its established type. Break returns the loop value;
nested transfers target the innermost loop and cannot cross a function call.
Iterations spend work without adding active function calls.

```zirql
loop (n = 5, total = 0) {
  if n == 0 { break total; }
  else { continue (n - 1, total + n); }
} | json
```

This short inline query returns `15` without a declaration preamble. Every
written branch must be pure and finish with a transfer. A missing transfer,
wrong continue arity or incompatible successor state is an error. Infinite
loops exhaust the existing work allowance.

`queue(items)` creates a FIFO from an eager positional sequence.
`queue<Operation>([])` supplies the element type for empty input.
`pop_front(q)` returns `absent` or `present((item, remaining))`, and
`push_back_all(q, items)` appends in input order. Decomposing a pop result retains
the remaining queue; it does not reconstruct its elements.
`identity_set<Operation>()`, `item in seen` and `insert(seen, item)` use checked
native identity. Insertion is idempotent. The set exposes neither iteration
order nor general set algebra. `SSAValue` identity includes the exact result or
block argument; same-named operations remain distinct.

The [complete reachability program](../examples/cli/worklist-reachability.zirql)
uses a destructured parameter and optional pop pattern:

```zirql
fn reaches((start, target): tuple<Operation, Operation>) -> bool {
  loop (pending = queue([start]), seen = identity_set<Operation>()) {
    match (pop_front(pending)) {
      present((current, remaining)) => {
        if current in seen {
          continue (remaining, seen);
        } else if same_operation(current, target) {
          break true;
        } else {
          let {targets, ..} = user_info(current);
          continue (push_back_all(remaining, targets), insert(seen, current));
        }
      },
      _ => break false
    }
  }
}
let start = filter(op("graph.start"));
let target = filter(op("graph.end"));
let disconnected = filter(op("graph.disconnected"));
report {
  "zero_edges": reaches((start, start)),
  "reachable": reaches((start, target)),
  "disconnected": reaches((start, disconnected))
} | json
```

Run it against the [valid SSA fixture](../examples/cli/worklist-reachability.mlir):

```sh
zirium --program-file examples/cli/worklist-reachability.zirql \
  examples/cli/worklist-reachability.mlir
```

The result has `zero_edges: true`, `reachable: true` and `disconnected: false`.
`same_operation(a, b)` compares checked
Operation handles. `user_info(op)` returns the explicit native record
`{targets: sequence<Operation>}`. Targets follow result index, then the existing
SSA use-site order, including successor arguments and repeated uses. The view
reuses the document's use index. It returns complete adjacency or a located
error; an incomplete document cannot become an empty adjacency list or false
reachability. The fixture includes duplicate routes and two distinct
`graph.node` operations.

Queues and identity sets are computational values. They can occur in locals,
function signatures, native results and loop state. Rendering or serializing
one, including inside a tuple, sequence or record, fails at the expression or
emitter. Native Rust callers can retain results and supply them to another
request against the same unchanged document. An edit or different document
invalidates the context even for empty queues and sets. Retained aliases and
earlier scan states keep their original values.

## Native composition boundary

The development checkout lowers filters, matches, ordering, maps, existence,
fixed points and paths directly to Zirql composition steps. Text queries, Rust
builders and Python builders use the same preparation and iteration rules.
Zirium supplies native predicates, navigation, handle identity and output shape;
diff queries additionally retain their existing supported forms and side rules.
A native result remains native until a query explicitly projects scalar fields.

Selectors and `where_exists` receive one subject. A `fixpoint` body receives the
whole current selection. Match sources and guards retain the original input and
only earlier named captures. Equal ordering keys keep input order; unique keeps
the first native identity; empty capture results keep their field schema.
Existing strict-closure, empty-extremum, budget and emission behavior applies.
The [native map recipe](query-dsl.md#nested-queries-and-native-maps) uses these
rules through the public builders. The
[architecture note](architecture/syntax-representation.md#query-expressions)
describes the remaining adapter hooks.

## Named fold and scan steps

`fold(items, initial, step)` applies a named pure function from left to right.
`scan` retains each post-step state and excludes the initial state. Both resolve
the step's arity, purity and state contract before iteration, including on empty
input. Items then initial are evaluated once; a step error fails the request.
Keyed arrays require an explicit positional conversion.

```zirql
fn accumulate((count, total): tuple<int, int>, item: int) -> tuple<int, int> {
  (count + 1, total + item)
}
report {
  "fold": fold([2, 3, 5], (0, 0), accumulate),
  "scan": scan([2, 3, 5], (0, 0), accumulate),
  "empty_fold": fold([], (0, 0), accumulate),
  "empty_scan": scan([], (0, 0), accumulate)
} | json
```

The [saved program](../examples/cli/fold-scan.zirql) returns fold `[3, 10]`, scan
`[[1, 2], [2, 5], [3, 10]]`, empty fold `[0, 0]` and empty scan `[]`. Signatures
also accept `queue<T>` and `identity_set<T>` with native set element types.
Loops, reducers, native inspection, copying and rendering share the existing
request budget. The
64-call default, expression nesting, reentry and CLI output buffering remain
unchanged.

## A bounded constructed report

This report combines a tail accumulator with optional scores, exact string
matching and stable ordering. Save the program as `report.zirql`:

```zirql
fn accumulate(n: int, total: int) -> int {
  if n == 0 { total } else { accumulate(n - 1, total + n) }
}
let rows = [{"name":"beta","score":2}, {"name":"alpha","score":2},
            {"name":"gamma","score":null}];
let scored = [{"name":r.name,"score":(r.score default 0)}
              for r in rows if contains(r.name,"a")];
report {"total":accumulate(1000,0),"rows":sort_by(scored,"score"),
        "first":min_by(scored,"score")} | json
```

Run from the repository root against the existing containment fixture:

```sh
zirium --program-file report.zirql examples/cli/subtree-cost.mlir
```

The total is `500500`. Rows appear as `gamma` with score zero, then `beta`
and `alpha` with score two; the equal scores retain input order. `first` is
the original `gamma` record. The same
program text can be passed as the inline query argument.

The default supplies zero for an absent score; ordinary structural ordering
places an absent key first. It does not catch arithmetic, type, host or budget
errors. String matching is case-sensitive, with no Unicode normalization or
regex interpretation; an empty search string matches. Sorting uses structural
keys and preserves ties even with `descending=true`. Empty extrema
return absence, and extrema choose the first input on a tie.

Typed function boundaries check actual selected arguments and results. They
do not infer model types or validate every runtime fact statically. Native
identity and document/source ownership stay with the host. External parameters
remain `int`, `string` or `bool`. All written branches receive preflight checks;
only selected lazy branches execute.

The accumulator reuses its active frame; pending arithmetic in a recursive
caller requires another frame. Neither form bypasses cumulative work/results
limits, expression nesting or the native reentry guard. Bodies prepare once
within the submitted evaluation and share unchanged admitted arguments. A new
evaluation prepares against its current document and bindings. Owned results
and rendering still pay their applicable costs.

The [shared execution measurements](https://zayenz.se/zirql/guides/computation/)
record both bounded tail state and small-query regressions. Those library
measurements exclude host parsing/rendering and do not predict CLI latency.

## Shared data JSON

The development dependency pins Zirql `49c30dac58ed2614fb9d6ba97dafe1432fa7a9cc`
and uses its ordinary JSON codec for constructed
reports and data inside native maps, sequences and captures. Scalars stay JSON
scalars, positional collections stay arrays, and records preserve field order.
General dictionaries retain explicit key/value entries; nominal members and
indexed arrays use tagged data so their names and declared shape survive.

For a mixed native and calculated report, run this against the existing fixture:

```sh
zirium 'let subject = filter(op("func.func")) | head(1);
  fn pair(value: Operation) -> tuple<Operation,int> { (value, 1) }
  report pair(subject) | json' examples/cli/function-report.mlir
```

The outer tuple contains an ordinary operation sequence followed by scalar `1`.
Zirium supplies the operation descriptor; Zirql traverses the tuple and renders
the constructed scalar. A singleton operation in a capture field is one descriptor; ordinary operation
sequences remain arrays. `--jsonl` keeps the `document` and `result` envelope.
An operation emitted as MLIR remains a source string in that envelope. Diff
output retains its `zirium.diff.v1` schema, document names, side and comparison
metadata. Both CLIs buffer the whole request, including later rendering errors.
The [Markdown emitter](#markdown-reports) uses the same native projection
boundary and shared Zirql table rendering.

## Explicit scalar conversions

Project native operations to ordinary scalars before converting them. From the
repository root, save this query as `conversion-report.zq` and run it against the
existing function report fixture:

```zirql
fn label(total:int)->string { "operations=" ++ to_string(total) }
let total = input | count;
report {"label":label(total), "parsed":parse_int("002"), "price":parse_float("1.25e1")};
```

```sh
cargo run --locked -p zirium -- \
  --program-file conversion-report.zq examples/cli/function-report.mlir
```

The report retains the operation count in its label, with parsed integer `2` and
price `12.5`. [The shared scalar reference](https://zayenz.se/zirql/guides/library-strings/#parse_int)
defines exact decimal grammar, overflow and underflow errors, signed zero and
round-trippable formatting. These common calls work in ordinary reports and pure
functions in this development checkout's pinned Zirql revision. Native handles
need their existing scalar projection; conversion never formats a handle or changes
its owner, identity or coordinates.

## Collection selection and cardinality

The development checkout uses the [shared collection stages](https://zayenz.se/zirql/guides/library-pipelines/)
and [length](https://zayenz.se/zirql/guides/library-collections/#length) for
ordinary report data and admitted native selections/maps. Operation predicates
retain their native subjects, document ownership and completeness policy.

```text
let limit=2;
let message="two" ++ " rows";
report [3,1,3] | filter(_>1) | head(limit) | reverse | check(limit,message);
```

This emits `[3,3]`. Keyed arrays retain axes and missing cells; project cells
explicitly before removal or reversal. These additions require the development
checkout; released Zirium 0.6.1 retains its published support.

Legacy JSON map value predicates use the existing checked ordinary-data
projection. JSON integers outside signed 64-bit range are refused with a located
conversion error; retained entries keep their original JSON values. Null projects
to ordinary absence. Native ordering retains its separate exact key contract.

Unique and tally use Zirql's shared [structural identity contract](https://zayenz.se/zirql/guides/library-pipelines/#unique).
Native operation keys retain document ownership. Distinct operations with the
same name remain separate; use `names | tally` to count spellings explicitly.
Saved tally dictionaries support lookup and dictionary continuations:

```zirql
let hist = input | tally;
hist | reverse | head(1) | check(1) | sort_by(value) | json
```

String keys retain ordinary dictionary JSON. General keys use tagged entry
pairs and the existing native descriptor projection only for rendering.
