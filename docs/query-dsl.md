# Structured queries in Rust and Python

These examples use Zirium 0.6.1. Repository-relative inputs are in the
[`v0.6.1` checkout](https://github.com/zayenz/zirium/tree/v0.6.1);
see [getting started](getting-started.md) for installation.

Build queries with methods and predicates, then evaluate them against a semantic
document. Rust and Python use the same expression tree and return native values.
Queries are read-only; use the [textual language](query-language.md) for CLI
queries and edits.

Textual [positional comprehensions](query-language.md#positional-comprehensions)
construct native Zirql values from explicit scalar projections. Rust callers
can match `NativeValue::Constructed(report)` and inspect `report.value()` in a textual `Query`
callback before rendering. The typed Rust and Python builders keep their
existing operation handles, strings, counts, and maps. The
[report builders](#constructed-report-builders) also construct shared values,
comprehensions and correlated rows. [Native capture builders](#native-capture-builders)
retain operations and exact SSA values before an explicit scalar projection.

## Start a query

In Python, `Document.query()` returns the result described by the expression:

```python
import zirium
from zirium.query import ops, op

source = '''
%c = "example.make"() : () -> i32
"example.consume"(%c, %c) : (i32, i32) -> ()
'''
lowered = zirium.parse_text(source).lower_strict()
assert lowered.document is not None, lowered.diagnostics
document = lowered.document

consumers = ops().filter(op("example.make")).users().unique()
selected = document.query(consumers)        # list[SemanticOperation]
count = document.query(consumers.count())   # int
assert [operation.name for operation in selected] == ["example.consume"]
assert count == 1
```

The equivalent Rust expressions use the same methods:

```rust
use zirium::query::{ops, op};

let consumers = ops().filter(op("example.make")).users().unique();
let selected = document.query(&consumers)?;        // Vec<OperationId>
let count = document.query(&consumers.count())?;   // usize
```

Expressions are immutable and reusable. Each `document.query()` call evaluates
the expression against that document's current state.

Rust `Document::query()` uses the baseline registry. For other dialect semantics,
use `document.query_with_registry(&expression, &registry)`, passing the registry
used to parse and lower the document. Python retains that registry in the document
and uses it automatically.

## Required scalar inputs

A typed request carries root declarations through composition, erasure and
conversion to a report expression. Attach declarations with `with_parameters`
and supply native scalar values at evaluation:

```rust
use zirium::query::{ConstructedValue, EvaluationLimits, Located,
    ParameterDeclaration, ParameterType, ReportExpr, report};

let query = report(ReportExpr::name("minimum_ops")).with_parameters(vec![
    ParameterDeclaration {
        name: Located::new("minimum_ops".to_owned(), 0..0),
        ty: Located::new(ParameterType::Int, 0..0),
    },
]);
let result = query.evaluate_with_parameters(
    &document, &registry,
    &[("minimum_ops".to_owned(), ConstructedValue::Integer(2))],
    EvaluationLimits::default(),
)?;
assert_eq!(result.value(), &ConstructedValue::Integer(2));
```

Root declarations are visible throughout the builder request. Identical
metadata shared by composed expressions is retained once; conflicting
metadata reaches admission and fails. A local scope or projection cannot
shadow a root declaration. `op_expr(ReportExpr::name("operation_name"))`
checks a supplied string. Rust also has `head_expr`, `tail_expr`, `attr_expr`,
`defs_at_expr` and `users_at_expr` for shared scalar expressions.

Python report requests use the same declarations and admission:

```python
from zirium.query import ReportExpr, report

query = report(ReportExpr.name("minimum_ops")).with_parameters(
    {"minimum_ops": "int"}
)
assert document.query(query, parameters={"minimum_ops": 2}) == 2
```

Declaration types are `"int"`, `"string"` and `"bool"`; inputs are actual Python
`int`, `str` and `bool` values. Booleans remain Boolean values rather than integer
inputs. Integers must fit signed 64 bits; overflow raises `OverflowError`.
Unsupported value kinds raise `TypeError`. Strings remain exact data.
Conversion, admission, evaluation and native result conversion spend the same
request allowance, including input names and string bytes.

Every declaration needs one input; missing or extra inputs and type mismatches
fail before native document work. Existing evaluation methods supply an empty
input set. For parsed Rust requests, `Query::parameters()` exposes declarations,
`Query::evaluate_with_parameters` accepts typed scalar pairs, and
`evaluate_with_parameters_and_budget` accepts `ParameterInputs::Typed` or
`ParameterInputs::Cli` with the caller's budget. Library emission callbacks
retain their existing behavior; callers buffer emissions when a later failure
must discard earlier output. The CLI already buffers complete requests.

Text queries also accept shared predicate arguments, for example
`filter(p => one(p | names) == "example.make")` and
`filter(one(_ | names) == "example.make")`. The
[predicate reference](query-language.md#predicates-and-filtering) covers
constructed filters, destructuring and native candidate scopes. Rust and
Python builders keep their typed native predicates; textual arguments require
no host-language callable or closure.

## Compose predicates and streams

Available predicates are `op(name)`, `dialect(name)`, `has_attr(name)`,
`result_type(spelling)`, `string_attr_eq(name, value)`, and `always(bool)`.
Combine them with `&` and `|`; negate with `!` in Rust and `~` in Python:

```python
from zirium.query import has_attr

arithmetic = ops().filter(
    (op("arith.addi") | op("arith.muli")) & ~has_attr("skip")
)
```

Python's `and`, `or`, and `not` do not build predicates. Converting a predicate or
query expression to `bool` raises `TypeError` with guidance on the supported
operators. String arguments are data; they need no query-language escaping.

Navigation methods include `defs`, `users`, `parent`, `children`, `root(predicate)`,
`subtree`, `slice`, `backward_slice`, `forward_slice`, `closure`, and `reachable`.
`slice()` is an alias for `backward_slice()`. Rust has `defs_at(index)` and
`users_at(index)` for indexed navigation; Python accepts `defs(index)` and
`users(index)`. Indices select the current operation's operand or result,
respectively.

Navigation preserves order and duplicates. A consumer using the same result twice
appears twice in `users()`. Add `unique()` when counting distinct operations.
`union(other)`, `intersect(other)`, and `difference(other)` combine operation or
string queries. Union keeps first appearance from the left operand, then the right.
Intersection and difference deduplicate in left-operand order. Each operand is
a complete expression evaluated with the same input.

`head(n)`, `tail(n)`, and `reverse()` apply to streams and dictionaries;
`sort()` orders strings or dictionary keys.
These operate on computed streams and do not avoid parsing the document or
executing earlier stages.

Graph traversal has the same structural meaning and dialect limitations as the
textual language. In particular, `closure()` is one scope-retaining dependency
step; `backward_slice()` follows transitive SSA definitions,
`forward_slice()` follows transitive SSA users, and `reachable()` also follows
supported calls and branches. For repeated closure, use:

```python
from zirium.query import input

fragment = arithmetic.fixpoint(input().closure())
```

A general `fixpoint(body)` replaces its selection until the ordered stream is
unchanged. It preserves duplicate semantics, detects cycles, and obeys work and
item limits. For accumulating transitive users, use
`arithmetic.fixpoint(input().union(input().users()))`.

`MapQuery` returns ordered `Vec<(String, T)>` entries in Rust and a Python
`dict` in insertion order. `sort_by_value()` stably orders scalar values;
`sort()` orders keys. Both return a dictionary query, as do `reverse()`,
`head(n)` and `tail(n)`. Equal values retain their current order, including
after reversal. `head_expr` and `tail_expr` accept a `ReportExpr` bound.

`reachable()` treats an unregistered operation as a retained leaf by default;
it does not follow that operation's regions, operands, successors, or symbol
references. Other queued operations are still processed. Pass `strict=True` to
`Document.query` to reject such an operation by name. Known malformed calls,
missing or external callees, and unsupported registered references remain
errors. `closure()` and `fixpoint(input().closure())` always use strict
reference semantics.

## Nested queries and native maps

`ops()` always starts from every operation in the document. `input()` starts from
the selection supplied to a nested query. At top level, both start from all
operations. In `map_by`, `sort_by`, and `where_exists`, the nested input is one
operation at a time.

This query maps each function name to a histogram of its represented body:

```python
from zirium.query import input

functions = ops().filter(op("func.func"))
report = functions.map_by(
    key=input().string_attr("sym_name").one(),
    value=input().children().subtree().names().tally(),
)
histograms = document.query(report)  # dict[str, dict[str, int]]
```

Rust uses references for nested expressions:

```rust
use zirium::query::input;

let functions = ops().filter(op("func.func"));
let report = functions.map_by(
    &input().string_attr("sym_name").one(),
    &input().children().subtree().names().tally(),
);
let histograms = document.query(&report)?; // Vec<(String, Vec<(String, usize)>)>
```

`one()` requires exactly one string and returns a scalar. Missing or multiple
strings are errors. `map_by` rejects duplicate keys. `tally()` instead combines
equal strings and returns their occurrence counts in first-occurrence order.
`map_by` retains input order. Empty inputs produce empty maps.

Map values can also be operation selections, type or attribute selections,
strings, scalar strings, counts, or nested maps. Operations remain usable handles;
there is no conversion through JSON. Build larger reports with normal Python or
Rust data structures.

To retain operations based on a relationship:

```python
matmuls = ops().filter(op("stablehlo.dot_general"))
with_add_users = matmuls.where_exists(
    input().users().filter(op("stablehlo.add"))
)
```

`where_exists` accepts an operation query and keeps an input operation when that
query returns a nonempty selection. It preserves the original operations,
order and duplicates. A complete empty relation rejects its subject; unresolved
relations and nested errors fail. The text form is `where_exists(users |
filter(op("stablehlo.add")))`. Relations are pure and are checked even when the
outer input is empty.

`sort_by`, `min_by`, `max_by`, `min_all_by`, and `max_all_by` accept scalar string
or count queries. For example, `functions.sort_by(input().children().count())`
sorts by direct child count. Sorting is stable; single extrema retain the first
tie, and all-extrema retain every tie. Extrema on empty streams are errors.

## Projections and editing

| Expression | Rust result | Python result |
| --- | --- | --- |
| Operation query | `Vec<OperationId>` | `list[SemanticOperation]` |
| `names()` | `Vec<String>` | `list[str]` |
| `string_attr(name)` | `Vec<String>` | `list[str]` |
| `result_types()`, `operand_types()` | `Vec<TypeId>` | `list[SemanticType]` |
| `attributes(name)` | `Vec<AttributeId>` | `list[SemanticAttribute]` |
| String query followed by `one()` | `String` | `str` |
| `count()` | `usize` | `int` |
| `tally()` | `Vec<(String, usize)>` | `dict[str, int]` |

Type and attribute queries expose `spellings()` for retained MLIR text.
`string_attr(name)` decodes string attributes, omits missing attributes, and
reports an error for a present non-string attribute or undecodable UTF-8.
Use `attributes(name).spellings()` to inspect those retained spellings.

For example, a configured call target is a symbol-reference attribute, so read
its retained MLIR spelling rather than treating it as a string attribute:

```python
calls = ops().filter(op("vendor.invoke"))
targets = document.query(calls.attributes("target").spellings())
```

`calls.string_attr("target")` is appropriate only when `target` is an MLIR
string attribute; it reports an error for a symbol reference such as `@callee`.

Rust exposes methods according to the result type: a string query has no `users`
method. Python exposes corresponding classes and type declarations, including
`QueryExpr[T]` for annotating reusable expressions.

Evaluated selections keep their membership, while their handles refer to live
objects and reflect later edits. Accessing deleted or invalidated objects raises
a stale-handle error. Scalars and strings keep their evaluated values.

Select first, then use the existing edit transaction:

```python
selected = document.query(ops().filter(has_attr("analysis.tag")))
with document.edit() as edit:
    for operation in selected:
        edit.remove_attribute(operation, "analysis.tag")
```

## Limits and errors

Python accepts per-evaluation limits:

```python
selected = document.query(
    consumers, max_work=1_000_000, max_items=100_000, strict=True
)
```

Rust accepts them on the expression:

```rust
use zirium::query::{EvaluationLimits, EvaluationOptions};

let selected = consumers.evaluate_with_options(
    &document,
    &registry,
    EvaluationOptions { strict_unknown_references: true },
    EvaluationLimits { max_work: 1_000_000, max_items: 100_000 },
)?;
```

The defaults are 10 million work units and 1 million items. Limits cover expression
nodes, stage inputs, visited predicate nodes, traversal work, and nested result
sizes. They are not byte or wall-clock bounds. Boolean evaluation short-circuits;
skipped branches consume no evaluation work.

Nested expressions and predicates have a 64-level limit; flat
pipelines, boolean chains, and set chains do not consume a level per term.
Excessive nesting is reported when the expression is evaluated.

Ordinary and diff programs share statement order, immutable binding scope, groups and set
sequencing with Zirql. Native selectors reenter the expression runner with the
same document or comparison, bindings and budget. Typed diff fixed-point bodies
and navigation use the same active allowance. The textual `Query` API also accepts
`evaluate_with_program_limits`; `--max-results` exposes its aggregate allowance
in the CLI. Typed expression APIs retain their existing limits and native values.
See [query language limits](query-language.md) for the
aggregate units and the caller-owned printing boundary.

Text parsing returns `query::QueryError` with an owned `String` message and a
query byte position. `query::parser::Diagnostic` also owns its message; it is
`Clone` and no longer `Copy`. Its `message(&self) -> &str` and `range(&self)`
accessors borrow the diagnostic. Direct `QueryError` construction supplies a
`String`, for example `message: "invalid query".into()`. Ownership lets detailed
shared errors survive parsing without replacing them with a generic message.

Evaluation returns `EvaluationError` in Rust and raises `ValueError` in Python.
Python rejects invalid argument types with `TypeError`. Evaluation runs in Rust
with shared document access and without holding the Python interpreter lock;
queries cannot contain Python callbacks, edits, or emitters.

Typed diff builders start with `changes()`. A `ChangeQuery` accepts ordinary and
change predicates; `.before()` and `.after()` return side-scoped operation
queries. Rust returns `DiffOperationSelection` rather than bare operation IDs.
Python resolves those selections to wrappers backed by immutable snapshots.
Use `diff_op_input()` to build a relative fixed-point body, for example
`changed_ops.fixpoint(diff_op_input().slice())`. Diff value projections include
`names()`, `attr(...)`, `result_types()`, and `operand_types()`.
See [semantic diff](semantic-diff.md) for examples.

## Constructed report builders

`OpQuery::project` returns a `ReportQuery` whose native result is
`ConstructedReport`. `bound_ops` names a saved native selection or a projection
subject. `ReportExpr` contains the shared value tree; construction reads no
document and accepts no query-source strings or evaluator callbacks.

```rust
use zirium::query::{ReportExpr, bound_ops, op, ops};
let f = bound_ops("f");
let rows = ops().filter(op("func.func")).project("f", [
    ("name", ReportExpr::from(f.attr("sym_name").one())),
    ("operations", ReportExpr::from(f.children().subtree().count())),
    ("matches", ReportExpr::from(f.children().subtree().filter(op("arith.addi")).count())),
]);
let native_report = document.query(&rows)?;
let value = native_report.value();
```

StringQuery, scalar-string queries, CountQuery and ReportQuery convert explicitly
to ReportExpr. Scalar literals, arrays, records, field/index access, generator
clauses, reducers and requirements all use the shared evaluator. `ReportGenerator`
constructs `in` or immutable binding clauses. `ReportExpr::scope` binds values
in order and returns one head through these same clauses. `op_expr` accepts a
checked string expression while the ordinary `op("name")` builder remains
available.
Python converts constructed lists, tuples and sets to ordered lists. Ordinary
string-key dictionaries become Python dictionaries; general dictionaries use
`{"kind": "dictionary", "entries": [{"key": key, "value": value}]}` so nested
collection keys do not require Python hashability. The Rust value retains its
collection kind until conversion. The Python builders retain their existing
constructors; general set and dictionary literals are available through the
textual language and the shared Rust value API.
`attr(name)` follows the textual attribute-spelling projection; `string_attr(name)`
continues to require and decode a string attribute.

Typed constructed requests select aggregate limits automatically.
`evaluate_with_program_limits` accepts explicit limits, and
`evaluate_output_with_budget` retains the caller's remaining allowance for
conversion. Native handles stay document-owned; report values carry their query
location. Render explicitly with ConstructedReport's budgeted JSON methods.

Python mirrors this surface through `zirium.query`:

```python
from zirium.query import ReportExpr as E, bound_ops, op, ops, report, require
f = bound_ops("f")
rows = ops().filter(op("func.func")).project("f", {
    "name": f.attr("sym_name").one(),
    "operations": f.children().subtree().count(),
    "matches": f.children().subtree().filter(op("arith.addi")).count(),
})
result = document.query(report(require(True, E(rows), "stop")))
```

Document.query returns Python bool/int/float/string/list/dict data directly,
with record insertion order. ReportExpr also accepts scalar-query wrappers and
nested list/dict construction. ReportGenerator and ReportExpr.comprehension
supply value generators; `.sum()`, `.length()`, `.forall()` and `.exists()`
reduce arrays. `ReportExpr::group_by("field")` in Rust and `.group_by("field")`
in Python group positional records through the shared evaluator. Groups contain
`key` and `items`, with first-seen keys and original row order and duplicates.
The [grouping language reference](query-language.md#stable-record-groups) defines
supported keys and failures.

Reuse the function rows above, whose `matches` field counts `arith.addi`
operations. Rust returns a native `ConstructedReport`:

```rust
let grouped = document.query(&zirium::query::report(ReportExpr::from(rows).group_by("matches")))?;
let value = grouped.value();
```

The corresponding Python call returns ordered list/dict data:

```python
result = document.query(report(E(rows).group_by("matches")))
```

With the [function fixture](cli-examples.md#group-functions-by-match-count),
both obtain key `1` with `double`/`mix`, then key `0` with `plain`.
Python conversion spends the remaining request budget before
allocating collections. Ordinary operation/type/attribute results retain their
existing semantic wrappers.

## Transform report data

The methods in this section require the development checkout and its Zirql Git
pin `967eb05c80fd122fb5a91cd9056cce124f844150`. They accept shared expressions
and patterns; they do not execute Rust or Python callbacks. Functions and
parameter declarations compose with the existing report metadata.

```python
import zirium
from zirium.query import ReportExpr as E, ReportGenerator as G, ReportPattern as P
from zirium.query import op, ops, report

lowered = zirium.parse_text('"example.seed"() : () -> ()').lower_strict()
assert lowered.document is not None
document = lowered.document
n = E.name("n")
selected = E.comprehension(
    n, [G.in_pattern(P.present(P.name("n")), [None, 1, 2], case=True)]
)
assert document.query(report(selected.map(P.name("n"), n + 1).windows(2))) == [[2, 3]]
assert document.query(report(E([1, 2]).flat_map(P.name("n"), [n, n]))) == [1, 1, 2, 2]

rows = E([{"id": 1}, {"id": 1}, {"id": 2}])
assert document.query(report(rows.runs_by("id").map(
    P.name("run"), E.name("run").field("items").length()
))) == [2, 1]
assert document.query(report(E({"id": 1, "amount": 2}).with_fields({"amount": 3}))) == {
    "id": 1, "amount": 3
}
assert document.query(report(E([{"id": 1}]).align([{"id": 2}], "id").map(
    P.name("entry"), E.name("entry").field("key")
))) == [1, 2]

native_rows = E([{"operation": E.native(ops().filter(op("example.seed"))), "amount": 1}])
result = document.query(report(native_rows.map(
    P.name("row"), E.name("row").with_fields({"amount": 2})
)))
assert result[0]["operation"].name == "example.seed"
assert result[0]["amount"] == 2
```

`ReportExpr.native(query)` retains native operations, SSA values or captures
without converting them to JSON. Python results keep document-owning wrappers.
An ordinary native selection keeps its existing kind; constructing an array of
native records produces positional report data. Rust exposes the same methods:
`map(pattern, body)`, `flat_map(pattern, body)`, `windows(size)`, `runs_by(field)`,
`with_fields(fields)` and `align(after, field)`. `ReportGenerator::in_pattern`
accepts a located `Pattern`, a source, an optional filter and the selective
`case` flag. Existing `in_values` generators remain strict.

Zirql enumerates children and maps payloads. Zirium's normalization callback
handles retained expression wrappers; native-root discovery decomposes only
host stages. Native type conversion, operation ownership, coordinates and
read-only admission remain in Zirium.

## Bounded native path builders

The typed builder uses the same traversal as the
[short command](query-language.md#bounded-ssa-use-paths):

```rust
use zirium::query::{ops, op, PathEdge};

let paths = ops().filter(op("arith.muli"))
    .paths(PathEdge::Users, op("func.return"), 2);
let native = document.query(&paths)?;
```

`paths` returns `CaptureQuery`; `native` is `CaptureMatches`. On
`examples/cli/function-report.mlir`, its one row retains the native `mix`
operations `[arith.muli, arith.addi, func.return]`. The row's `source` and `sink`
are singleton selections, and `steps` is `NativeValue::OperationSequence`.
It remains a sequence for a zero-edge witness. Native IDs and source ranges
remain available before rendering.

Python supplies the supported edge name explicitly:

```python
from zirium.query import ops, op, capture_ops

paths = ops().filter(op("arith.muli")).paths(
    "users", to=op("func.return"), max_depth=2
)
native = document.query(paths)
rows = document.query(paths.project("p", {"steps": capture_ops("p", "steps").names()}))
```

Python path rows are dictionaries with native `SemanticOperation` endpoints
and a list of native `SemanticOperation` steps. Those wrappers retain their
document owner. Ordinary singleton capture fields keep their existing wrapper,
including fields named `steps`. Negative or overflowing Python bounds fail
before document evaluation.

The target accepts ordinary predicates. Root parameter declarations survive
path reuse, erasure, nesting and projection; dynamic target arguments use the
existing typed parameter admission. `PathEdge::Users` enumerates direct SSA
users in result/use-site order. The bound counts edges, matching roots produce
`[source]`, and equal-length alternatives follow that order. No witness within
the bound does not prove global disconnection.

Paths use the active aggregate budget for traversal, copies, projection and
Python conversion. Library callers can retain an explicit
`Budget::for_program(ProgramLimits { ... })`; `EvaluationLimits` keeps its
per-stream and work ceilings. Paths remain read-only and are unavailable on diff
queries.

## Order constructed values

`ReportExpr::sort(descending)` and `sort_by(field, descending)` stably order
positional scalars or records. Pass `false` for ascending and `true` for
descending; ties keep input order in both directions. Python provides the same
methods with keyword-only `descending=False`. The four extrema methods are
`.min()`, `.max()`, `.min_by("field")` and `.max_by("field")`. Empty input returns
absence (`None` in Python), and the first tied input wins. Original rows stay intact; the builders retain function and parameter
declarations.

```rust
use zirium::query::{ReportExpr as E, report};
let rows = E::array([
    E::record([("id", E::string("A")), ("score", E::integer(2))]),
    E::record([("id", E::string("B")), ("score", E::integer(1))]),
]);
let ordered = report(rows.clone().sort_by("score", false));
let best = report(rows.max_by("score"));
let empty_default = report(E::array([]).min().default(E::integer(0)));
```

```python
from zirium.query import ReportExpr as E, report
rows = E([{"id": "A", "score": 2}, {"id": "B", "score": 1}])
ordered = report(rows.sort_by("score", descending=True))
best = report(rows.max_by("score"))
empty_default = report(E([]).min().default(0))
```

Keys require present homogeneous integers, finite floats, exact strings or one
nominal type. Missing/absent fields and incompatible keys fail. Project an
explicit lookup/default key before ordering when needed. The shared evaluator
extracts keys once, charges scratch/comparisons and retains original values.
Native `OpQuery` and map ordering methods keep their existing selector behavior.
See [constructed ordering](query-language.md#sort-constructed-rows-and-select-extrema)
for input conversion, empty/tie rules and the executable report.

## Optional report values

`ReportExpr::null()` constructs absence in Rust. The direct `.occurs()`,
`.absent()`, `.lookup(key)`, `.has_key(key)` and `.default(fallback)` methods
build the same nodes as the textual language. They retain required parameter
metadata from both operands, including skipped fallbacks.

```rust
use zirium::query::{ReportExpr as E, report};

let info = E::record([("id", E::integer(4)), ("stored", E::null())]);
let query = report(E::record([
    ("value", info.clone().lookup(E::string("missing")).default(E::integer(8))),
    ("stored", info.clone().has_key(E::string("stored"))),
    ("info", info),
]));
let result = document.query(&query)?;
```

Python accepts `None` in report scalars, lists and dictionaries and returns it
for stored absence. The corresponding methods accept ordinary report inputs:

```python
from zirium.query import ReportExpr as E, report

info = E({"id": 4, "stored": None})
result = document.query(report({
    "value": info.lookup("missing").default(8),
    "stored": info.has_key("stored"),
    "missing": info.has_key("missing"),
    "info": info,
}))
assert result == {
    "value": 8, "stored": True, "missing": False,
    "info": {"id": 4, "stored": None},
}
```

Lookup misses return absence, while wrong key/container types and strict missing
access still fail. Default evaluates its fallback only for absence; zero,
false and empty containers remain present. Scope, purity and known-type checks
still cover both operands. See the [textual optional report](query-language.md#optional-fields-and-lazy-defaults)
for inline/file usage and the reusable report fixture with a native projection.
Required external parameters continue to accept int/string/bool only; passing
`None` as a parameter fails before execution. Optional report values add no new
MLIR semantics or edit format.

## Native capture builders

`SsaQuery` and `CaptureQuery` retain native SSA values and capture rows.
`OpQuery::operand` returns one exact `ValueId`; `CaptureQuery` returns `CaptureMatches`, whose
`rows()` expose ordered native records and whose `range()` retains the query
location. Neither result is a constructed scalar value.

Build the [native capture example](query-language.md#native-ssa-captures) with
concrete immutable clauses:

```rust
use zirium::query::{bound_ops, bound_ssa, capture_ops, matches, op, ops,
    same_value, ReportExpr as E, report};

let captures = matches()
    .from("add", &ops().filter(op("arith.addi")))
    .bind("lhs", &bound_ops("add").operand(0))
    .bind("rhs", &bound_ops("add").operand(1))
    .where_(&same_value(&bound_ssa("lhs"), &bound_ssa("rhs")))
    .build();
let native = document.query(&captures)?;

let add = capture_ops("m", "add");
let rows = captures.project("m", [
    ("function", E::from(add.root(op("func.func")).attr("sym_name").one())),
    ("operation", E::from(add.names().one())),
]);
let full = report(E::scope(
    [("rows", rows.into())],
    E::record([
        ("matches", E::name("rows")),
        ("count", E::name("rows").length()),
    ]),
));
let result = document.query(&full)?;
```

On the original function fixture, `native` contains one row with fields `add`,
`lhs` and `rhs`; the SSA handles are equal. `result` contains only the `double`
addition and count 1. `bound_ops` and `bound_ssa` refer to earlier captures.
`capture_ops(subject, field)` and `capture_ssa(subject, field)` select a field
from a projection's native record and check its promised kind, even when the
capture collection is empty. Match guards combine with `&`, `|` and `!` in Rust.

Python mirrors these builders with `.from_` and `.where_`:

```python
from zirium.query import (
    ReportExpr as E, bound_ops, bound_ssa, capture_ops,
    matches, op, ops, report, same_value,
)

captures = (
    matches()
    .from_("add", ops().filter(op("arith.addi")))
    .bind("lhs", bound_ops("add").operand(0))
    .bind("rhs", bound_ops("add").operand(1))
    .where_(same_value(bound_ssa("lhs"), bound_ssa("rhs")))
    .build()
)
native = document.query(captures)
assert native[0]["lhs"] == native[0]["rhs"]

add = capture_ops("m", "add")
rows = captures.project("m", {
    "function": add.root(op("func.func")).attr("sym_name").one(),
    "operation": add.names().one(),
})
full = report(E.scope(
    {"rows": rows},
    {"matches": E.name("rows"), "count": E.name("rows").length()},
))
result = document.query(full)
```

Python returns a list of dictionaries in capture-field order. Operation fields
are `SemanticOperation` wrappers and SSA fields are existing `SemanticValue`
wrappers, retaining their owning document. A standalone `SsaQuery` returns one
`SemanticValue`. Match guards combine with `&`, `|` and `~`; converting one to
`bool` raises `TypeError`.

Capture builders retain root parameter declarations through reuse, erasure,
nesting and projection. Attach declarations with `CaptureQuery.with_parameters`
and supply values through the existing evaluation methods. Parsing, matching,
native copies, scalar projection and Python conversion spend one request
allowance. Invalid scope, native types, fields and effects fail before
enumeration, including for empty sources and unchosen Boolean branches.
`operand` requires one operation and a resolved operand; `same_value` compares
the full result or block-argument identity rather than its defining operation.
Native captures remain read-only and are unavailable on the diff surface.

## Destructuring reports

Rust exposes shared `Pattern`, `PatternKind` and `Located` values.
`ReportExpr::tuple` preserves tuple shape; `ReportExpr::block` takes ordered
`(Pattern, ReportExpr)` bindings and a final value. `ReportExpr::value_match`
takes a subject and `ReportMatchArm { pattern, guard, value }` arms.
`FunctionDefinition::with_patterns` accepts `PatternParameter` declarations;
`FunctionDefinition::new` retains its name-only parameter signature. The
[complete Rust example](../crates/zirium/examples/typed_functions.rs) uses a
tuple parameter and a tail call through a local block and value match.

Python's report builders use reusable `ReportPattern` and `ReportMatchArm`
owners. They accept the existing report inputs and run through `Document.query`:

```python
from zirium.query import ReportExpr as E, ReportPattern as P, ReportMatchArm as Arm
from zirium.query import report

cost = E.name("cost")
query = report(E.value_match(E({"cost": 4}).lookup("cost"), [
    Arm(P.absent(), {"kind": "absent", "cost": 0}),
    Arm(P.present(P.name("cost")), {"kind": "large", "cost": cost}, guard=cost > 10),
    Arm(P.present(P.name("cost")), E.block(
        [(P.name("answer"), cost + 1)],
        {"kind": "small", "cost": E.name("answer")},
    )),
    Arm(P.wildcard(), {"kind": "fallback", "cost": 0}),
]))
assert document.query(query) == {"kind": "small", "cost": 5}
```

`P.tuple([...])`, `P.record({...}, ignore_rest=True)`, `P.literal(value)` and
`E.tuple([...])` cover the same structural forms as the text reader.
`pattern.located(start, end)` retains source coordinates for diagnostics.
`E.conditional(condition, selected, otherwise)` builds a conditional value.
Patterns, arms and expressions refuse Python truth-value conversion. Queries
can be reused against another current document; they retain no active budget.
Python has no function-definition builder or textual-program entry point.

The [language guide](query-language.md#destructuring-and-guarded-value-matches)
describes exact shapes, scope and ordered guards. All branches receive static
checks, while runtime evaluation visits only demanded values. Native capture
enumeration and operation/SSA ownership keep their existing boundaries.

## Building typed functions in Rust

Use Zirium's builders below to define and call typed functions while retaining
checked MLIR operation handles. The
[Zirql embedding guide](https://zayenz.se/zirql/guides/embedding-programs/)
provides more detail on the shared value, pattern and function nodes.

`FunctionDefinition` uses shared `FunctionParameter`, `FunctionType` and value
nodes. It constructs declarations directly; no generated source is parsed.
Attach all definitions to the submitted `QueryExpr` with `with_functions`.
Composition shares identical definitions and retains conflicting definitions
for preflight to reject.

```rust
use zirium::query::{
    FunctionDefinition, FunctionParameter, FunctionType as T,
    FunctionLimits, Located, ReportExpr as E, ValueBinaryOperator as B, report,
};
let parameter = |name: &str| FunctionParameter {
    name: Located::new(name.to_owned(), 0..0),
    declaration: T::Integer,
};
let accumulate = FunctionDefinition::new(
    "accumulate", vec![parameter("n"), parameter("total")], T::Integer, vec![],
    E::conditional(
        E::name("n").binary(B::Equal, E::integer(0)),
        E::name("total"),
        E::call("accumulate", [
            E::name("n").binary(B::Subtract, E::integer(1)),
            E::name("total").binary(B::Add, E::name("n")),
        ]),
    ),
)?;
let query = report(E::call("accumulate", [E::integer(1000), E::integer(0)]))
    .with_functions(vec![accumulate])
    .with_function_limits(FunctionLimits { active_depth: 1 });
```

Run the [complete library example](../crates/zirium/examples/typed_functions.rs):

```sh
cargo run -p zirium --locked --example typed_functions
```

It evaluates the same native tree as the CLI, then the accumulator, and prints
`8` and `500500`. Evaluation and explicit JSON conversion use one budget.

`FunctionType::Native(FunctionNativeType::Operation)` and `SSAValue` describe
checked document-owned IDs. `E::native(&query)` retains a typed query operand;
`E::operation(id)` and `E::ssa_value(id)` supply explicit handles checked on
function admission. `E::operation_sequence(ids)` preserves sequence shape,
including zero or one operation. `E::direct_children` and `E::cost_annotation`
use the same native primitives as the textual reader. `function_call::<T>`
starts a typed native query through the existing `QueryResult` conversion;
`report(E::call(...))` returns `ConstructedReport` for scalar/report values.

Functions never retain the document, caller bindings, result history or active
budget. One request prepares every body and written entry, then invokes those
entries against that request's current context. Native callback arguments
retain a shared owner after a function frame returns. Passing or inspecting
unchanged constructed data shares its immutable owner; constructing a new row
or rendering output still pays for the produced data. External parameters keep
their existing int/string/bool transport. Python keeps its existing query builders and native result conversion; there
is no Python function builder API.

## Building native worklists in Rust

`ReportExpr::loop_`, `break_` and `continue_` construct shared control nodes.
`fold` and `scan` take items, initial state and a named step.
`queue`, `typed_queue`, `pop_front`, `push_back_all`, `identity_set`, `is_in`
and `insert` expose the same computational operations as the textual reader.
Use `FunctionType::Queue(Box::new(element))` and
`FunctionType::IdentitySet(FunctionNativeType::Operation)` in signatures.
`typed_queue` takes a `SignatureType` for an explicit empty element
contract. All builders merge their operands' function and parameter metadata.

The [complete typed reachability example](../crates/zirium/examples/typed_worklists.rs)
constructs destructured native parameters, optional pop arms, a loop and explicit
`user_info`/`same_operation` primitives without generating query text:

```sh
cargo run -p zirium --locked --example typed_worklists
```

It prints the same zero-edge, reachable and disconnected results as the
[text program](../examples/cli/worklist-reachability.zirql).
`evaluate_output_with_budget` retains native computational results. Supply a
retained result with `ReportExpr::native_input(output)` for continued library
use; handles and computational context are checked against the current document.
Use one active budget for evaluation and explicit conversion. Ordinary native
tuples and positional sequences can be converted to JSON. A queue or identity
set anywhere inside a value produces a located conversion refusal.

The shared runtime retains immutable queue/set roots and scan snapshots.
Advancing a queue or inserting an identity preserves previously retained values.
The host adapter supplies exact document owner/revision and checked object
identity, plus complete direct SSA-user adjacency; it runs no second loop or
worklist evaluator. Python keeps its existing query builder API.

## Match projected strings

Rust `ReportExpr` and Python `ReportExpr` provide `contains`, `starts_with` and
`ends_with`. Each method accepts another report expression; Python also accepts
ordinary string arguments. Matching is exact and case-sensitive, empty searches
match, and both operands must evaluate to strings. Unicode text is compared
without normalization. Scans and failure diagnostics use the query budget.

```rust
use zirium::query::{ReportExpr as E, report};
let query = report(E::string("Ångström").contains(E::string("ström")));
```

```python
from zirium.query import ReportExpr as E, op, ops, report

label = E(ops().filter(op("func.func")).attr("sym_name").one())
result = document.query(report(label.starts_with("work")))
```

Select exactly one function before using `one()`. Use `ReportGenerator` to
filter several projected names; the [string-matching example](../examples/cli/string-matching.zirql)
shows the equivalent text program and Unicode controls. Domain name extraction
continues to use native projections such as `attr("sym_name")` and `names`.
