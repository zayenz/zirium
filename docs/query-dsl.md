# Structured queries in Rust and Python

Build queries with methods and predicates, then evaluate them against a semantic
document. Both APIs construct a shared Rust expression tree. Evaluation returns
native values and does not parse query source, print output, or edit the document.
The [textual query language](query-language.md) remains available for the CLI.

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

Expressions are immutable. Assigning an expression to a variable saves the query,
not its result. Each call to `document.query()` evaluates against the current
document. The same expression can run against other documents.

Rust `Document::query()` uses the baseline registry. For other dialect semantics,
use `document.query_with_registry(&expression, &registry)`, passing the registry
used to parse and lower the document. Python retains that registry in the document
and uses it automatically.

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
`subtree`, `slice`, `closure`, and `reachable`. Rust has `defs_at(index)` and
`users_at(index)` for indexed navigation; Python accepts `defs(index)` and
`users(index)`. Indices select the current operation's operand or result,
respectively.

Navigation preserves order and duplicates. A consumer using the same result twice
appears twice in `users()`. Add `unique()` when counting distinct operations.
`union(other)`, `intersect(other)`, and `difference(other)` combine operation or
string queries. Operation sets use document order; string sets preserve first
appearance. Each operand is a complete expression evaluated with the same input.

`head(n)`, `tail(n)`, and `reverse()` apply to streams; `sort()` applies to strings.
These operate on computed streams and do not avoid parsing the document or
executing earlier stages.

Graph traversal has the same structural meaning and dialect limitations as the
textual language. In particular, `closure()` is one scope-retaining dependency
step; `slice()` follows transitive SSA definitions, and `reachable()` also follows
supported calls and branches. For repeated closure, use:

```python
from zirium.query import input

fragment = arithmetic.fixpoint(input().closure())
```

A general `fixpoint(body)` replaces its selection until the ordered stream is
unchanged. It preserves duplicate semantics, detects cycles, and obeys work and
item limits. For accumulating transitive users, use
`arithmetic.fixpoint(input().union(input().users()))`.

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
let histograms = document.query(&report)?; // BTreeMap<String, BTreeMap<String, usize>>
```

`one()` requires exactly one string and returns a scalar. Missing or multiple
strings are errors. `map_by` rejects duplicate keys. `tally()` instead combines
equal strings and returns their occurrence counts. Maps have lexically ordered
keys. Empty inputs produce empty maps.

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
query returns a nonempty selection. Errors in the nested query propagate.

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
| `tally()` | `BTreeMap<String, usize>` | `dict[str, int]` |

Type and attribute queries expose `spellings()` for retained MLIR text.
`string_attr(name)` decodes string attributes, omits missing attributes, and
reports an error for a present non-string attribute or undecodable UTF-8.
Use `attributes(name).spellings()` to inspect those retained spellings.

Rust exposes methods according to the result type: a string query has no `users`
method. Python exposes corresponding classes and type declarations, including
`QueryExpr[T]` for annotating reusable expressions.

Materialized selections keep their membership; handles refer to live document
objects. Later operation edits are visible through operation handles. Deleted or
invalidated objects raise the existing stale-handle errors when accessed. Scalar
and string results retain their evaluated values.

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
selected = document.query(consumers, max_work=1_000_000, max_items=100_000)
```

Rust accepts them on the expression:

```rust
use zirium::query::EvaluationLimits;

let selected = consumers.evaluate(
    &document,
    &registry,
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

Evaluation returns `EvaluationError` in Rust and raises `ValueError` in Python.
Python rejects invalid argument types with `TypeError`. Evaluation runs in Rust
with shared document access and without holding the Python interpreter lock;
queries cannot contain Python callbacks, edits, or emitters.
