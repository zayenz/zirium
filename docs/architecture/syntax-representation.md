# Syntax representation

Zirium owns the source bytes separately from its concrete syntax tree (CST).
Tokens refer to byte ranges, preserving comments, whitespace, malformed syntax,
and invalid UTF-8 without copying each spelling.

The parser stores start, token, and finish events in a four-byte transient
representation. Compaction uses an explicit stack to produce a flat node table.
Each subtree occupies a contiguous preorder range. Parent links are indexed
lazily when requested. Token and node layouts are private implementation details.

## Construction invariants

Parser events reference each lexer token once in source order. This permits
compaction to move the lexer token vector into the CST. The event buffer is
released before the completed node vector is trimmed.

The public `SyntaxTree::from_events` constructor accepts arbitrary token-event
order and uses a token copy and bitmap to validate indices, duplicates,
omissions, source order, and root coverage. Invalid nesting and non-monotonic
token ranges are errors. Compaction itself does not consume call-stack space
proportional to input nesting.

Semantic lowering creates separate storage with document-local interning.
Retention profiles determine whether that document also keeps the source, CST,
and source mappings. See [retention profiles](../getting-started.md#choose-a-retention-profile).

## Query expressions

Text queries and typed builders retain located Zirql `ExpressionStep` values.
`Value` and `Projection` hold ordinary values and correlated scalar fields.
Filter, match, ordering, mapping, existence, fixed-point and path composition
also have direct core steps. Their immutable bodies share `Arc` storage, so
reusing a typed query does not copy its nested expression trees.

A request registers its functions once, then prepares shared expressions through
`PreparedFunctions::prepare_expression`. The prepared entry retains the result
kind and written arguments. Execution supplies bindings, relative input,
document access and the same budget through `evaluate_expression`. Diff programs
use `evaluate_native_expression` with the same prepared native machinery and the
shared statement driver. Both routes retain native results from selector bodies,
including results that also support scalar projection.

The text frontend delegates function collection, parameter and pattern-let
reading, lexical name progression and separators to Zirql's `read_program`.
It moves the result into Zirium's public first/rest expression model without
copying expression trees. Parameter positions and the final-result flag remain
frontend metadata used by existing builders and preparation. Zirium's callbacks
admit native signature types and reserved names, select native versus projected
binding results, and recognize legacy `name =` and `do` syntax.

Zirql owns composition scope and iteration. Filter and selector bodies receive
one subject; a fixed-point body receives the complete current selection. Match
sources and guards see the original input and earlier named captures. Collection
steps own stable unique, tally, bounds and count. Empty capture results retain
their declared schema, and selector ties retain input order.

Zirium's `ProjectionHost` hooks admit native kinds and scalar ordering keys,
expose subjects, restore selection and map shapes, and check exact handle
identity. Guard and path hooks supply MLIR predicates and SSA neighbors. Native
`Stage` payloads retain navigation, dialect facts, edits and emission. The
strict closure specialization stays native because it follows dialect-defined
edges. Diff relation checks retain the before/after side even for empty results.
See the [embedding boundary](https://zayenz.se/zirql/guides/host-integration/)
and [native composition guide](https://zayenz.se/zirql/guides/native-programs/).

Zirium keeps request preparation, frontend metadata, MLIR ownership and
stale-handle checks. Emission payloads retain the requested format and
pass-through behavior; CLI buffering remains a request boundary. Capture records
retain handles, while projected reports contain ordinary Zirql data. The
[typed query guide](../query-dsl.md#nested-queries-and-native-maps) gives a runnable
composition recipe; the [report builders](../query-dsl.md#constructed-report-builders)
show the explicit projection boundary.

## Measure representation costs

```sh
cargo run --release -p zirium --example representation_benchmark
```

The benchmark uses seed `0x5a495249554d0001`, one warm-up, and ten measured runs
per case. It covers token-dense and trivia-heavy streams near 4 KiB, 256 KiB,
and 4 MiB, plus region nesting at depths 8, 64, and 256.

Output includes construction, compaction, traversal, parent-index timing, peak
live allocation, exact retained bytes, and machine/toolchain metadata. Traversal
reads each node's kind, error flag, and token span and passes a checksum to
`black_box`. Use the [processing benchmarks](processing-benchmarks.md) to measure
the complete parser and semantic pipeline.
