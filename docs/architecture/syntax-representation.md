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
`Value` holds an ordinary value expression, including record construction;
`Projection` holds correlated scalar fields. Both share their immutable bodies.
The remaining `Stage` payloads contain Zirium's native operations and the
composition stages awaiting migration. Their `Arc` storage also makes typed
query reuse share nested trees.

A submitted request registers its functions once, then prepares each shared
expression through `PreparedFunctions::prepare_expression`. The prepared entry
retains its result kind and written arguments. Execution supplies fresh bindings,
relative input, document access and the same budget through `evaluate_expression`.
Zirql checks projection fields before enumerating subjects and retains declared
empty row schemas. The [projection contract](https://zayenz.se/zirql/guides/host-integration/#correlated-native-projection)
describes the subject and scalar-value hooks.

Zirium keeps statement and parameter bookkeeping, native-only stream execution,
MLIR ownership and stale-handle checks. Emission payloads retain the requested
format and pass-through behavior; CLI buffering remains a request boundary.
Native capture records retain handles, while projected report records contain
ordinary Zirql data. See the [typed report builders](../query-dsl.md#constructed-report-builders)
for the public Rust interface.

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
