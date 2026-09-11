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
