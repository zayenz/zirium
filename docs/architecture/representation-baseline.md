# Syntax representation baseline

Zirium compacts parser start, token, and finish events into a flat node table
using an explicit stack. Each subtree occupies a contiguous pre-order range.
Parent links are built as a separate index when first requested. Token and node
layouts are private implementation details.

The measurements below record the original representation baseline. They
explain its storage costs; the [processing benchmarks](processing-benchmarks.md)
cover later parser measurements.

## Measurement protocol

Run `cargo run --release -p zirium --example representation_benchmark`. The
executable uses seed `0x5a495249554d0001`, performs one untimed warm-up and ten
measured runs for every case, and prints medians for event construction,
compaction, pre-order traversal, parent-index construction, process-scoped peak
live allocation, and exact retained representation bytes. It also prints `rustc
-Vv`, target, release profile, OS, and CPU metadata.

The fixed matrix is token-dense and trivia/comment-heavy generic-operation
streams near 4 KiB, 256 KiB, and 4 MiB, plus nested-region streams at depths 8,
64, and 256.

## Recorded run

Recorded on Apple M1 Max / Darwin 25.5.0, target `aarch64-apple-darwin`, release
profile, with `rustc 1.97.1 (8bab26f4f 2026-07-14)`, LLVM 22.1.6, and seed
`0x5a495249554d0001`:

| case | events | nodes | tokens | construction ns | compaction ns | traversal ns | parent index ns | peak live bytes | exact retained bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense-4096 | 514 | 129 | 256 | 3,000 | 11,750 | 208 | 458 | 28,984 | 8,200 |
| dense-262144 | 32,770 | 8,193 | 16,384 | 110,875 | 604,792 | 13,583 | 18,208 | 1,851,448 | 524,296 |
| dense-4194304 | 524,290 | 131,073 | 262,144 | 1,976,875 | 12,860,834 | 217,333 | 294,416 | 29,622,328 | 8,388,616 |
| trivia-4096 | 258 | 65 | 128 | 1,459 | 5,500 | 83 | 250 | 14,520 | 4,104 |
| trivia-262144 | 16,386 | 4,097 | 8,192 | 52,875 | 303,167 | 6,792 | 9,084 | 925,752 | 262,152 |
| trivia-4194304 | 262,146 | 65,537 | 131,072 | 1,295,250 | 5,715,042 | 108,667 | 146,625 | 14,811,192 | 4,194,312 |
| nested-depth-8 | 26 | 9 | 8 | 375 | 1,084 | 0 | 208 | 1,280 | 424 |
| nested-depth-64 | 194 | 65 | 64 | 1,208 | 6,625 | 83 | 625 | 10,072 | 3,336 |
| nested-depth-256 | 770 | 257 | 256 | 3,084 | 37,917 | 417 | 1,209 | 40,216 | 13,320 |

Traversal reads each node's kind, error flag, and token span, then passes an
accumulated checksum to `black_box`. It took about 0.2 ms for 131,073 dense
nodes and 0.1 ms for 65,537 trivia nodes in this run. These are historical
measurements; use the processing benchmark to measure the current parser.

## Compaction invariants

Compaction rejects invalid event nesting and non-monotonic token ranges. It
uses an explicit stack, so input nesting does not consume the Rust call stack.
Retained and peak allocation grew linearly with event, node, and token counts
in these fixtures.
