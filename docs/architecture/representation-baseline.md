# Syntax representation baseline

Zirium retains the event-to-pre-order representation as its phase-0 correctness baseline. Parser-style start, token, and finish events are compacted with an explicit stack into a flat node table. Subtrees occupy contiguous pre-order ranges. Parent links are omitted until first requested, and then built as a separate lazy index. The physical token and node layouts remain private and provisional.

## Measurement protocol

Run `cargo run --release -p zirium --example representation_benchmark`. The executable uses seed `0x5a495249554d0001`, performs one untimed warm-up and ten measured runs for every case, and prints medians for event construction, compaction, pre-order traversal, parent-index construction, process-scoped peak live allocation, and exact retained representation bytes. It also prints `rustc -Vv`, target, release profile, OS, and CPU metadata.

The fixed matrix is token-dense and trivia/comment-heavy generic-operation streams near 4 KiB, 256 KiB, and 4 MiB, plus nested-region streams at depths 8, 64, and 256.

## Recorded run

Recorded on Apple M1 Max / Darwin 25.5.0, target `aarch64-apple-darwin`, release profile, with `rustc 1.97.1 (8bab26f4f 2026-07-14)`, LLVM 22.1.6, and seed `0x5a495249554d0001`:

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

Traversal reads the kind, error flag, and token span from every visited node and black-boxes an accumulated checksum. Its median rises with the number of visited nodes (about 0.2 ms for 131,073 dense nodes and 0.1 ms for 65,537 trivia nodes). Exact retained and peak-live bytes grow linearly with generated event, node, and token counts. Compaction accumulates descendant errors on the open-node stack, and each finish event propagates its error flag to its parent in constant work. Nested input uses no input-controlled recursion. The nested timing rows are reference measurements, not a claim about current scaling.

## Decision and follow-up rule

The event representation is retained. Invalid event nesting and non-monotonic token ranges are rejected during iterative compaction, and nested input does not use input-controlled recursion. If measurements expose nonlinear retained or peak memory, or later work exposes another correctness limitation, that limitation should become separate reviewed work; this baseline will not silently acquire an alternative builder.
