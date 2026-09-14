# Structural editing capability matrix

This page is a deliberately narrow map of the structural transformations
currently exposed by each interface. Zirium is not a general IR-construction
library: edits operate on a complete semantic document. Identity-bearing
handles (operations, blocks, and values) belong to that document, while Rust
`TypeSpec` and `AttributeSpec` are arena-independent descriptions.

| Capability | Rust semantic editor | Python semantic edit | CLI |
| --- | --- | --- | --- |
| Insert operation | Yes: root or existing block; [Rust insertion test](../crates/zirium/tests/editing.rs#L655). The `OperationSpec` must be regionless. | Yes: root or existing block; [`insert_root` example](getting-started.md#write-output). Operand/value wrappers must be same-document; specs contain existing wrappers. | No operation insertion. |
| Erase operation | Yes, only a regionless operation with no live uses; [Rust erase test](../crates/zirium/tests/editing.rs#L531). | Yes, with the same regionless/no-live-uses restrictions; [Python erase test](../python/tests/test_editing.py#L132). | No operation erasure. |
| Move operation | Not supported. Insert a new regionless operation or edit attributes instead; operation identity and ordering are not movable. | Not supported; there is no move command. | Not supported. |
| Regions and blocks | Read and traverse existing regions/blocks; [Rust structure test](../crates/zirium/tests/editing.rs#L655). Insertion can target an existing block, but creation, deletion, and movement are not expressible. | Read and traverse existing regions/blocks; [Python traversal example](../python/tests/test_semantic.py#L35). Insertion can target an existing block; no creation, deletion, or movement. | Queries can inspect nested structure, but CLI edits do not change regions or blocks; see [nested traversal](query-dsl.md#projections-and-editing). |
| Signatures and result types | Result types can be replaced with checked [`TypeSpec`](https://docs.rs/zirium/latest/zirium/semantic/struct.TypeSpec.html) values; [Rust replacement test](../crates/zirium/tests/editing.rs#L459). `TypeSpec` and `AttributeSpec` are arena-independent; only operand/value handles must come from this document. | Result types can be replaced with checked same-document wrappers; [Python replacement test](../python/tests/test_editing.py#L378). Fresh type strings are not parsed by edits. | No signature editing. |
| Successors | Successor arguments can be rewired; [Rust successor test](../crates/zirium/tests/editing.rs#L749). Creation, removal, and target movement are unsupported. | Successor arguments can be rewired; [Python successor test](../python/tests/test_editing.py#L340). No creation/removal/target movement. | No successor editing or successor projection in the CLI query language. |
| Locations | Existing locations can be inspected and printed; [Rust location test](../crates/zirium/tests/semantic.rs#L1140). Location creation or replacement is unsupported. | Locations are available at the syntax level via [`trailing_location`](../python/tests/test_semantic.py#L1060); semantic operations have no location accessor. Location editing is unsupported. | Locations are retained/emitted as part of MLIR text but are not query-reportable or editable. |
| Construction | `OperationSpec` constructs only a regionless operation from existing operand handles plus arena-independent [`TypeSpec`](https://docs.rs/zirium/latest/zirium/semantic/struct.TypeSpec.html) and `AttributeSpec`; [Rust insertion test](../crates/zirium/tests/editing.rs#L655). It is not general IR construction. | `OperationSpec` constructs only a regionless operation from same-document wrappers; [Python insertion example](getting-started.md#write-output). It returns no operation handle. | No IR construction; query literals only construct report values, not operations. |
| Provisional handles | Rust `insert` returns an `OperationId` usable within the open transaction; [rollback/stale-handle test](../crates/zirium/tests/editing.rs#L2244) shows it is stale if the edit is dropped or fails. | No provisional operation wrapper is returned. Look up the inserted operation after commit; buffered edits roll back when the context exits with an exception. | No semantic handles. |

## Transaction and output consequences

Rust `Document::edit` and Python `document.edit()` work on a private copy and
publish changes only after validation succeeds. Failed or abandoned edits do
not partially update the document. Python commands are buffered until the
context exits normally. Identity-bearing values, operations, and insertion
blocks must belong to the edited document; stale and foreign handles are
rejected. Rust `TypeSpec` and `AttributeSpec` are arena-independent and may
describe fresh valid spellings; Python wrapper specs require same-document
wrappers where they carry identity.

Insertion and erasure invalidate source mappings. Consequently, a document
that has undergone either operation cannot use preserving output. Use
canonical output for the edited semantic document, or retain a separate
`ParsedFile` if byte-for-byte original bytes are needed. Attribute edits on a
hybrid document can still use preserving output; the [output-mode guide](../README.md#output-modes)
shows the distinction.

The CLI's `set_attr` and `remove_attr` commands edit selected operations and
emit a complete document with `do ...; emit`; they do not construct or
rearrange IR. See the [CLI tag example](cli-examples.md#tag-selected-operations)
and the [CLI remove-attribute example](cli-examples.md#remove-a-tag-while-reading-standard-input).
Each edit stage commits atomically, all emissions across statements and inputs
are buffered, and stdout remains empty if a later query, evaluation, or
printing step fails.

The [Python erasure test](../python/tests/test_editing.py) shows a concrete
regionless erase and the stale-handle error after commit.
