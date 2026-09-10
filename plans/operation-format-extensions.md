# Operation format extensions

Implementation plan, 2026-09-10. Implemented in the current working tree.
Based on the supplied extension
requests and inspection of the current Rust parser, registry, lowering, tests,
and Python configuration. The requester's private corpus was not available for
validation, so its two type-sharing assumptions still need confirmation.

Extend the existing format interpreter into a small compositional language.
Keep shapes as convenient structural grammars, preserve existing format
semantics, and make alternatives explicit within one registration. Type
bindings must describe how printed types apply to operands and results; their
meaning must not depend on the operation name or separator spelling.

## Findings that change the proposed solution

The request correctly identifies the two-template restriction. In
`crates/zirium/src/dialect/format.rs`, scanning already produces a vector of
steps, but validation requires six directives in one of two arrangements.
Removing that restriction alone is insufficient: scanning inserts syntax nodes
specifically for conversions and constants, and `lower_operation_format` in
`crates/zirium/src/dialect.rs` assumes exactly those two cases.

Several details in the request need correction before implementation:

- **P1 touches parsing and lowering.** `formatted_operation` explicitly matches
  `to` and `into`; it does not reuse the shape separator matcher. Lowering also
  selects between those two words. Adding an enum variant alone will not work.
- **P2's recommended count rule does not cover `select`.** Three operands and
  two printed types satisfy neither “one shared type” nor “one per operand.”
  The intended sharing must be expressed explicitly. Equal `i32` spellings in
  the example hide which operands each type describes.
- **The missing result clause is a semantic question.** The request's
  `round_stochastic` example binds a result but prints only two operand types.
  A list parser cannot determine its result type. Likewise, deleting the result
  clause from a conversion format does not say that results share operand types.
- **P1 does not make operand arity exact.** `$operands` remains 0..N. Replacing
  `binary_operands` with that format tightens the separator but loosens operand
  count. Exact binary syntax needs a way to capture two operands explicitly.
- **P3's binding rule needs exceptions.** Results are bound by the operation's
  SSA header, not earlier directives in its body. Conversely, `$value` is an
  inline attribute, not an SSA operand. These roles must remain distinct.
- **Not every adjacent directive is ambiguous.** The existing
  `$operands attr-dict` pair is valid because `%` and `{` distinguish them.
  Validate actual capture boundaries, rather than banning adjacency wholesale.
- **Round-trip claims refer to different output paths.** CLI `emit` can retain
  source spelling; that does not establish an inverse custom printer for
  arbitrary formats. Preserve the existing generic semantic-print fallback.

The operand format also currently accepts a parenthesized operand type list:
the parser calls `type_list` for both operand and result bindings. Preserve this
alongside the documented single shared operand type.

## Recommended public design

### 1. Composable sequences with a deliberately small vocabulary

Retain `operation_formats` and its string entries. Compile descriptions once
when constructing the registry. Support sequences built from:

| Element | Meaning |
| --- | --- |
| Backtick literal | Exact token spelling, including identifiers and punctuation. |
| `$operands` | Existing variadic comma-separated SSA operands. |
| `$operands[i]` | One SSA operand at a specified zero-based position. |
| `$value` | Existing inline literal attribute, lowered as `value`. |
| `$callee` | A symbol reference, lowered as the `callee` attribute. |
| `attr-dict` | Attribute dictionary, absent or present in input. |
| `type(...)` | A type capture with explicit semantic targets, described below. |
| `types($operands)` | A bare comma-separated list with one type per operand. |

Literals should use the MLIR lexer to validate token spellings. Support
punctuation sequences such as `=>` when they form valid lexer tokens; reject
comments, whitespace-bearing literals, malformed strings, and empty literals.
Match token spelling as well as kind, so `as` cannot match any identifier.
An unsupported character is a load error, not a new lexer feature request.

Allow `attr-dict` to be absent from the description or appear once wherever its
boundary is unambiguous. Distinguish an optional dictionary in the input from an
optional directive in the description. Retain the operation boundary and
trailing-location handling shared with existing parsing.

For the first implementation, require indexed operand captures to appear once,
in contiguous order starting at zero. Do not mix them with `$operands` in the
same format. This supplies fixed arity without adding an arity schema or named
operand declarations. Arbitrary reordering, slices, named operands, repetition,
optional groups, regions, and ODS import can wait for concrete use cases.

### 2. Explicit type sharing; preserve old spellings

Keep the current meaning of existing directives:

- `type($operands)` accepts the existing single type or parenthesized type list.
  A single type broadcasts over the operands; retain zero-operand behavior.
- `type($results)` accepts a single result type or a parenthesized result list.
- `type($result)` captures one result type, as in the current constant template.
- `type($value)` is the inline attribute's type, not the operation's result type.

Add `types($operands)` for an explicitly per-operand bare list. Do not silently
widen `type($operands)` to consume commas: this would change the boundaries of
existing directives and still would not solve `select`.

Extend `type(...)` to accept multiple targets. A multi-target capture consumes
exactly one type and assigns it to every listed target. An aggregate target in
that form means all members of that group. For example:

```text
# Exact binary comparison, one shared input type and an explicit output type:
$operands[0] `,` $operands[1] attr-dict `:` type($operands) `->` type($results)

# Three operands, with the second printed type shared by operands 1 and 2:
$operands[0] `,` $operands[1] `,` $operands[2] attr-dict `:` type($operands[0]) `,` type($operands[1], $operands[2]) `->` type($results)

# One printed type shared by operands and results:
$operands attr-dict `:` type($operands, $results)

# Distinct operand types followed by explicit result types:
$operands attr-dict `:` types($operands) `->` type($results)
```

The `select` mapping above assumes the two printed types mean condition type
and branch-value type. Confirm that interpretation against the dialect's actual
definition; do not infer it from two identical `i32` strings.

If `round_stochastic` returns the first operand's type, it can use:

```text
$operands[0] `,` $operands[1] attr-dict `:` type($operands[0], $results) `,` type($operands[1])
```

That return-type relationship also needs confirmation. The format language
should require the relationship to be stated, whichever it is. A missing result
type on an operation with SSA results must produce a lowering diagnostic;
there is no universal “use the last printed type” convention.

Validate target existence, duplicate assignments, incompatible capture modes,
and obvious delimiter ambiguity at load time. Validate runtime group sizes and
type assignments against actual operands and SSA results during lowering.
Preserve old-template behavior separately from any new checks that would reject
previously accepted input; tightening legacy validation is a separate change.

### 3. One registration containing explicit alternatives

Add an optional top-level `operation_alternatives` list, defaulting to empty:

```json
{
  "builtins": [],
  "operation_shapes": [],
  "operation_alternatives": [
    {
      "name": "op.arg_in",
      "alternatives": [
        {"format": "$value `:` type($value) attr-dict `:` type($result)"},
        {"shape": "operand_clauses"}
      ]
    }
  ]
}
```

Each alternative contains exactly one `shape` or `format`. Require at least
two, disallow nested alternatives, and reject unknown fields. This supports
shape/shape, format/format, and mixed cases without overloading either existing
entry type or adding alternation syntax to the format language.

Try alternatives in local order and select the first complete syntactic match.
A recovered operation or one carrying syntax errors is not a match. Once syntax
selects a branch, a lowering failure stays a lowering failure; do not switch
grammars based on whether SSA values resolve.

Preserve duplicate errors within a configuration and conflicts across files.
An identical ordered alternatives entry can be deduplicated across files;
reversing the order is a conflicting definition. File order never selects a
winner. Built-in collisions retain their existing restrictions.

Both parser and lowerer must use the selected alternative. Store its index on
the parsed operation using a small internal annotation; choose its concrete
storage after checking the compact-tree conventions. Do not guess it later by
trying lowerers. Reuse it through hybrid/source-preserving paths and refresh it
on reparsing. Semantic generic printing need not remember it.

The existing checkpoints already record token position, builder events,
diagnostics, and nesting depth. Refactor shape and format parsing to expose a
non-recovering attempt, restore all state after a failed attempt, and recover
only once if every branch fails. Report the furthest failure, with ties resolved
by declaration order. Fatal resource errors must propagate, not trigger fallback.

This is medium-sized work: changing the registry value to a vector is only one
part of it. Keep the existing single-grammar path cheap.

### 4. Symbol references with a defined role

Use `$callee attr-dict` for P5 and reuse the existing symbol-path parser,
including quoted names and nested references. This operation has no operands or
results and lowers to `() -> ()`, with its symbol retained as `callee`.

Prefer `$callee` to an underspecified `$symbol`: symbol references and symbol
definitions have different meanings. This convention does not register vendor
dependency analysis or make `closure` follow the reference automatically.
Arbitrary attribute names for captured symbols are outside this first extension.

## Implementation sequence

| Step | Work and completion criterion | Main files |
| --- | --- | --- |
| 1 | Ship `->` through compilation, parsing, and lowering. Document the finite templates as they exist at that release; return a useful load error listing them. A comparison lowers with the correct inputs and outputs and rejects `to` when registered with `->`. | `dialect/format.rs`, `parser/custom.rs`, `dialect.rs`, `docs/custom-formats.md` |
| 2 | Replace template validation and template-specific type handling with composable captures. Add general literals, indexed operands, shared type targets, explicit type lists, and `$callee`. Complete parser-to-lowering examples before broadening the grammar further. | Same Rust files, `semantic/lowering.rs`, syntax kinds/accessors as needed |
| 3 | Add explicit alternatives, transactional attempts, and selected-branch lowering. Both `arg_in` spellings parse, lower according to the chosen grammar, and round-trip through supported output paths. | `dialect/config.rs`, `dialect.rs`, `parser/grammar.rs`, `parser/custom.rs`, `semantic/lowering.rs` |
| 4 | Finish public integration and migration examples. Match Rust/Python schema behavior, document inspection of alternatives and precise diagnostics, and exercise CLI loading and emit. | `python/zirium/config.py`, Python exports/stubs, `crates/zirium-python/src/registry.rs`, docs and integration tests |

Paths in the table are relative to `crates/zirium/src` unless otherwise stated.
Steps 2 and 3 are the substantial changes. Step 3 can follow step 1 directly if
fixing the reported round-trip failure takes priority. Schema and API updates
required by each step should land with it; step 4 is the final integration pass.

For step 2, retain the existing step vector and introduce explicit capture nodes
for types instead of wrapping a whole textual trailer in `FunctionType`.
Associate captures in source order with their compiled targets. Build the
normalized function signature from those captures during lowering, avoiding
separator searches through source text. Adapt the legacy templates to that same
path while preserving their behavior. Reuse existing type, attribute, operand,
and symbol subparsers; no parser generator or general grammar framework is needed.

Update registry equality/fingerprints for literal text and ordered alternatives.
Continue supporting existing registry inspection; add a direct alternatives
inspection method rather than making `operation_shape()` return a misleading
single shape. Follow the existing Rust/Python ownership rules for registries.

Replace `OperationFormat::parse`'s undifferentiated failure with a structured
error carrying the description offset and reason. Surface operation name,
description, and expected token/directive through Rust, CLI, and Python. At
runtime include the failed format or alternative and the actual failure span.
Once sequences compose, stop listing finite templates as the complete language.
Keep Rust as the authoritative semantic validator; do not duplicate it in
Pydantic or invent a second validation schema for format strings.

## Focused validation and migration

Extend existing tests instead of translating every probe row into a separate
test. The valuable cases cross parsing and semantic boundaries:

1. Existing `to`, `into`, and typed-constant forms retain semantic fields,
   zero operands, attributes, parenthesized lists, and newline boundaries.
2. Arrow parsing with nested arrows/commas inside opaque and aggregate types
   assigns the correct input/result types; a wrong separator is rejected.
3. Fixed binary arity rejects extra operands. A three-operand/two-type case uses
   distinguishable input types to prove sharing, not just successful parsing.
   Include missing result mapping and wrong list length diagnostics.
4. Alternatives select the correct lowerer. A partially consumed failed branch
   leaves no nodes or diagnostics behind; all-branch failure preserves the next
   operation. Check duplicate and cross-file ordering rules.
5. A callee-only operation retains its full symbol path and attributes in generic
   semantic output. It acquires neither result types nor symbol-definition status.
6. One unregistered generic fixture with regions, block arguments, attributes,
   and nested queries works both with and without a same-name custom registration.
7. Check CLI source-preserving emit separately from generic semantic print and
   reparse. Compare semantic structure and fields, not only operation counts.

Use the existing `operation_formats.rs`, `dialect.rs`, `cli.rs`, and Python
registry tests. During implementation run the focused Rust tests, then the
workspace checks and Python registry tests against the rebuilt extension.
Exercise rollback with malformed input in the existing parser fuzz setup; no
new benchmark project or exhaustive cross-product suite is warranted.

Keep bundled shape registrations and their accepted syntax unchanged. Publish
migration examples for comparisons, shared-type operations, `select`, alternatives,
and callee-only calls. Do not mass-convert presets as part of this work.

Before claiming the requester's full coverage, obtain the actual type-sharing
rules for `select` and `round_stochastic`, plus the failing two-mode fixture and
expected semantic fields. In particular, `operand_clauses` provides structural
support for `dim(...)`; alternatives do not turn it into a dialect-specific
interpretation of that clause. Those details gate the coverage claim, not the
generic implementation.
