**CLI query language review and follow-up — 2026-09-10**

The follow-up fixes were implemented after revision `d0c9705`. The seven
findings below have been addressed through code or explicit documentation of
the structural representation. The original review is retained below as the
before-state; its implementation descriptions and timings are historical.

- Dot dimension projection retains both sides, and generic printing uses nested
  arrays that reparse without losing either side. The decoder can now be queried
  and tagged directly because `return` shorthand and multi-type returns parse.
- Recovered custom operations produce warnings. `--strict` rejects recovery;
  repeatable `--preset` selects dialects directly. Help, version, preset listing,
  and options after the query are supported. Input files are read one at a time.
- Work and stream-size limits stop growing fixed points with a diagnostic and
  no stdout. Pure `fixpoint(closure)` uses a worklist, and retained subtrees are
  expanded once. Explicit iteration emissions retain their original behavior.
- Attribute projection errors on undecodable byte strings and shares decoding
  with the semantic API. The profiling fixture's duplicate-use expectation is
  corrected. Compact reductions remain structural; this distinction is now
  documented alongside counting and navigation.
- `names`, `operand_types`, `result_types`, `dialect`, and `result_type` cover
  basic inventories and type inspection. Operation JSON includes type arrays.
  Indexed `defs`/`users` and SSA-only `slice` support output-specific inspection
  while stopping dependency traversal at block arguments.

Relationship queries can already be expressed with intersection. The new
[CLI examples](../cli-examples.md) demonstrate ten matmuls with add consumers,
type inventories, and separate decoder output slices. The three return operands
produce slices of 234, 152, and 140 operations. General shape comparisons can use
the type information in JSON; no additional expression language was added for
grouping or numeric analysis. `slice` follows explicit operands and does not
infer region, call, or per-result execution dependencies.

The final measurements ran without concurrent compilation or tests on the same
machine and compiler as the review:

| Case | Before | After |
| --- | ---: | ---: |
| CLI closure, 128 additions | 7.22 ms | 4.82 ms |
| CLI closure, 512 additions | 35.7 ms | 7.55 ms |
| CLI closure, 2,048 additions | 473 ms | 15.29 ms |
| CLI closure, 8,192 additions | 7,392 ms | 50.24 ms |
| Evaluator closure, 1,024 additions | 117.29 ms | 0.244 ms |
| CLI repeated argument-scope expansion, 2,048 additions | 132 ms | 15.56 ms |

The largest CLI closure is about 147 times faster. Ordinary name filtering
remains comparable (77.6 versus 74.9 µs at 7,169 operations). A single closure
step has additional bookkeeping cost (614 versus 680 µs at that width).
The nonconverging subtree probe now reports its work-limit error in about
0.23 seconds. These limits are not byte-memory or wall-clock bounds; buffered
emissions and general fixed points can still be expensive.

Validation passed: 432 workspace tests, one doctest, 130 Python tests (two
skipped), Clippy with warnings denied, Rustdoc with warnings denied, and Rust
and Python formatting/lint checks. The generated harness ran 191 subprocess
probes with 162 explicit checks, including 108 comparisons against twelve
independent StableHLO DAG models. The unmodified profiling command now passes.

The [profiling summary](query-profiling.md) records the measurements and
reproduction commands. Raw CLI and profiling outputs were stored locally under
`target/query-review/`; those generated files are not distributed with the
repository.

**Original review — revision `d0c9705`**

The query language works well for filtering, inspecting attributes, navigating
explicit SSA edges, and making small edits to fully understood input. Its stream
model is consistent and its implementation is small enough to follow. I would
keep that design. The main weaknesses are incomplete StableHLO semantics that
look like successful queries, unbounded duplicate growth in fixed points, and
the cost of deep dependency slices.

The reference documentation is substantially better than the CLI's discovery
experience. The most useful next work is to make partial understanding visible,
repair the demonstrated semantic gaps, and make dependency traversal practical.
A larger general-purpose query language is not needed yet.

**Findings, in priority order**

1. **High: StableHLO attribute projection can silently lose information.**
   The decoder's first dot has `contracting_dims = [0] x [1]`, but
   `attr("contracting_dims")` and operation JSON expose only `[0]`. Attention
   dots with `[2] x [2]` and `[2] x [0]` both project `[2]`. Batching dimensions
   lose the right-hand side too. This prevents a query from distinguishing
   materially different contractions. The generic clause parser captures the
   first attribute value and consumes the remaining tokens without including
   them in that attribute. See
   [clause parsing](../../crates/zirium/src/parser/custom.rs#L615),
   [preset registration](../../crates/zirium/registries/stablehlo.json#L38), and
   [the input](../../examples/cli/stablelm-decode.mlir#L46).
   Preserve the entire paired value, or expose both sides explicitly. An
   attribute-presence test is insufficient here; compare the actual projection.

2. **High: accepted fixed points can grow forever without triggering cycle detection.**
   On `module { %c = arith.constant 1 : i32 }`, `fixpoint(subtree) | count`
   exceeded the harness's one-second timeout. This is not evidence of a slow
   finite computation: the module stays selected and adds another copy of the
   constant on every iteration. Every stream is different, so the
   [cycle detector](../../crates/zirium/src/query.rs#L201) cannot terminate it.
   `fixpoint(subtree | unique)` returns `2` immediately. A true repeating cycle
   does produce the documented error. Add an evaluation limit with a useful
   diagnostic, and document duplicate growth next to fixed-point examples.
   Silently changing all fixed points to sets would change the language's
   existing stream semantics.

3. **High for semantic analysis: the CLI hides recovery that changes answers.**
   The decoder query `filter(op("stablehlo.broadcast_in_dim") and has_attr("dims")) | count`
   returns **0** with the default registry and **31** with the StableHLO preset.
   Transitive users of its matmuls return **144** and **184**, respectively.
   Both variants exit successfully with empty stderr. This is a dangerous
   distinction for scripts: an empty attribute selection can mean missing
   parser knowledge rather than missing attributes.
   [The CLI explicitly accepts unknown-custom recovery](../../crates/zirium/src/main.rs#L121)
   and suppresses those diagnostics. Expose incomplete parsing on stderr and
   offer a strict mode for automation. The query language could also expose an
   `is_unparsed` predicate for inspection, but that alone would not make existing
   scripts reliable.

   Even with the preset, the checked-in decoder's final `return` is recovered
   under that literal name: `filter(op("func.return")) | count` returns **0**.
   Matmul closure fails on unregistered `return`, and tagging the matmuls fails
   with `cannot edit an incomplete document`. Rewriting just that terminator in
   generic quoted `func.return` syntax allows both queries to succeed. Merely
   qualifying the custom spelling as `func.return` still produces syntax errors
   on its multiple type operands. This is a concrete compatibility gap in the
   project's own example, not a query-composition failure.

4. **Medium: closure repeatedly visits old work, including already-retained scopes.**
   Deep chains show approximately quadratic time. The measured CLI slice grows
   from **0.47 s at 2,048 additions to 7.39 s at 8,192**, while count grows from
   **16 ms to 50 ms**. Besides generic fixed-point reevaluation,
   [closure](../../crates/zirium/src/query.rs#L579) scans document order on every
   iteration. [Scope retention](../../crates/zirium/src/query.rs#L694) traverses
   a subtree again even if it already retained that scope. A 2,048-operation
   chain repeatedly using one function argument takes **132 ms**, despite
   reaching the complete function in very few expansion steps.
   First avoid repeated subtree expansion within one closure evaluation. Then
   consider a worklist implementation for the common `fixpoint(closure)` case.
   Preserve error checking and per-iteration emissions for general fixed points;
   don't add an optimizer framework to solve this one case.

5. **Medium: the documented profiling command currently fails.**
   `cargo test --release -p zirium --test query_profile -- --ignored --nocapture --test-threads=1`
   fails with `users: left 80, right 64`. In the generated fixture, the first add
   consumes the same constant twice. `users` correctly preserves both use sites,
   so the [expected value](../../crates/zirium/tests/query_profile.rs#L41) should
   be `width * (depth + 1)`. I ran a temporary copy with only that expectation
   corrected; all its checks passed. The published performance numbers describe
   an older revision and should not substitute for a working benchmark.

6. **Medium: projecting a byte string silently drops a present attribute.**
   On `module { "vendor.thing"() {tag = "\FF"} : () -> () }`,
   `attr("tag") | count` returns **0**. The attribute exists, but
   [decoding](../../crates/zirium/src/query.rs#L536) requires UTF-8 and
   [projection](../../crates/zirium/src/query.rs#L465) uses `filter_map`.
   This contradicts the documented rule that projection drops operations
   *without* the attribute. Report a decoding error or preserve an escaped
   representation; silently deleting the value makes counts misleading.

7. **Medium: compact and explicit reductions expose different operation graphs.**
   An explicit reduction fixture has **7** total operations and **2** children
   under its reduction. Its compact `applies stablehlo.add` equivalent has **5**
   operations and **0** reduction children. The
   [parser deliberately accepts it as regionless](../../crates/zirium/src/parser/custom.rs#L743).
   Consequently, counting reducer arithmetic or extracting reducer bodies
   depends on the assembly spelling. Supporting a structural textual view is
   reasonable, but document that this does not reconstruct implicit operations.
   The decoder's documented **237** count is a Zirium structural count, not a
   normalized semantic operation count.

**What works and what is pleasant to use**

The normal `cargo test -p zirium --quiet` run passed **426 tests**, with two
ignored profiling tests. The new CLI harness completed **176 subprocess probes**
plus eight median summaries. **141 probes had explicit expected-result checks**,
including **96 graph comparisons** across twelve generated 60-node StableHLO
DAGs. The remaining probes record limitations rather than treating those
limitations as the desired behavior.

The independent graph model agreed on ordered `defs`, ordered `users`,
deduplication, union, intersection, difference, dependency closure, and
accumulating transitive users. Generated explicit reductions also passed their
structural checks. Flat pipelines and boolean chains of 10,000 terms completed;
excessive nesting produced a positioned diagnostic instead of a crash.

`filter(...) | users | unique` reads naturally. Pipes, grouping, boolean
predicates, and operation sets compose predictably. Attribute edits retain the
selection, `input` restores the full document, and JSON makes shell integration
possible without another output language. Buffered CLI output is a useful
all-or-nothing contract. The parser's flat vectors and bounded recursive nesting
are sensible implementation choices. SSA users use a lazy index rather than
rescanning the document for every result.

The largest usability costs are discovery and multiplicity. `--help` currently
returns `unknown option: --help`; options must appear before the query, and the
CLI offers no direct preset-name flag. Counting users counts use occurrences,
including one consumer that uses a value twice. This is documented, but examples
should usually include `unique` when the question is about distinct operations.

**Measured efficiency**

Release build, Apple M1 Max, arm64, rustc 1.98.1. The corrected existing benchmark
uses seven warmed in-process samples and excludes parsing/lowering and printing.
At 7,169 operations, median evaluation times were:

| Query | Median |
| --- | ---: |
| Direct name scan | 32.5 µs |
| Count | 23.7 µs |
| Name filter and count | 77.6 µs |
| Boolean filter | 230 µs |
| Users and count | 237 µs |
| Union of navigations | 964 µs |
| Subtree and filter | 358 µs |
| One closure step | 614 µs |
| Fixed-point closure, shallow functions | 3.45 ms |

Ordinary width scaling is approximately linear; name filtering costs about
2.4 times a direct scan including stream construction. There is no evidence
here that ordinary filters need substantial optimization.

The CLI stress harness uses one warmup and three measured subprocess runs per
chain case. These medians include startup, query-file reading, MLIR parsing,
lowering, evaluation, and output; fixture generation is excluded.

| Chain additions | Count | Dependency slice |
| --- | ---: | ---: |
| 128 | 5.31 ms | 7.22 ms |
| 512 | 7.20 ms | 35.7 ms |
| 2,048 | 16.0 ms | 473 ms |
| 8,192 | 49.8 ms | 7,392 ms |

The warmed evaluator benchmark independently confirms depth scaling: 16, 128,
and 1,024 additions took 0.038, 1.84, and 117 ms. Small subprocess timings should
not be read as precise evaluator costs. Peak memory was not measured. Code
inspection also shows all input files and all emitted output are buffered;
large multi-file jobs and `fixpoint(... | emit)` can therefore consume substantial
memory. That follows the documented atomic-output policy, but input files could
still be read one at a time.

**Useful decoder queries and the missing pieces they reveal**

Run these with `--registry crates/zirium/registries/stablehlo.json` before the
query argument and `examples/cli/stablelm-decode.mlir` after it.

| Question | Query | Observed result |
| --- | --- | --- |
| Which functions contain matmuls? | `filter(op("stablehlo.dot_general")) \| root(op("func.func")) \| unique \| attr("sym_name") \| json` | `["main"]` |
| Which matmuls have batching dimensions? | `filter(op("stablehlo.dot_general") and has_attr("batching_dims")) \| count` | 4 attention-related dots |
| Which operations immediately consume matmuls? | `filter(op("stablehlo.dot_general")) \| users \| unique \| count` | 21 distinct consumers |
| What is downstream of the matmuls? | `filter(op("stablehlo.dot_general")) \| fixpoint(filter(true) union users) \| count` | 184 structurally reachable operations |
| How are broadcasts shaped? | `filter(op("stablehlo.broadcast_in_dim")) \| attr("dims") \| unique \| json` | `[]`, `[1]`, `[0]` spellings |
| Where are KV-cache updates? | `filter(op("stablehlo.dynamic_update_slice")) \| count` | 4 updates |

The downstream count describes explicit SSA reachability, not execution
dependence through arbitrary region or call semantics. Closure is also coarse
at block arguments: retaining their owning function includes its whole body.
After the decoder terminator is expressed generically, matmul closure selects
**236 of 237 operations**. That is useful scope retention but provides little
isolation of an individual attention or MLP computation.

I would prioritize these additions:

- **Type and shape inspection:** expose operand/result types and a small set of
  predicates for element type, rank, and dimensions. This enables finding f32
  matmuls, large intermediates, or unexpected conversions. Operation JSON
  currently contains only name and attributes, so an external tool cannot
  recover this information either.
- **Operation-name projection and simple dialect matching:** make an operation
  histogram or dialect inventory easy. Today JSON plus `jq` can build the
  histogram; exact-name `or` chains are cumbersome for dialect-wide filtering.
- **Relationship predicates:** an existential subquery would express “matmuls
  consumed by a transpose” while retaining the matmul selection. Navigation
  alone discards the originating operation; a general pattern language would
  be a much larger commitment.
- **Result/operand-aware slicing:** useful for following logits separately from
  the two cache results, and for stopping at function inputs instead of expanding
  the whole function. Start with a concrete slicing operation rather than
  general variables or user-defined functions.

Numeric attribute comparisons would also help, but complete attribute values
and type inspection should come first. Grouping, sorting, and arithmetic can
remain external JSON processing until real workflows justify them.

**Documentation and reproducibility**

The [language reference](../query-language.md) clearly explains precedence,
duplicate preservation, set inputs, selection versus printing, fixed-point
replacement, edits, and emissions. Keep that material. Add a short CLI quick
reference, preset setup directly in the StableHLO examples, recovery visibility,
compact-reduction behavior, and the nonconverging-duplicate example. Document
that value-set operations preserve first appearance rather than source order:
the probe selecting value `2` union value `0` returns `["2", "0"]`.
The profiling document also lists a “root and filter” row although the current
benchmark case is subtree and filter.

Reproduce the generated probes with:

```sh
cargo build --release -p zirium --bin zirium
mkdir -p target/query-review
python3 python/benchmarks/query_language_review.py > target/query-review/results.json
```

The [harness](../../python/benchmarks/query_language_review.py) is standard-library
Python and creates fixtures and query files temporarily. Local outputs from this
review were `target/query-review/results.json`, `profile.txt`, and
`profile-corrected.txt` in the same directory. These generated files are not
tracked; use the harness to produce results for the current checkout.
To reproduce the latter, copy `query_profile.rs` to a temporary integration-test
file, change only the users expectation to `width * (depth + 1)`, run that test
with the same release flags, then remove the copy.

No query implementation or existing tests were changed. These measurements
review the local CLI and structural parser; they do not establish full StableHLO
verification or executable equivalence of emitted fragments.
