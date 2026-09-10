# Review for the 0.1.0 release

Reviewed on 2026-09-10, including the query and skill changes completed during
this review. The package version remains 0.0.13. This review prepares the code
for the release; it does not create a release tag or publish packages.

The repository is suitable for a scoped 0.1.0 release once the release candidate
passes the full CI matrix. The separation of lossless syntax, semantic storage,
registry knowledge, and output contracts is sound. The largest demonstrated
problem was CLI printing cost, which is fixed. The remaining limitations should
stay explicit in the release description: selected custom syntax, structural
analysis, and bounded editing are the supported contract.

The review covered the Rust library, CLI and query implementation, Python
bindings and declarations, custom-format configuration and presets, existing
corpus and regression coverage, agent skills, documentation, and packaging and
release workflows. It combined source inspection, executable examples,
regression tests, generated graph checks, short fuzz runs, and local package
builds. It did not independently verify every dialect against upstream tools.

| Area | Assessment |
| --- | --- |
| Code organisation | Clear ownership boundaries and useful modules. The large grammar and semantic-value files are now split by responsibility, preserving their existing interfaces and algorithms. |
| Rust interface | Explicit and predictable. Registry and retention choices are verbose but meaningful. Checked handles and atomic edit commits provide useful guarantees. |
| Python interface | Convenient for inspection, with typed wrappers and packed tables for bulk access. Editing is deliberately narrower than the Rust API. |
| CLI | Useful for scripts and interactive analysis. Diagnostics, strict mode, program files, and predictable output contracts are strong. Standard input and pipe handling are corrected. |
| Query language | The ordered-stream model composes well. Bindings, set operations, projections, maps, and reports have clear roles. Duplicate and scope semantics need the existing examples. |
| Efficiency | Common selection printing is substantially faster. Worklist dependency traversal is appropriate. Evaluation remains eager, and some specialised printing and nested query paths can still have quadratic costs. |
| Custom formats | A practical, shared configuration model with early validation and explicit alternatives. Shapes and formats describe supported syntax rather than full dialect semantics. |
| Agent skills | Useful guidance on decisions agents commonly get wrong. Companion skills are now optional, references are easier to find, and version selection is consistent. |
| Documentation | Coherent references and examples, with candid compatibility boundaries. Release lockfile instructions and skill discovery needed correction. |

**Fixes made during the review**

- **Printing complexity:** selected custom output rebuilt the full SSA-name
  replacement map for every operation, using a document scan for every name.
  It now derives replacements once from the printer's existing canonical-name
  map. Trailing-comment handling also scanned every token for every operation;
  it now locates the end with binary search and walks backwards over trivia.
- **Invalid selections:** `Document::write_selection` silently discarded foreign
  or erased handles. It now rejects them before touching the output sink. A
  regression checks a mixed valid/invalid selection so partial success cannot
  hide the mistake.
- **CLI integration:** arguments retain native path bytes, `-` denotes MLIR
  stdin, and a downstream closed pipe exits quietly. Help, version, preset
  listing, and normal output share the same write path. Native byte filenames
  are tested on Linux, where the filesystem supports them; closed pipes are
  tested on Unix.
- **Query costs:** ancestor visits in `root` consume the work budget. Operation
  extrema select their keys with a linear scan instead of sorting all keys,
  preserving the first tie or every tie in input order.
- **Python declarations:** `SemanticType.kind` includes `"complex"`, matching
  the runtime. The pending registry assertion is formatted so Ruff passes.
- **Release and documentation:** version changes must refresh both lockfiles.
  README guide links work from package registries, and it now lists the skills.
  Query docs explain eager evaluation, initial-stream limits, concatenated JSON
  results, and buffered output. Skills have independent fallback references and
  respect the custom-format skill's explicit invocation policy.

These are focused corrections. The review preserves the existing representation,
query grammar, public editing model, and registry design.

**Module organisation**

The reorganisation groups the existing implementation by responsibility:

- `parser/grammar.rs` retains parser entrypoints, operation and region parsing,
  and token access. Its `attributes`, `types`, and `recovery` submodules hold
  attribute syntax, type and affine syntax, and diagnostics and recovery.
- `semantic/values.rs` retains shared text helpers, alias expansion state,
  symbol handling, and SSA resolution. Its `types`, `attributes`, `locations`,
  and `affine` submodules hold the corresponding lowering code.

Both module families remain private. Public entrypoints and lowering interfaces
are unchanged. Existing function bodies were checked against the originals,
allowing only visibility and formatting differences.

**Printing measurements**

The fixture is a module containing independent `arith.constant` operations.
Each measurement runs the release CLI with `--strict` and an empty query,
checks the emitted operation count, and captures output. Results include process
startup, parsing, lowering, query evaluation, printing, and pipe I/O. One warmup
precedes three measured runs; the table reports their medians.

Measured on Apple M1 Max, macOS arm64, rustc 1.98.1, without concurrent builds or
tests. The baseline binary predates the printing fixes; the final binary includes
changes through `99f09ec`.

| Constants | Before, ms | After, ms |
| ---: | ---: | ---: |
| 128 | 8.47 | 4.32 |
| 512 | 173.82 | 5.94 |
| 1,024 | 1,247.66 | 7.97 |

The largest case is about 157 times faster. This establishes the improvement
for this workload; it is not a throughput guarantee for every dialect or query.
The benchmark is intentionally small and has no timing threshold:

```sh
cargo build --release -p zirium --bin zirium
python3 python/benchmarks/selection_printing_benchmark.py
# An optional positional binary path permits before/after comparisons.
```

**Interfaces and limits worth keeping visible**

Rust callers must pass the same registry through parsing, lowering, verification,
and custom printing. Python retains it automatically. Library parsing defaults
to the empty registry, while the CLI defaults to baseline; the registry guide
already explains this distinction. Strict lowering establishes complete
structural information for supported input. Full registered verification remains
an explicit operation.

Python's `AttributeSpecHandle` and `OperationSpec` snapshot existing values.
There is no equally convenient general constructor for an arbitrary fresh type
or attribute, and buffered insertion does not return a handle for reuse within
the same transaction. This is workable for inspection and targeted edits, but
should remain a documented boundary for users expecting program construction.
A builder API would be a separate design task.

The query language needs no larger expression system for this release. Keep
`count`'s terminal role, immutable bindings, and duplicate-preserving navigation
explicit. Operation printing includes enclosing shells and descendants even
when those operations did not participate in a preceding count. `slice`,
`reachable`, and `closure` serve different purposes; the examples are valuable
because treating them as interchangeable produces misleading analyses.

Input selection and stage evaluation are eager. A narrow `filter` or `head`
does not reduce parsing cost. Output is buffered across statements and input
files to avoid partial stdout on failure, so memory grows with emitted data.
General fixed points and per-operation queries that repeatedly restore a large
binding can still be expensive. Custom call and branch printers also retain
linear canonical-name lookups per referenced value; call-heavy output deserves
separate profiling if it becomes a primary workload. None of the documented
work limits is a total byte-memory or wall-clock bound.

Custom formats have useful early checks for missing operand types, overlapping
assignments, conflicting registrations, and invalid alternatives. Broad clause
shapes still require representative input checks: similar-looking syntax does
not establish the same operand, result, or region meaning. Preset names include
core-only placeholders, and compact forms can omit implicit operation bodies.
The compatibility and preset documentation should retain those qualifications.

The skill validator passed for all four folders, and their examples were reviewed
against the APIs and CLI tests. This is structural and example validation, not a
measurement of agent success rates on independent tasks.

**Validation and release handoff**

Local checks passed on macOS arm64:

- 467 Rust workspace tests on stable and Rust 1.88, with two opt-in tests ignored;
  the Rust doctest also passes.
- Clippy on both toolchains and Rustdoc with warnings denied; Rust formatting.
- 132 Python tests on the rebuilt CPython 3.14 extension, with two skips; Ruff
  formatting/lint and ty. Declared class members were checked against the
  rebuilt extension, and the README Python example ran successfully.
- The existing query harness completed 229 subprocess probes, including 200
  explicit checks, with no timeouts. Its independent generated graph checks
  cover selection, dependency traversal, and aggregation.
- The opt-in release query-profile smoke run passes.
- Five-second fuzz budgets for lexer, parser, and semantic lowering completed
  without failures: 173,997, 68,371, and 11,032 executions respectively. These
  are smoke checks, not an exhaustive fuzzing campaign.
- Rust crate packaging and verification; a version-specific CPython 3.14 macOS
  arm64 wheel; and a source distribution built and installed in a clean
  environment. Twine accepts the package metadata.

Linux and the full CPython 3.11–3.14 wheel matrix remain the CI release gate.
After choosing 0.1.0, update the versions, lockfiles, version references, and
changelog using the [release guide](../releasing.md), then run CI on that exact
commit. No version tag, push, or publication was performed by this review.
