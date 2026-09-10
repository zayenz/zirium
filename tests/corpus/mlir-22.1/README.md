# MLIR 22.1 corpus

[`manifest.toml`](manifest.toml) records the grammar and fixture sources for
compatibility with `llvmorg-22.1.0`. Add one `[[families]]` entry for each lexical
or grammar family. Each entry identifies the upstream implementation and rule,
positive fixtures and their sources, intentional Zirium differences, and
applicable license notices. A sampled fixture alone does not define the grammar.

`generic-baseline/` contains project-authored generic MLIR and focused malformed
variants for testing recovery. These fixtures contain no copied LLVM source.

`lexer.mlir` is a compact, project-authored fixture covering lexical forms. The
manifest links each family to the tagged upstream implementation. Zirium
preserves whitespace and comments, accepts arbitrary bytes, recovers with
invalid tokens, and applies explicit file and token limits.
