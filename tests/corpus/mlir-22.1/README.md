# MLIR 22.1 corpus manifest

`manifest.toml` fixes compatibility evidence to `llvmorg-22.1.0`. Each owned lexical or grammar family must name the upstream implementation file and parsing or lexing rule used as authority, every positive fixture and its source, intentional Zirium differences, and the applicable license notice. Add one `[[families]]` entry per family; do not treat a sampled fixture or the MLIR documentation as a complete grammar.

`generic-proving/` contains the project-authored positive fixture from the base brief and focused malformed recovery variants. They contain no copied LLVM source text and therefore need no LLVM license notice beyond that fact.

`lexer.mlir` is a compact project-authored checklist fixture. The manifest ties
each family it exercises to the tagged implementation source. Zirium retains
whitespace and comments, accepts arbitrary bytes, recovers with invalid tokens,
and applies explicit file and token limits.
