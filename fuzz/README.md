# Fuzzing

The fuzz package runs separately from the workspace checks. Each command below
runs a target for five seconds.

## Lexer

The lexer target checks that arbitrary bytes can be reconstructed exactly:

```sh
(cd fuzz && RUSTC_BOOTSTRAP=1 cargo fuzz run lexer -- -max_total_time=5)
```

## Parser

The parser target checks lossless reconstruction, bounded completion, and CST
structure for arbitrary input:

```sh
(cd fuzz && RUSTC_BOOTSTRAP=1 cargo fuzz run parser -- -max_total_time=5)
```

## Semantic lowering

The semantic target parses bounded input with the baseline registry, runs strict
and best-effort lowering, and checks each returned document:

```sh
(cd fuzz && RUSTC_BOOTSTRAP=1 cargo fuzz run semantic_lowering -- -max_len=4096 -max_total_time=5)
```
