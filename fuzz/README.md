# Lexer fuzzing

The fuzz package is intentionally excluded from the workspace default gate.
Run a bounded arbitrary-byte reconstruction smoke test with:

```sh
(cd fuzz && RUSTC_BOOTSTRAP=1 cargo fuzz run lexer -- -max_total_time=5)
```

# Parser fuzzing

The parser target checks arbitrary bytes for lossless reconstruction, bounded
completion, and a structurally valid CST. Run the bounded smoke check with:

```sh
(cd fuzz && RUSTC_BOOTSTRAP=1 cargo fuzz run parser -- -max_total_time=5)
```
