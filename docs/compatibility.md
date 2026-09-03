# Compatibility and local wheel checks

Zirium 0.0.2 requires Rust 1.85 or newer and supports version-specific CPython
extensions for conventional CPython 3.11 through 3.14. The release artifact
promise covers Linux x86_64 and macOS arm64, because those platforms have
passing quality and release workflow artifact lanes. Other platform source
builds may work, but are not supported release platforms. The CI quality
workflow is the release-artifact evidence for this promise. It does not use or
claim the stable ABI (abi3 or abi3t). CPython 3.14
free-threaded builds are experimental and are not part of the stable promise.

Run the complete local Rust quality gate. The commands match the CI jobs:

```sh
cargo fmt --check
cargo +1.85 test --workspace --all-targets --all-features
cargo +1.85 clippy --workspace --all-targets --all-features -- -D warnings
cargo +stable test --workspace --all-targets --all-features
cargo +stable clippy --workspace --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo +stable doc --workspace --all-features --no-deps
```

## CPython compatibility

With `uv` installed, run the formatter, linter, type checker, and package tests
against each conventional interpreter. `uv` downloads a missing interpreter
when necessary.

```sh
for version in 3.11 3.12 3.13 3.14; do
  env_dir="target/python-$version"
  uv venv --python "$version" --clear "$env_dir"
  uv pip install --python "$env_dir/bin/python" maturin pytest ruff ty
  VIRTUAL_ENV="$env_dir" "$env_dir/bin/maturin" develop --uv
  "$env_dir/bin/ruff" format --check python
  "$env_dir/bin/ruff" check python
  "$env_dir/bin/ty" check python
  "$env_dir/bin/pytest" -q
done
```

For the experimental CPython 3.14 free-threaded check, install a free-threaded
interpreter and select it explicitly:

```sh
uv python install 3.14t
uv venv --python 3.14t --clear target/python-3.14t
uv pip install --python target/python-3.14t/bin/python maturin pytest ruff ty
VIRTUAL_ENV=target/python-3.14t target/python-3.14t/bin/maturin develop --uv
target/python-3.14t/bin/ruff format --check python
target/python-3.14t/bin/ruff check python
target/python-3.14t/bin/ty check python
target/python-3.14t/bin/pytest
```

The parsing, file I/O, lowering, edit commit, verification, printing, and
structural-comparison entry points release Python while doing Rust-only work.
The Python tests include concurrent semantic reads and editing checks; run the
same test suite in both conventional and free-threaded lanes.

## Build and import a local wheel

Build a wheel for one selected interpreter, then install that exact wheel into
a clean environment. The resulting filename contains a version-specific tag
such as `cp314-cp314`; it must not contain `abi3` or `abi3t`. The CI quality
workflow repeats this check for CPython 3.11, 3.12, 3.13, and 3.14 on Linux
x86_64 and macOS arm64.

Wheel builds call maturin directly because its PEP 517 path does not run the
Linux wheel audit. Linux CI builds inside a manylinux2014 container and rejects
a wheel without a manylinux tag. macOS CI requires a native arm64 runner and
rejects a wheel without a macOS arm64 tag. `uv build` remains suitable for the
sdist.

```sh
rm -rf target/wheels
uv run --isolated --python 3.14 --with maturin==1.15.0 \
  maturin build --release --locked --compatibility pypi \
  --interpreter python --out target/wheels
wheel=$(find target/wheels -name 'zirium-*.whl' -print -quit)
case "$wheel" in
  *abi3*) echo "unexpected stable-ABI wheel: $wheel" >&2; exit 1 ;;
esac
uv venv --python 3.14 --clear target/wheel-import-venv
uv pip install --python target/wheel-import-venv/bin/python "$wheel"
target/wheel-import-venv/bin/python -c \
  'import zirium; d=zirium.parse_text("\"test\"() : () -> ()").lower_strict().document; assert d.operation_table().count == 1'
```

Change `3.14` to any supported conventional interpreter version to build and
check that version's wheel. Build an sdist and validate it with:

```sh
uv build --sdist --python 3.14 --out-dir target/sdist --clear
uvx twine check target/sdist/*
```
## Resource limits

Rust callers can use `ParsedFile::parse_with_limits` with `ParseLimits`. Python's `parse_bytes`, `parse_text`, and `parse_file` accept the same limits as keyword-only arguments: `max_file_bytes`, `max_tokens`, `max_delimiter_depth`, `max_payload_bytes`, `max_numeric_literal_bytes`, and `max_attribute_depth`. Existing calls keep their current defaults.

Exceeding `max_file_bytes` rejects the input before lexing (`ParseFileError::ResourceLimit` in Rust and `ResourceLimitError` in Python). Other syntax limits produce a lossless parsed file with diagnostics. Attribute-depth exhaustion during lowering produces an invalid attribute sentinel: strict lowering has no document, while best-effort lowering returns a document only when it remains structurally valid.
