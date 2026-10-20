# Compatibility and local wheel checks

Zirium 0.6.1 requires Rust 1.88 or newer. Wheels support conventional CPython
3.11 through 3.14 on Linux x86_64 and macOS arm64; CI checks both platforms.
Wheels use version-specific ABIs, without `abi3` or `abi3t`. Other platforms and
free-threaded CPython 3.14 are unsupported, though source builds may work.

## CLI binaries

Starting with 0.1.0, standalone archives target Linux x86_64 with static musl
(kernel 3.2 or newer) and macOS arm64 (macOS 11 or newer). They do not require Python, Rust, or LLVM. Binary release jobs use
Rust 1.88.0, check the extracted CLI on Ubuntu 22.04 and macOS 15, and verify
that a strict StableHLO query works outside the checkout. Minimum OS versions
are build baselines, not additional CI runners. Other CLI platforms currently
require a source build.

See [installation](../README.md#cli-binaries) for downloads and
[releasing](releasing.md#cli-archives) for packaging details.

## Rust checks

Run these commands from the repository root. They match the CI jobs:

```sh
cargo +stable fmt --all -- --check
cargo +1.88.0 test --workspace --all-targets --all-features -- --test-threads=1
cargo +1.88.0 clippy --workspace --all-targets --all-features -- -D warnings
cargo +stable test --workspace --all-targets --all-features -- --test-threads=1
cargo +stable clippy --workspace --all-targets --all-features -- -D warnings
cargo +stable test --doc --workspace --all-features
RUSTDOCFLAGS="-D warnings" cargo +stable doc --workspace --all-features --no-deps
```

Zirium 0.6.1 uses the verified Zirql 0.2.0 registry release. The
[release guide](releasing.md#zirql-dependency-and-source-distributions) describes
the registry dependency and sdist contents. Run
`cargo +stable package -p zirium --locked` to verify the Rust archive against
crates.io. The release workflow enables this check before publication.

## CPython compatibility

With `uv` installed, run the formatter, linter, type checker, and package tests
against each conventional interpreter. `uv` downloads a missing interpreter
when necessary.

```sh
for version in 3.11 3.12 3.13 3.14; do
  env_dir="target/python-$version"
  UV_PROJECT_ENVIRONMENT="$env_dir" uv sync --locked --python "$version"
  UV_PROJECT_ENVIRONMENT="$env_dir" uv run --locked --no-sync maturin develop --locked --manifest-path crates/zirium-python/Cargo.toml
  "$env_dir/bin/ruff" format --check python
  "$env_dir/bin/ruff" check python
  "$env_dir/bin/ty" check --python "$env_dir/bin/python" python
  "$env_dir/bin/pytest" -q
done
```

The parsing, file I/O, lowering, edit commit, verification, printing and
structural-comparison entry points release Python while doing Rust-only work.
The existing tests include concurrent semantic reads and editing checks.
Free-threaded interpreters are outside the supported release matrix.

## Build and import a local wheel

Build a wheel, then install it in a clean environment. Its filename must have
a version-specific tag such as `cp314-cp314`, without `abi3` or `abi3t`. CI
repeats this check across the supported interpreter and platform matrix.

Wheel builds call maturin directly because its PEP 517 path does not run the
Linux wheel audit. Linux CI builds inside a manylinux2014 container and rejects
a wheel without a manylinux tag. macOS CI requires a native arm64 runner and
rejects a wheel without a macOS arm64 tag. `uv build` remains suitable for the
sdist.

```sh
uv sync --locked --python 3.14
uv run --locked --no-sync maturin build --release --locked --compatibility pypi \
  --interpreter python --out target/wheels
wheel=$(find target/wheels -name 'zirium-*.whl' -print -quit)
case "$wheel" in
  *abi3*) echo "unexpected stable-ABI wheel: $wheel" >&2; exit 1 ;;
esac
check_dir=$(mktemp -d)
uv venv --python 3.14 "$check_dir/venv"
uv pip install --python "$check_dir/venv/bin/python" "$wheel"
(cd "$check_dir" && ./venv/bin/python -I -c \
  'import zirium, zirium._zirium; from zirium.query import ops; assert zirium.__version__ == "0.6.1"; print(zirium._zirium.__file__); d=zirium.parse_text("\"test\"() : () -> ()").lower_strict().document; assert d is not None; assert d.query(ops().names().tally()) == {"test": 1}')
```

Change `3.14` to any supported conventional interpreter version to build and
check that version's wheel. Build an sdist and validate it with:

```sh
uv build --sdist --python 3.14 --out-dir target/sdist --clear
uv run --locked --no-sync twine check target/sdist/*
```

Install the sdist into another fresh environment and repeat the isolated query:

```sh
sdist_check=$(mktemp -d)
uv venv --python 3.14 "$sdist_check/venv"
uv pip install --no-cache --python "$sdist_check/venv/bin/python" target/sdist/zirium-0.6.1.tar.gz
(cd "$sdist_check" && ./venv/bin/python -I -c \
  'import zirium; from zirium.query import ops; assert zirium.__version__ == "0.6.1"; d=zirium.parse_text("\"test\"() : () -> ()").lower_strict().document; assert d is not None; assert d.query(ops().names().tally()) == {"test": 1}')
```

The [release guide](releasing.md#zirql-dependency-and-source-distributions) records
the registry dependency and sdist requirements.

## Resource limits

Rust callers can use `ParsedFile::parse_with_limits` with `ParseLimits`.
The CLI accepts `--max-file-bytes N`; it reads at most one byte beyond that
boundary from each path or standard input so growing and streamed inputs cannot
bypass the limit. The limit counts bytes, including the UTF-8 encoding of text.
Python's `parse_bytes`, `parse_text`, and `parse_file` accept the same limits as
keyword-only arguments: `max_file_bytes`, `max_tokens`, `max_delimiter_depth`,
`max_payload_bytes`, `max_numeric_literal_bytes`, `max_attribute_depth`, and
`max_alias_expansion_depth`. Omitted limits use their defaults.

Exceeding `max_file_bytes` rejects the input before lexing
(`ParseFileError::ResourceLimit` in Rust and `ResourceLimitError` in Python).
Other syntax limits produce a lossless parsed file with diagnostics. Exceeding
the attribute-depth limit during lowering produces an invalid attribute
placeholder. Strict lowering returns no document; best-effort lowering returns
one only if its structure remains valid. The alias expansion limit defaults to
64 and covers type, attribute, affine, memref, and location aliases through one
shared budget. Exceeding the limit produces an invalid semantic value and the
diagnostic `alias expansion depth exceeds limit of 64`, with the effective limit
substituted.

Semantic diff covers Zirium's represented operation, type, attribute, region,
CFG, and SSA structure. It rejects incomplete documents, resource-backed
attributes, and unrepresented trailing file metadata. Opaque retained values
are compared byte-for-byte; see [semantic diff](semantic-diff.md).
