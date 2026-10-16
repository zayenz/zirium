# Releasing Zirium

Zirium publishes one Rust crate, one Python package and standalone CLI
archives. The internal `zirium-python` crate is a build component and remains
`publish = false`. The selected release is 0.5.0, with Zirql 0.1.0 as its
shared-language dependency.

## Prepare a candidate

Keep the versions in `crates/zirium/Cargo.toml`,
`crates/zirium-python/Cargo.toml`, `pyproject.toml`, `Cargo.lock` and `uv.lock`
at 0.5.0. Update the changelog, installation commands and versioned API/source
links together. Refresh the lockfiles without upgrading unrelated dependencies:

```sh
cargo check --workspace
uv lock
```

Run the [compatibility checks](compatibility.md), inspect the Rust package
selection with `cargo package -p zirium --list --allow-dirty`, and build a local
wheel and source distribution. Check both licenses, Python stubs, `py.typed`,
registry data and source manifests. Install each Python archive in a separate
fresh environment outside the checkout, check its version and import paths,
and run a strict semantic query. Extract each CLI archive outside the checkout
and check its checksum, version and strict StableHLO query.

Release's manual candidate mode accepts a full 40-character commit ID:

```sh
candidate=$(git rev-parse HEAD)
gh workflow run release.yml --ref main -f mode=candidate -f source_ref="$candidate"
```

The workflow must already be on `main`. Candidate mode checks out that exact
commit, checks package versions, runs Quality, and uploads the existing Python
and two CLI target artifacts. It creates no tag or GitHub Release and disables
all three publishing jobs. It skips Quality's registry-package check because
that check requires the dependency to be available on crates.io. A successful
candidate run therefore does not establish registry resolution.

## Zirql dependency and source distributions

Before Zirql 0.1.0 is published, the candidate manifest uses both the reviewed
Git revision and a matching registry version:

```toml
zirql = { git = "https://github.com/zayenz/zirql", rev = "6b6184fbf9998ece38c7e8cd5391770b5db32e04", version = "=0.1.0" }
```

Cargo source builds use that public Git commit. Cargo normalizes the Rust crate
archive to the registry requirement `=0.1.0`; it cannot verify that archive
against crates.io until Zirql 0.1.0 is published. A local staged package check
is separate evidence and does not replace this registry check.

The candidate sdist retains the Git-plus-version declaration in
`crates/zirium/Cargo.toml` and the Git revision in its root `Cargo.lock`.
It contains both Rust crates, so the extension's `../zirium` path resolves
inside the archive. Installing it needs Rust and access to the declared Git
and registry dependencies, but no sibling checkout. The wheel contains the
compiled extension and needs no Rust toolchain.

For the final registry-backed release, first publish and verify Zirql 0.1.0.
Then replace the Zirql dependency above with:

```toml
zirql = "=0.1.0"
```

Refresh `Cargo.lock` and the locked Python environment, rebuild every final
artifact, and repeat the clean wheel/sdist installations. Inspect the sdist's
stored manifests and lockfile again: Zirql must resolve from crates.io. Commit
that final source and run the candidate workflow against its exact commit.
Do not publish artifacts built before this finalization as the registry-backed
source distribution.

Run the real registry package checks with no path or registry patches:

```sh
cargo package -p zirium --locked
cargo publish -p zirium --dry-run --locked
```

Retain the final source commit, filenames, SHA-256 hashes and check results.
A changed source commit requires rebuilt artifacts and fresh checks.

## Trusted publishing setup

Both registries trust GitHub owner `zayenz`, repository `zirium`, workflow
`release.yml`, and their respective environment:

| Registry | GitHub environment |
| --- | --- |
| crates.io | `crates-io` |
| PyPI | `pypi` |

Both environments require manual approval by `zayenz`. Publishing jobs receive
short-lived credentials; no registry token needs to be stored in GitHub.
CLI publication uses the workflow's GitHub token and runs independently of
these registry approvals.

## Publish 0.5.0

Publication is a separate maintainer action. Once the final commit is on
`main`, its artifacts have passed review and Zirql is available on crates.io,
tag that checked commit and push only that tag:

```sh
git tag -a v0.5.0 -m "Zirium 0.5.0"
git push origin v0.5.0
```

Do not use `git push --tags`; local tags may not belong to the public release
history. Release checks that the tag names a commit on `main`, matches all
package versions and has a changelog entry. It runs Quality with actual
registry package verification enabled.

Quality builds conventional CPython 3.11–3.14 wheels for Linux x86_64
(manylinux) and macOS arm64, plus the source distribution. PyPI receives these
checked artifacts directly. There are no stable-ABI wheels. Open **Review
deployments**, select the registry environments and approve their uploads.
The internal extension crate is never published to crates.io.

After publication, check fresh registry installations:

```sh
cargo info zirium@0.5.0
cargo install zirium --version '=0.5.0' --locked
uv run --no-project --isolated --with zirium==0.5.0 python -I -c \
  'import zirium; from zirium.query import ops; assert zirium.__version__ == "0.5.0"; d=zirium.parse_text("\"test\"() : () -> ()").lower_strict().document; assert d is not None; assert d.query(ops().names().tally()) == {"test": 1}'
```

Check the crates.io, [versioned Rust API](https://docs.rs/zirium/0.5.0/zirium/),
PyPI and GitHub release pages before announcing the release. Site publication
follows the separate procedure in [`site/PUBLISHING.md`](../site/PUBLISHING.md).

## CLI archives

Release builds the CLI with Rust 1.88.0 and the committed lockfile for
`x86_64-unknown-linux-musl` and `aarch64-apple-darwin`. Linux uses Ubuntu 22.04
with musl tools and rejects dynamically linked executables. macOS uses
`macos-15` with `MACOSX_DEPLOYMENT_TARGET=11.0`. The platform baselines are Linux
kernel 3.2 and macOS 11; CI runs on the listed runners, not the minimum OS
versions.

Each archive contains `zirium`, `LICENSE-MIT` and `LICENSE-APACHE` under
`zirium-0.5.0-TARGET/`, with a `.tar.gz.sha256` sidecar. The build job extracts
the archive outside the checkout, checks its version and runs a strict StableHLO
query against the embedded registry. Candidate runs upload these archives as
workflow artifacts only.

For publication, the publishing job verifies the downloaded checksums, uploads
the archives to a draft GitHub Release, then publishes it. Only this job has
`contents: write`. Check both platform downloads before announcing the release.

## Retry after a workflow failure

The three channels publish independently. Retry only the failed channel after
checking whether its upload succeeded. Do not move or replace a published tag.
If the workflow needs a fix, commit it to `main` and run it against the existing
tag:

```sh
gh workflow run release.yml --ref main -f mode=publish -f tag=v0.5.0 -f registry=pypi
```

Use `registry=crates-io`, `registry=pypi` or `registry=binaries` for one channel.
`registry=both` selects crates.io and PyPI; `registry=all` selects all three and
is the manual default. Tag pushes also select all three. Binary retries may
replace assets in a draft release but refuse to replace a published release's
assets. Registry publishing retains the environment approvals.
