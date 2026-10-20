# Releasing Zirium

Zirium publishes one Rust crate, one Python package and standalone CLI
archives. The internal `zirium-python` crate is a build component and remains
`publish = false`. The selected release is 0.6.0, with Zirql 0.2.0 as its
shared-language dependency.

## Prepare a candidate

Keep the versions in `crates/zirium/Cargo.toml`,
`crates/zirium-python/Cargo.toml`, `pyproject.toml`, `Cargo.lock` and `uv.lock`
at 0.6.0. Update the changelog, installation commands and versioned API/source
links together. Refresh the lockfiles without upgrading unrelated dependencies:

```sh
cargo check --workspace
uv lock
```

Use the existing Quality and Release workflows for validation and packaging.
A successful build and publishing action are sufficient release assurance.
The [compatibility guide](compatibility.md) describes local checks for development
or investigating a concrete failure; repeating fresh installations and archive
audits is not a release gate.

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

The final release resolves the verified Zirql 0.2.0 release from crates.io:

```toml
zirql = "=0.2.0"
```

The sdist contains both Rust crates, so the extension's `../zirium` path
resolves inside the archive. Its stored manifests and root `Cargo.lock` must
resolve Zirql from crates.io, with no Git dependency or sibling checkout.
Installing the sdist needs Rust and access to the registry dependencies. The
wheel contains the compiled extension and needs no Rust toolchain.

Refresh `Cargo.lock` without upgrading unrelated dependencies and commit the
final source. The Release workflow builds the artifacts and checks registry
resolution. For manual publication, successful CI and the normal publish
command are sufficient; use no path or registry patches:

```sh
cargo publish -p zirium --locked
```

Publish Zirql before any Zirium release that requires its new version. Version,
dependency and documentation changes do not reopen accepted code reviews or
require repeating language tests.

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

## Publish 0.6.0

Once publication is authorized, the final commit is on `main` and Zirql is
available on crates.io, tag that commit and push only that tag:

```sh
git tag -a v0.6.0 -m "Zirium 0.6.0"
git push origin v0.6.0
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

After successful publication, report the version and links to the
[Rust API](https://docs.rs/zirium/0.6.0/zirium/),
[crates.io package](https://crates.io/crates/zirium),
[PyPI package](https://pypi.org/project/zirium/) and
[GitHub release](https://github.com/zayenz/zirium/releases/tag/v0.6.0).
Publish the docs through [`site/PUBLISHING.md`](../site/PUBLISHING.md).
A successful deployment completes the docs release; further validation needs
a concrete failure, known defect or explicit request.

## CLI archives

Release builds the CLI with Rust 1.88.0 and the committed lockfile for
`x86_64-unknown-linux-musl` and `aarch64-apple-darwin`. Linux uses Ubuntu 22.04
with musl tools and rejects dynamically linked executables. macOS uses
`macos-15` with `MACOSX_DEPLOYMENT_TARGET=11.0`. The platform baselines are Linux
kernel 3.2 and macOS 11; CI runs on the listed runners, not the minimum OS
versions.

Each archive contains `zirium`, `LICENSE-MIT` and `LICENSE-APACHE` under
`zirium-0.6.0-TARGET/`, with a `.tar.gz.sha256` sidecar. The build job extracts
the archive outside the checkout, checks its version and runs a strict StableHLO
query against the embedded registry. Candidate runs upload these archives as
workflow artifacts only.

For publication, the publishing job verifies the downloaded checksums, uploads
the archives to a draft GitHub Release, then publishes it. Only this job has
`contents: write`. Trust the successful publishing job's artifact checks.

## Retry after a workflow failure

The three channels publish independently. Retry only the failed channel after
checking whether its upload succeeded. Do not move or replace a published tag.
If the workflow needs a fix, commit it to `main` and run it against the existing
tag:

```sh
gh workflow run release.yml --ref main -f mode=publish -f tag=v0.6.0 -f registry=pypi
```

Use `registry=crates-io`, `registry=pypi` or `registry=binaries` for one channel.
`registry=both` selects crates.io and PyPI; `registry=all` selects all three and
is the manual default. Tag pushes also select all three. Binary retries may
replace assets in a draft release but refuse to replace a published release's
assets. Registry publishing retains the environment approvals.
