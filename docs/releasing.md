# Releasing Zirium

Zirium publishes one Rust crate and one Python package. The internal
`zirium-python` crate is only a build component and must remain marked with
`publish = false`.

Pushing a version tag starts Release, which runs Quality on the tagged commit.
After the checks pass, approve the `crates-io` and `pypi` deployments to publish
the packages. PyPI receives the checked artifacts.

## Trusted publishing setup

Both registries use trusted publishing for the `zirium` package. Each trusts
GitHub owner `zayenz`, repository `zirium`, and workflow `release.yml`, with a
separate environment:

| Registry | GitHub environment |
| --- | --- |
| crates.io | `crates-io` |
| PyPI | `pypi` |

Both environments require manual approval by `zayenz`. Publishing jobs receive
short-lived credentials; no registry token needs to be stored in GitHub.

## Check the release commit

Choose the new version and update these files:

- `crates/zirium/Cargo.toml`
- `crates/zirium-python/Cargo.toml`
- `pyproject.toml`
- `CHANGELOG.md`
- Version references in `README.md` and `docs/compatibility.md`

Keep the three package versions identical and refresh both lockfiles after
changing them:

```sh
cargo check --workspace
uv lock
```

Commit `Cargo.lock` and `uv.lock` with the version changes. The editable Python
package's version is recorded in `uv.lock`; leaving it unchanged makes the
release's locked environment check fail.

Run the local quality checks described in the
[compatibility guide](compatibility.md), then inspect the package contents:

```sh
cargo publish -p zirium --dry-run --locked
cargo package -p zirium --list
```

Inspect the crate under `target/package/`, the source distribution, and a local
version-specific wheel. Commit the release changes and wait for Quality to pass
on that commit. The release maintainer then tags it and approves publication.

## Publish a version

Tag the checked commit and push only that tag. The commands below use 0.1.0
as an example; substitute the version being released throughout.

```sh
git tag -a v0.1.0 -m "Zirium 0.1.0"
git push origin v0.1.0
```

Do not use `git push --tags`. This repository may contain local tags that are
not part of the public release history.

Release verifies that the tag names a commit on `main` and matches all package
versions. Quality checks the Rust package, builds and installs a source
distribution, and builds version-specific wheels for conventional CPython
3.11 through 3.14 on Linux x86_64 (manylinux) and macOS arm64. PyPI publishes
those artifacts directly; no stable-ABI wheels are produced.

When these checks pass, open the workflow run, select **Review deployments**,
select both environments, and approve. Each publishing job uploads its package
independently. The internal `zirium-python` crate is not published to crates.io.

After approving the deployment, query the Rust package and install the Python
package from their registries:

```sh
cargo info zirium@0.1.0
uv run --no-project --isolated --with zirium==0.1.0 python -c \
  'import zirium; assert zirium.parse_text("\"test\"() : () -> ()")'
```

Check the crates.io, docs.rs, and PyPI pages before announcing the release.

## Retry after a workflow failure

The two uploads are not atomic. If one succeeds and the other fails, rerun only
the failed publishing job. Do not rerun a successful upload.

Do not move or replace a published release tag. If the workflow itself needs a fix,
commit it to `main`, then run the updated workflow against the existing tag.
Select only the registry whose upload has not succeeded:

```sh
gh workflow run release.yml --ref main -f tag=v0.1.0 -f registry=pypi
```

Use `registry=crates-io` for a Rust-only retry or `registry=both` if neither
upload succeeded. Check the registry first if an upload's outcome is uncertain.
The workflow builds the tagged source, not the current `main` checkout, and
still requires approval before publishing.
