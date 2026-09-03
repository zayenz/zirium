# Releasing Zirium

Zirium publishes one Rust crate and one Python package. The internal
`zirium-python` crate is only a build component and must remain marked with
`publish = false`.

Pushing a version tag starts the Release workflow. After its package checks
pass, approve the `crates-io` and `pypi` deployments to publish both packages.

## Prepare the registries and repository

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

Keep the three package versions identical and refresh `Cargo.lock` after
changing them. Run the local quality checks described in the
[compatibility guide](compatibility.md), then check the package that Cargo
would upload:

```sh
cargo publish -p zirium --dry-run --locked
cargo package -p zirium --list
```

Review the generated crate under `target/package/`. Commit the final release
changes and wait for the Quality workflow to pass on `main`.

## Publish a version

Tag the checked commit and push only that tag. For example, for 0.0.2:

```sh
git tag -a v0.0.2 -m "Zirium 0.0.2"
git push origin v0.0.2
```

Do not use `git push --tags`. This repository may contain local tags that are
not part of the public release history.

The Release workflow checks that the tagged commit is on `main` and that the
tag matches all package versions. It verifies the Rust package, builds and
installs the source distribution, and builds separate manylinux wheels for
CPython 3.11 through 3.14.

When these checks pass, open the workflow run, select **Review deployments**,
select both environments, and approve. Each publishing job uploads its package
independently. The internal `zirium-python` crate is not published to crates.io.

After approving the deployment, query the Rust package and install the Python
package from their registries:

```sh
cargo info zirium@0.0.2
uv run --no-project --isolated --with zirium==0.0.2 python -c \
  'import zirium; assert zirium.parse_text("\"test\"() : () -> ()")'
```

Check the crates.io, docs.rs, and PyPI pages before announcing the release.

## Retry after a workflow failure

The two uploads are not atomic. If one succeeds and the other fails, rerun only
the failed publishing job. Do not rerun a successful upload.

Keep published release tags unchanged. If the workflow itself needs a fix,
commit it to `main`, then run the updated workflow against the existing tag.
Select only the registry whose upload has not succeeded:

```sh
gh workflow run release.yml --ref main -f tag=v0.0.2 -f registry=pypi
```

Use `registry=crates-io` for a Rust-only retry or `registry=both` if neither
upload succeeded. Check the registry first if an upload's outcome is uncertain.
The workflow builds the tagged source, not the current `main` checkout, and
still requires approval before publishing.
