# Releasing Zirium

Zirium publishes one Rust crate and one Python package. The internal
`zirium-python` crate is only a build component and must remain marked with
`publish = false`.

The first public release requires registry account setup. Do not create the
release tag until every item in the preparation section is complete.

## Prepare the registries and repository

1. Make the GitHub repository public.
2. Confirm that the `zirium` names are still available on crates.io and PyPI.
3. Sign in to crates.io, verify the account email address, and create an API
   token for the first publication.
4. Add a pending trusted publisher for `zirium` on PyPI with these values:
   - owner: `zayenz`
   - repository: `zirium`
   - workflow: `release.yml`
   - environment: `pypi`
5. Create the `pypi` environment in the GitHub repository and require manual
   approval before deployment.

The crates.io token is needed only for the first release. crates.io trusted
publishing can be configured after the crate exists.

## Check the release commit

Run the local quality checks described in the
[compatibility guide](compatibility.md), then check the package that Cargo
would upload:

```sh
cargo publish -p zirium --dry-run --locked
cargo package -p zirium --list
```

Review the generated crate under `target/package/`. Also check that the version
is identical in these files:

- `crates/zirium/Cargo.toml`
- `crates/zirium-python/Cargo.toml`
- `pyproject.toml`
- `CHANGELOG.md`

Commit the final release changes and wait for the Quality workflow to pass on
`main`.

## Publish 0.0.1

Publish the Rust crate first:

```sh
cargo publish -p zirium --locked
```

Confirm that version 0.0.1 appears on crates.io and that its documentation
build has started on docs.rs. Then tag the same commit and push only that tag:

```sh
git tag -a v0.0.1 -m "Zirium 0.0.1"
git push origin v0.0.1
```

Do not use `git push --tags`. This repository may contain local tags that are
not part of the public release history.

The Release workflow checks the tag against all package versions, builds and
installs the source distribution, and builds separate manylinux wheels for
CPython 3.11 through 3.14. The PyPI upload waits for approval in the `pypi`
environment and uses trusted publishing, so it needs no stored PyPI token.

After approving the deployment, query the Rust package and install the Python
package from their registries:

```sh
cargo info zirium@0.0.1
uv run --no-project --isolated --with zirium==0.0.1 python -c \
  'import zirium; assert zirium.parse_text("\"test\"() : () -> ()")'
```

Check the crates.io, docs.rs, and PyPI pages before announcing the release.
