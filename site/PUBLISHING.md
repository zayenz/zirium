# Publish Zirium at zayenz.se/zirium/

Build the accepted Zirium 0.5.1 release source, using Node 22.12 or newer.
The homepage and guides select 0.5.1 installation commands, Rust API pages and
repository source links. Check that the release tag and API destinations exist
after package publication before importing the final site:

```sh
cd site
npm ci
npm run check
SITE_URL=https://zayenz.se SITE_BASE=/zirium npm run build
```

Review `dist/` with `npm run preview`. Check the homepage, a guide, search,
mobile layout, dark/light themes and the link back to all projects. The output
contains the homepage, guides, assets, search index and sitemap.

From the zayenz.se repository, inspect the import and then apply it:

```sh
npm run import:project -- --project zirium --source /absolute/path/to/zirium/site/dist --dry-run --replace
npm run import:project -- --project zirium --source /absolute/path/to/zirium/site/dist --replace
```

Use the actual absolute path to this build. `--replace` explicitly replaces an
existing imported site. Review the changes and follow zayenz.se's Git-based
deployment process. Its hosted build runs the release checks; do not repeat
them locally solely for an imported documentation update. A ready production
deployment is sufficient release assurance. The main site publishes the static
output; it does not build this repository or run its language toolchains during
deployment.

Keep all guide edits in `docs/`, homepage edits in `site/src/landing.mdx`, and
build configuration in `site/`. Do not edit the imported HTML. To add a guide,
include it in `scripts/prepare-content.mjs` and the sidebar in
`astro.config.mjs`. The package lock fixes website dependencies independently
of Rust and Python releases.

Local checks and candidate artifacts do not deploy the site. Retain the source
commit and reviewed build; import and deployment are separate maintainer actions.
