# Zirium website

The homepage and documentation are served at <https://zayenz.se/zirium/>.
Astro and Starlight build the site; `docs/` remains the source of the guides.
`src/landing.mdx` owns homepage prose and `scripts/prepare-content.mjs` lists
published guides explicitly. Relative guide links become website links;
repository-file links point to GitHub. The Rust API stays on docs.rs.

The query guides explain Zirium's MLIR operations, output and edit rules. Links
to the Zirql language reference explain the shared expression rules. Keep
Zirium commands and host restrictions here so readers can use the tool without
reading another site's tutorial first.

Use Node 22.12 or newer:

```sh
cd site
npm ci
npm run dev
npm run check
npm run build
```

The default site origin is `https://zayenz.se` and the base path is `/zirium`.
`SITE_URL` and `SITE_BASE` override these values for a different deployment.
Generated content, `public/`, `.astro/`, dependencies and `dist/` are ignored.

See [PUBLISHING.md](PUBLISHING.md) for the deployment handoff. Building the site
does not run Rust tests, benchmarks or release checks.

## Zirql syntax highlighting

Use `zirql` for query code fences. Zirql owns the common grammar;
`../assets/syntax/zirql.tmLanguage.json` is a checked-in copy from the full
commit in `../assets/syntax/zirql-revision.txt`. Do not edit that copy directly.
From a Zirql checkout, run its revision-pinned synchronization commands:

```sh
node scripts/sync-syntax.mjs sync <full-Zirql-commit> ../zincite ../zirium
node scripts/sync-syntax.mjs check <full-Zirql-commit> ../zincite ../zirium
```

Review and commit the grammar and revision marker here, then run this site's
`check` and `build` commands. Builds use the local asset and need no Zirql
checkout. See [Zirql's contributor procedure](https://github.com/zayenz/zirql/blob/main/site/README.md#syntax-highlighting)
for the common update path.

The site configuration adds Zirium's host keywords and retains the legacy
surface's hash comments and decimal numbers, omitting single-quoted names and
block comments. Common forms come from Zirql. Highlighting describes spelling;
parsing and host policy decide which constructs a query may execute.

Existing `zirium` code fences also use the Zirql grammar. MLIR input keeps its
separate fence label.
