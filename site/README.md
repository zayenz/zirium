# Zirium website

The homepage and documentation are served at <https://zayenz.se/zirium/>.
Astro and Starlight build the site; `docs/` remains the source of the guides.
`src/landing.mdx` owns homepage prose and `scripts/prepare-content.mjs` lists
published guides explicitly. Relative guide links become website links;
repository-file links point to GitHub. The Rust API stays on docs.rs.

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

Use `zirql` for query code fences. The grammar in
`../assets/syntax/zirql.tmLanguage.json` is an unchanged copy of the
[shared Zirql grammar](https://github.com/zayenz/zirql/blob/main/assets/syntax/zirql.tmLanguage.json).
Copy updates from that file to keep the three documentation sites consistent.
The site configuration adds host-specific keywords using the same token scopes.
Highlighting does not validate queries.

Existing `zirium` code fences also use the Zirql grammar. MLIR input keeps its
separate fence label.
