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
