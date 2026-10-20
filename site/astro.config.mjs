import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import zirql from '../assets/syntax/zirql.tmLanguage.json' with { type: 'json' };

// Keep shared Zirql token scopes; add only this host's syntax.
const queryLanguage = {
  ...zirql,
  patterns: [...zirql.patterns, {
    name: 'keyword.control.zirql',
    match: String.raw`(?<![\w.])\b(?:do|emit|union|intersect|except)\b(?!\s*:)`,
  }],
};

const base = process.env.SITE_BASE || '/zirium';
const site = process.env.SITE_URL || 'https://zayenz.se';
if (!/^\/(?:[a-zA-Z0-9_-]+\/)*[a-zA-Z0-9_-]*$/.test(base)) {
  throw new Error('SITE_BASE must be / or a path such as /zirium');
}
const url = new URL(site);
if (!['https:', 'http:'].includes(url.protocol) || url.pathname !== '/' || url.search || url.hash) {
  throw new Error('SITE_URL must be an HTTP(S) origin; set its path with SITE_BASE');
}

export default defineConfig({
  site, base, trailingSlash: 'always',
  integrations: [starlight({
    title: 'Zirium',
    description: 'Read, query and edit textual MLIR in Rust, Python and the command line, without linking LLVM.',
    logo: { light: '../assets/zirium-logo-dark.svg', dark: '../assets/zirium-logo-light.svg', replacesTitle: true, alt: 'Zirium' },
    favicon: '/favicon.svg',
    customCss: ['./src/styles/custom.css'],
    components: { Footer: './src/components/Footer.astro' },
    social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/zayenz/zirium' }],
    sidebar: [
      { label: 'Start here', items: [{ label: 'Overview', slug: '' }, 'guides/getting-started', 'guides/capabilities', 'guides/compatibility'] },
      { label: 'Queries and edits', items: ['guides/cli-examples', 'guides/query-language', 'guides/query-dsl', 'guides/structural-editing', 'guides/semantic-diff'] },
      { label: 'Dialects', items: ['guides/custom-formats', 'guides/registry-presets'] },
      { label: 'Development', collapsed: true, items: ['guides/architecture/syntax-representation', 'guides/architecture/query-profiling', 'guides/architecture/processing-benchmarks', 'guides/architecture/stress-instance-benchmark', 'guides/releasing'] },
      { label: 'Rust API', link: 'https://docs.rs/zirium/0.6.0/zirium/' },
      { label: 'Zirql language', link: 'https://zayenz.se/zirql/guides/language-reference/' },
      { label: 'All projects', link: 'https://zayenz.se/projects/' },
    ],
    expressiveCode: { shiki: { langs: [queryLanguage], langAlias: { zirium: 'zirql', mlir: 'text' } } },
  })],
});
