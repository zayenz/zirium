import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import zirql from '../assets/syntax/zirql.tmLanguage.json' with { type: 'json' };

// Zirium's legacy surface uses hash comments, decimal numbers and no quoted
// names or block comments. Keep these lexical distinctions local to the site;
// highlighting does not determine the host's evaluator capabilities.
const queryLanguage = {
  ...zirql,
  patterns: [
    { name: 'comment.line.number-sign.zirql', match: '#.*$' },
    { name: 'constant.numeric.zirql', match: String.raw`\b[0-9]+(?:\.(?!\.)[0-9]*)?(?:[eE][+-]?[0-9]+)?\b` },
    ...zirql.patterns.filter(({ name }) => ![
      'comment.line.number-sign.zirql', 'comment.block.zirql',
      'entity.name.quoted.zirql', 'constant.numeric.zirql',
    ].includes(name)),
    {
      name: 'keyword.control.zirql',
      match: String.raw`(?<![\w.])\b(?:do|emit|union|intersect|except)\b(?!\s*:)`,
    },
  ],
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
      { label: 'Rust API', link: 'https://docs.rs/zirium/0.6.1/zirium/' },
      { label: 'Zirql library', items: [
        { label: 'Overview and alphabetical index', link: 'https://zayenz.se/zirql/guides/library-reference/' },
        { label: 'Scalars and strings', link: 'https://zayenz.se/zirql/guides/library-strings/' },
        { label: 'Presence and lookup', link: 'https://zayenz.se/zirql/guides/library-presence/' },
        { label: 'Operators and access', link: 'https://zayenz.se/zirql/guides/library-operators/' },
        { label: 'Collections and reductions', link: 'https://zayenz.se/zirql/guides/library-collections/' },
        { label: 'Collection pipeline stages', link: 'https://zayenz.se/zirql/guides/library-pipelines/' },
        { label: 'Ordering and extrema', link: 'https://zayenz.se/zirql/guides/library-ordering/' },
        { label: 'Grouping and neighbourhoods', link: 'https://zayenz.se/zirql/guides/library-grouping/' },
        { label: 'Functions and control flow', link: 'https://zayenz.se/zirql/guides/library-control/' },
        { label: 'Shared native composition', link: 'https://zayenz.se/zirql/guides/library-native/' },
        { label: 'JSON and Markdown rendering', link: 'https://zayenz.se/zirql/guides/library-rendering/' },
        { label: 'Language and scope', link: 'https://zayenz.se/zirql/guides/language-reference/' },
      ] },
      { label: 'All projects', link: 'https://zayenz.se/projects/' },
    ],
    expressiveCode: { shiki: { langs: [queryLanguage], langAlias: { zirium: 'zirql', mlir: 'text' } } },
  })],
});
