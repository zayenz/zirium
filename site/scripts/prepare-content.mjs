import { mkdir, readFile, writeFile, copyFile, rm } from 'node:fs/promises';
import { resolve, relative, dirname, posix } from 'node:path';
import { fileURLToPath } from 'node:url';

const site = fileURLToPath(new URL('../', import.meta.url));
const root = resolve(site, '..');
const guides = [
  'getting-started', 'capabilities', 'compatibility', 'cli-examples',
  'query-language', 'query-dsl', 'structural-editing', 'semantic-diff',
  'custom-formats', 'registry-presets', 'releasing',
  'architecture/syntax-representation', 'architecture/query-profiling',
  'architecture/processing-benchmarks', 'architecture/stress-instance-benchmark',
];
const output = resolve(site, 'src/content/docs');
await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });

for (const name of guides) {
  const source = resolve(root, `docs/${name}.md`);
  const markdown = await readFile(source, 'utf8');
  const title = markdown.match(/^# (.+)\r?\n/);
  if (!title) throw new Error(`Missing page title in ${source}`);
  const body = markdown.slice(title[0].length).replace(/\]\(([^\s)]+)\)/g, (link, target) => {
    if (/^(?:[a-z]+:|#|\/)/i.test(target)) return link;
    const [path, anchor = ''] = target.split('#');
    const resolved = relative(root, resolve(dirname(source), path));
    const slug = resolved.match(/^docs\/(.+)\.md$/)?.[1];
    const suffix = anchor ? `#${anchor}` : '';
    if (slug && guides.includes(slug)) {
      return `](${posix.relative(`guides/${name}`, `guides/${slug}`) || '.'}/${suffix})`;
    }
    return `](https://github.com/zayenz/zirium/blob/main/${resolved}${suffix})`;
  });
  const target = resolve(output, `guides/${name}.md`);
  await mkdir(dirname(target), { recursive: true });
  await writeFile(target, `---\ntitle: ${JSON.stringify(title[1])}\neditUrl: false\n---\n${body}`);
}

await mkdir(resolve(site, 'public'), { recursive: true });
await copyFile(resolve(root, 'assets/zirium-element-dark.svg'), resolve(site, 'public/favicon.svg'));
await copyFile(resolve(site, 'src/landing.mdx'), resolve(output, 'index.mdx'));
