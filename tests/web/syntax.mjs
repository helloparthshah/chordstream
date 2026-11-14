// Fails if any inline <script> in the web pages, or any engine module, does not parse (run in CI before deploying).
import { readFileSync } from 'node:fs';
const AsyncFunction = (async () => {}).constructor;
const web = new URL('../../src/chordstream/web/', import.meta.url);
let n = 0;
for (const page of ['index.html', 'annotate.html', 'bench.html']) {
  const html = readFileSync(new URL(page, web), 'utf8');
  for (const m of html.matchAll(/<script(?![^>]*\bsrc=)([^>]*)>([\s\S]*?)<\/script>/g)) {
    const code = m[2];
    // Modules: drop the import lines and parse the rest as an async function body (top-level await).
    if (/type="module"/.test(m[1])) new AsyncFunction(code.replace(/^\s*import .*$/gm, ''));
    else new Function(code);
    n++;
  }
}
for (const f of ['sheet.js', 'api-local.js']) { new Function(readFileSync(new URL(f, web), 'utf8')); n++; }
await import(new URL('engine.js', web));
console.log(`syntax ok (${n} scripts + engine.js)`);
