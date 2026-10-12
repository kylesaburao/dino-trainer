'use strict';
// Builds the Claude artifact copy of the game page (index.html, the JS page) into
// dist/artifact/ and prints the `files` map for the Artifact publish call.
//
//   node tools/build-artifact.js
//
// The artifact host wraps the published page in its own <!doctype>/<head>/<body>, so
// this keeps index.html's <title>, stylesheet links and <body> content and drops the
// document tags. The page's scripts and style.css are published unchanged next to it
// (the map below), so the artifact runs exactly the shipped game. One artifact-only
// style adds the 16px side gutter the artifact contract asks for at phone width.
// AGENTS.md ("Published artifact") says when to rebuild and republish.
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const OUT = path.join(ROOT, 'dist', 'artifact');
const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');

const title = (html.match(/<title>[\s\S]*?<\/title>/) || [])[0];
const body = (html.match(/<body[^>]*>([\s\S]*)<\/body>/) || [])[1];
if (!title || !body) throw new Error('index.html: no <title> or <body>');
const styles = [...html.matchAll(/<link\s+rel="stylesheet"\s+href="([^"]+)"\s*>/g)].map(m => m[1]);
const scripts = [...body.matchAll(/<script\s+src="([^"]+)"><\/script>/g)].map(m => m[1]);
const local = [...styles, ...scripts];
for (const f of local) {
  if (/^[a-z]+:|^\/\//i.test(f) || f.includes('..')) throw new Error(`index.html references a non-local file: ${f}`);
  if (!fs.existsSync(path.join(ROOT, f))) throw new Error(`index.html references a missing file: ${f}`);
}

const page = [
  title,
  ...styles.map(href => `<link rel="stylesheet" href="${href}">`),
  '<style>/* artifact only: side gutter at phone width */ body { padding-inline: 16px; }</style>',
  body.trim(),
  '',
].join('\n');

fs.mkdirSync(OUT, { recursive: true });
fs.writeFileSync(path.join(OUT, 'index.html'), page);
const files = Object.fromEntries(local.map(f => [f, f]));
console.log(JSON.stringify({ file_path: path.relative(ROOT, path.join(OUT, 'index.html')), files }, null, 2));
