'use strict';
// pages.js: the game pages always play the latest rules and offer only models for
// them; a selector pick is remembered without reloading.
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');
const Core = require('../core.js');
const Pages = require('../pages.js');

const LATEST = Core.GAME_VERSIONS[Core.GAME_VERSIONS.length - 1];
const entries = [
  { id: 'old-a', gameVersion: 'v1' },
  { id: 'old-b', gameVersion: 'v2' },
  { id: 'new-a', gameVersion: LATEST },
  { id: 'new-b', gameVersion: LATEST },
];
function storage(value) {
  const d = value === undefined ? {} : { [Pages.MODEL_KEY]: value };
  return { d, getItem: (k) => (k in d ? d[k] : null), removeItem: (k) => { delete d[k]; } };
}

test('pages play the last entry of GAME_VERSIONS', () => {
  assert.strictEqual(Pages.GAME_VERSION, LATEST);
  assert.strictEqual(Pages.MODEL_KEY, 'dino-model');
});

test('forPages keeps only latest-rules entries', () => {
  assert.deepStrictEqual(Pages.forPages(entries).map(m => m.id), ['new-a', 'new-b']);
});

test('chooseModel: ?model= and the remembered choice count only for the latest rules', () => {
  let s = storage('new-b');
  assert.strictEqual(Pages.chooseModel(entries, '?model=new-a', () => s).id, 'new-a');
  assert.deepStrictEqual(s.d, { 'dino-model': 'new-b' });
  assert.strictEqual(Pages.chooseModel(entries, '?model=old-b', () => s).id, 'new-b', 'a non-latest ?model= is ignored');
  assert.deepStrictEqual(s.d, { 'dino-model': 'new-b' });
  for (const bad of ['old-b', 'no-such-model']) {
    s = storage(bad);
    assert.strictEqual(Pages.chooseModel(entries, `?model=${bad}`, () => s), null, `${bad}: fall back to the root model`);
    assert.deepStrictEqual(s.d, {}, `${bad}: the remembered choice is cleared`);
  }
  s = storage();
  assert.strictEqual(Pages.chooseModel(entries, '?game=v1', () => s), null);
  assert.strictEqual(Pages.chooseModel(entries, '', () => { throw new Error('no storage'); }), null);
});

// A fake window for rememberModel: storage, an address and history.replaceState.
function fakeEnv(href, { badStorage = false, badHistory = false } = {}) {
  const env = { d: {}, href, replaced: 0 };
  env.localStorage = {
    setItem: (k, v) => { if (badStorage) throw new Error('no storage'); env.d[k] = v; },
    removeItem: (k) => { if (badStorage) throw new Error('no storage'); delete env.d[k]; },
  };
  env.location = { get href() { return env.href; } };
  env.history = { state: null, replaceState: (s, t, url) => { if (badHistory) throw new Error('no history'); env.href = url; env.replaced++; } };
  return env;
}

test('rememberModel: a pick is stored, a root pick (\'\') forgets it', () => {
  const env = fakeEnv('file:///g/index.html');
  Pages.rememberModel('new-a', env);
  assert.deepStrictEqual(env.d, { 'dino-model': 'new-a' });
  Pages.rememberModel('', env);
  assert.deepStrictEqual(env.d, {});
});

test('rememberModel: the address becomes only ?model=<id> (hash kept), no reload', () => {
  const env = fakeEnv('file:///g/index-css.html?game=v1&x=2#h');
  Pages.rememberModel('new-b', env);
  assert.strictEqual(env.href, 'file:///g/index-css.html?model=new-b#h');
  Pages.rememberModel('', env);
  assert.strictEqual(env.href, 'file:///g/index-css.html#h');
  assert.strictEqual(env.replaced, 2);
});

test('rememberModel: failing storage still updates the address, failing history still stores', () => {
  let env = fakeEnv('file:///g/index.html', { badStorage: true });
  Pages.rememberModel('new-a', env);
  assert.strictEqual(env.href, 'file:///g/index.html?model=new-a');
  env = fakeEnv('file:///g/index.html', { badHistory: true });
  Pages.rememberModel('new-a', env);
  assert.deepStrictEqual(env.d, { 'dino-model': 'new-a' });
  assert.strictEqual(env.href, 'file:///g/index.html');
});

test('a model pick never reloads the pages', () => {
  for (const f of ['pages.js', 'game.js', 'css-nn.js']) {
    const src = fs.readFileSync(path.join(__dirname, '..', f), 'utf8');
    assert.ok(!/location\.(search|href)\s*=[^=]/.test(src), `${f} assigns location.search or location.href`);
    assert.ok(!/location\.(reload|assign|replace)\b/.test(src), `${f} reloads or navigates`);
    assert.ok(!/pickModel/.test(src), `${f} still uses pickModel`);
  }
  assert.strictEqual(Pages.pickModel, undefined);
});

test('rulesWarning only for other rules', () => {
  assert.strictEqual(Pages.rulesWarning(LATEST), '');
  assert.match(Pages.rulesWarning('v1'), new RegExp(`rules v1.*rules ${LATEST}`));
});

test('the harnesses can set other rules in a scratch copy (one GAME_VERSION line)', () => {
  const src = fs.readFileSync(path.join(__dirname, '..', 'pages.js'), 'utf8');
  assert.strictEqual(src.match(/^const GAME_VERSION = .*;$/gm).length, 1);
});

test('the pages read no ?game= option', () => {
  for (const f of ['pages.js', 'game.js', 'css-nn.js']) {
    const src = fs.readFileSync(path.join(__dirname, '..', f), 'utf8');
    assert.ok(!/get\('game'\)/.test(src), `${f} reads ?game=`);
  }
});
