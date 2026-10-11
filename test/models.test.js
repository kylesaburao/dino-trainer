'use strict';
// The committed outputs of `node trainer/cli.js export-models` (models.js,
// models-css/<id>.css, models-css/index.js) match the runs they come from, and
// the shipped entry is the root model.js. Fails when they go stale: regenerate
// with `node trainer/cli.js export-models`.
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const { BUNDLED_MODELS } = require('../trainer/cli.js');
const { modelToCss, modelId } = require('../trainer/css-model.js');
const models = require('../models.js');
const index = require('../models-css/index.js');
const shipped = require('../model.js');

test('models.js lists every BUNDLED_MODELS run, in order, with its best.json weights', () => {
  assert.deepStrictEqual(models.map(m => m.source), BUNDLED_MODELS.map(b => `${b.dir}/best.json`));
  for (const [i, b] of BUNDLED_MODELS.entries()) {
    const best = JSON.parse(fs.readFileSync(path.join(ROOT, b.dir, 'best.json'), 'utf8'));
    const m = models[i];
    assert.strictEqual(m.id, path.basename(b.dir));
    assert.strictEqual(m.label, b.label);
    assert.deepStrictEqual(m.layers, best.layers);
    assert.deepStrictEqual(m.params, Array.from(best.params));
    assert.strictEqual(m.gameVersion, best.gameVersion);
    assert.ok(m.id.startsWith(`${m.gameVersion}-`), `${m.id} is named for its rules`);
  }
});

test('each models-css/<id>.css is the CSS network of its models.js entry', () => {
  for (const m of models) {
    assert.strictEqual(m.css, `models-css/${m.id}.css`);
    assert.strictEqual(fs.readFileSync(path.join(ROOT, m.css), 'utf8'), modelToCss(m), m.id);
    assert.strictEqual(m.cssId, modelId(m));
  }
});

test('models-css/index.js is models.js without weights', () => {
  assert.strictEqual(index.length, models.length);
  for (const [i, e] of index.entries()) {
    assert.ok(!('params' in e), 'no weights in the CSS page index');
    for (const k of Object.keys(e)) assert.deepStrictEqual(e[k], models[i][k], `${e.id}.${k}`);
  }
});

test('exactly one entry is shipped, and it is the root model.js', () => {
  const s = models.filter(m => m.shipped);
  assert.strictEqual(s.length, 1);
  assert.strictEqual(s[0].source, shipped.source);
  assert.deepStrictEqual(s[0].params, shipped.params);
  assert.deepStrictEqual(s[0].layers, shipped.layers);
});
