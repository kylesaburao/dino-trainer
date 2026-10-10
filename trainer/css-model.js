'use strict';
// model.js -> model.css: the same MLP evaluated by the browser's style engine.
//
// Inputs are the 44 observation features, written by the page as --x0..--x43.
// Every hidden unit is a registered <number> custom property computed with
// calc(): tanh(z) = 1 - 2 / (exp(2z) + 1). The outputs --o0..--o2 and the chosen
// action --act (argmax, first index wins ties, like policy.js) are CSS too; the
// page only reads --act back. Used by `node trainer/cli.js export` (next to
// model.js) and `export-css`. The CSS is generated; never edit model.css.
const crypto = require('crypto');

const num = (v) => {
  if (!Number.isFinite(v)) throw new Error(`non-finite parameter ${v}`);
  return String(v);          // shortest round-trip form of the double
};

// Identifies the weights a model.css was generated from (checked by the page).
function modelId(data) {
  const h = crypto.createHash('sha256').update(JSON.stringify({ layers: data.layers, params: Array.from(data.params) })).digest('hex');
  return `${data.generation ?? 'na'}-${h.slice(0, 16)}`;
}

function modelToCss(data) {
  const { layers } = data;
  const params = Array.from(data.params);
  const L = layers.length - 1;
  const name = (l, i) => (l === 0 ? `--x${i}` : l === L ? `--o${i}` : `--a${l}-${i}`);
  const registered = [];
  const rules = [];
  for (let i = 0; i < layers[0]; i++) registered.push(name(0, i));

  let k = 0;
  for (let l = 1; l <= L; l++) {
    const nIn = layers[l - 1], nOut = layers[l];
    const bBase = k + nOut * nIn;
    for (let i = 0; i < nOut; i++) {
      const terms = [];
      for (let j = 0; j < nIn; j++) terms.push(`var(${name(l - 1, j)}) * ${num(params[k + i * nIn + j])}`);
      const z = `${terms.join(' + ')} + ${num(params[bBase + i])}`;
      registered.push(name(l, i));
      rules.push(l === L ? `  ${name(l, i)}: calc(${z});` : `  ${name(l, i)}: calc(1 - 2 / (exp(2 * (${z})) + 1));`);
    }
    k = bBase + nOut;
  }

  // argmax over the outputs, first index winning ties: running best value --mN
  // and index --bN; output n replaces the best only when strictly greater.
  const nOut = layers[L];
  rules.push('  --m0: var(--o0);', '  --b0: 0;');
  registered.push('--m0', '--b0');
  for (let n = 1; n < nOut; n++) {
    rules.push(`  --c${n}: max(0, sign(var(--o${n}) - var(--m${n - 1})));`);
    rules.push(`  --m${n}: max(var(--m${n - 1}), var(--o${n}));`);
    rules.push(`  --b${n}: calc(var(--c${n}) * ${n} + (1 - var(--c${n})) * var(--b${n - 1}));`);
    registered.push(`--c${n}`, `--m${n}`, `--b${n}`);
  }
  rules.push(`  --act: var(--b${nOut - 1});`);
  registered.push('--act');

  // Startup self-test: tanh(1) and the tie-free argmax machinery on constants.
  rules.push('  --nn-probe: calc(1 - 2 / (exp(2) + 1));', '  --nn-probe-sign: max(0, sign(-0.5));');
  registered.push('--nn-probe', '--nn-probe-sign');

  const id = modelId(data);
  return `/* Dino AI network, evaluated by CSS. Generated from model.js by trainer/css-model.js; do not edit.
   Model: layers ${layers.join('-')}, generation ${data.generation ?? 'n/a'}, game ${data.gameVersion || 'v1'}, ${params.length} params. */
${registered.map(p => `@property ${p} { syntax: "<number>"; inherits: false; initial-value: 0; }`).join('\n')}

.css-net {
  --nn-model-id: "${id}";
  --nn-game: "${data.gameVersion || 'v1'}";
  --nn-action-repeat: ${Math.max(1, data.actionRepeat | 0)};
${rules.join('\n')}
}
`;
}

module.exports = { modelToCss, modelId };
