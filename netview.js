// Network view: draws one decision of a dino-mlp-v1 model on a canvas. Nodes are
// filled by activation, edges coloured by weight sign, width = |weight|, opacity =
// |weight x source activation|, and a ring marks the action that was applied.
// Shared by the game page (game.js), the CSS-network page (css-nn.js) and the
// training dashboard's playback, so all three draw the same picture.
//
// The per-layer activations come from createForward(), a plain forward pass over
// model.params in policy.js's layout (per layer: weights out x in, row-major, then
// biases; tanh hidden layers, linear output, argmax). It is only for the picture:
// the player still decides through policy.js (or, on the CSS page, the style
// engine). capture() compares its argmax with the action actually applied and
// counts disagreements, which the page shows as a warning.
//
// Classic <script> (sets the global `DinoNetView`; needs policy.js first) and Node
// require() (no DOM is touched until createView() is called). Never uses Math.random.
(function (root) {
'use strict';

const Policy = (typeof module !== 'undefined' && module.exports) ? require('./policy.js') : root.DinoPolicy;

const ACTION_NAMES = ['none', 'jump', 'down'];
const DINO_NAMES = ['height', 'vy', 'jumping', 'ducking', 'speed'];
const OBST_NAMES = ['present', 'rel x', 'box y', 'box w', 'box h', 'small cactus', 'large cactus', 'pterodactyl',
  'group size', 'ptero low', 'ptero mid', 'ptero high', 'velocity'];
// Input groups as laid out by policy.js; generic labels if its layout ever changes.
const GROUPS = Policy.OBS_DIM === DINO_NAMES.length + Policy.OBS_K * OBST_NAMES.length
  ? [{ name: 'dino', from: 0, names: DINO_NAMES },
    ...Array.from({ length: Policy.OBS_K }, (_, k) => ({ name: `obstacle ${k + 1}`, from: DINO_NAMES.length + k * OBST_NAMES.length, names: OBST_NAMES }))]
  : [{ name: 'inputs', from: 0, names: Array.from({ length: Policy.OBS_DIM }, (_, i) => `input ${i}`) }];

// Light palette (the dashboard passes its theme's colours instead).
const DEFAULT_COLORS = { pos: '#2a78d6', neg: '#e34948', neutral: '#f0efec', ink: '#0b0b0b', muted: '#7a7973', text: '#52514e' };

const esc = (s) => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// Forward pass that keeps every layer's activations. run(x) -> argmax action index;
// acts[0] is the input, acts[last] the output logits.
function createForward(model) {
  const L = model.layers, p = model.params;
  if (p.length !== Policy.numParams(L)) throw new Error(`expected ${Policy.numParams(L)} params, got ${p.length}`);
  const acts = L.map(n => new Float64Array(n));
  function run(input) {
    acts[0].set(input);
    let k = 0;
    for (let l = 1; l < L.length; l++) {
      const nIn = L[l - 1], nOut = L[l], a = acts[l - 1], out = acts[l];
      const bBase = k + nOut * nIn, last = l === L.length - 1;
      for (let i = 0; i < nOut; i++) {
        let s = p[bBase + i];
        const row = k + i * nIn;
        for (let j = 0; j < nIn; j++) s += p[row + j] * a[j];
        out[i] = last ? s : Math.tanh(s);
      }
      k = bBase + nOut;
    }
    const o = acts[L.length - 1];
    let best = 0;
    for (let i = 1; i < o.length; i++) if (o[i] > o[best]) best = i;
    return best;
  }
  return { acts, run };
}

// A view on `canvas` (sized by CSS; the drawing follows its client size and the
// device pixel ratio). Options, all optional:
//   arch      element whose text shows the shape ("44-8-8-3 · 435 params") or why there is none
//   warn      element shown with the mismatch count once capture() disagrees with the applied action
//   warnText  (count) -> warning text
//   tip       absolutely positioned element (inside canvas.parentElement) for node tooltips
//   colors    object or () => object: { pos, neg, neutral, ink, muted, text }
//   font      CSS font family for the labels
// Calls: setModel(model) (null or a reason string clears it), idle(reason) (shape
// only, no decision), capture(x, chosen, check) (x = the observation the decision
// was made on, chosen = applied action index; check = compare with the argmax),
// draw() (repaints if anything changed; call at most once per animation frame).
function createView(canvas, opts = {}) {
  const ctx = canvas.getContext('2d');
  const font = opts.font || 'system-ui, sans-serif';
  let model = null, fwd = null, chosen = -1, argmax = -1, live = false, dirty = true, nodes = [];
  let mismatches = 0, draws = 0, decisions = 0;

  const colors = () => Object.assign({}, DEFAULT_COLORS, typeof opts.colors === 'function' ? opts.colors() : opts.colors);
  const setArch = (text) => { if (opts.arch) opts.arch.textContent = text; };

  function clear(why) {
    model = null; fwd = null; nodes = []; chosen = -1; argmax = -1; live = false;
    setArch(why ? `· ${why}` : '');
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  function setModel(m) {
    if (!m || typeof m === 'string') { clear(m || ''); return; }
    if (model && m.params === model.params) return;
    model = m;
    fwd = createForward(m);
    chosen = -1; argmax = -1; live = false;
    setArch(`${m.layers.join('-')} · ${m.params.length} params`);
    dirty = true;
  }

  // The shape without a decision (for example while AI mode is off).
  function idle() {
    if (!model) return;
    for (const a of fwd.acts) a.fill(0);
    chosen = -1; argmax = -1;
    if (live) { live = false; dirty = true; }
  }

  function capture(x, act, check) {
    if (!model) return -1;
    argmax = fwd.run(x);
    chosen = act;
    live = true;
    dirty = true;
    decisions++;
    if (check && act >= 0 && argmax !== act) {
      if (!mismatches) console.error(`network view mismatch: forward pass chose ${argmax}, the player applied ${act}`);
      mismatches++;
      if (opts.warn) {
        opts.warn.hidden = false;
        opts.warn.textContent = opts.warnText ? opts.warnText(mismatches) : `Forward pass disagreed with the player on ${mismatches} decision(s)`;
      }
    }
    return argmax;
  }

  // Margins fit the input-group and output labels in the current font.
  function layout(W, H) {
    ctx.font = `11px ${font}`;
    let labelL = 0;
    for (const g of GROUPS) labelL = Math.max(labelL, ctx.measureText(g.name).width);
    labelL = Math.ceil(labelL) + 18;
    ctx.font = `600 12px ${font}`;
    const labelR = Math.ceil(Math.max(ctx.measureText('-00.00').width, ...ACTION_NAMES.map(s => ctx.measureText(s).width))) + 22;
    const L = model.layers, padT = 10, padB = 24;
    const colX = L.map((_, l) => labelL + l * (W - labelL - labelR) / (L.length - 1));
    return L.map((n, l) => {
      let ys;
      if (l === 0) {
        const gap = 10, unit = (H - padT - padB - gap * (GROUPS.length - 1)) / n;
        ys = [];
        GROUPS.forEach((g, gi) => g.names.forEach((_, j) => { ys[g.from + j] = padT + gi * gap + (g.from + j + 0.5) * unit; }));
      } else if (l === L.length - 1) {
        const step = Math.min(46, (H - padT - padB) / n);
        ys = Array.from({ length: n }, (_, i) => H / 2 + (i - (n - 1) / 2) * step);
      } else {
        const unit = (H - padT - padB) / n;
        ys = Array.from({ length: n }, (_, i) => padT + (i + 0.5) * unit);
      }
      const r = l === L.length - 1 ? 9 : Math.max(2, Math.min(5, (H - padT - padB) / n * 0.38));
      return ys.map((y, i) => ({ x: colX[l], y, r, l, i }));
    });
  }

  // Diverging fill: neutral at 0, `pos` for positive, `neg` for negative.
  function mix(a, b, t) {
    const pa = parseInt(a.slice(1), 16), pb = parseInt(b.slice(1), 16);
    const c = (sh) => Math.round(((pa >> sh) & 255) * (1 - t) + ((pb >> sh) & 255) * t);
    return `rgb(${c(16)},${c(8)},${c(0)})`;
  }

  function draw() {
    if (!model || !dirty) return false;
    const dpr = (typeof window !== 'undefined' && window.devicePixelRatio) || 1, W = canvas.clientWidth, H = canvas.clientHeight;
    if (!W || !H) return false;   // hidden: keep `dirty` and paint once shown
    dirty = false;
    if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const { pos, neg, neutral, ink, muted, text } = colors();
    const L = model.layers, p = model.params, acts = fwd.acts;
    nodes = layout(W, H);

    // Edges: colour = weight sign, width = |weight|, opacity = |weight x source
    // activation| (with no decision yet: a faint |weight|).
    let k = 0;
    for (let l = 1; l < L.length; l++) {
      const nIn = L[l - 1], nOut = L[l], a = acts[l - 1];
      let maxW = 1e-9, maxC = 1e-9;
      for (let i = 0; i < nOut * nIn; i++) {
        const w = Math.abs(p[k + i]);
        if (w > maxW) maxW = w;
        const c = Math.abs(p[k + i] * a[i % nIn]);
        if (c > maxC) maxC = c;
      }
      for (let i = 0; i < nOut; i++) {
        const to = nodes[l][i];
        for (let j = 0; j < nIn; j++) {
          const w = p[k + i * nIn + j];
          const alpha = live ? Math.abs(w * a[j]) / maxC : 0.25 * Math.abs(w) / maxW;
          if (alpha < 0.04) continue;
          const from = nodes[l - 1][j];
          ctx.globalAlpha = Math.min(0.85, alpha);
          ctx.strokeStyle = w >= 0 ? pos : neg;
          ctx.lineWidth = 0.4 + 1.6 * Math.abs(w) / maxW;
          ctx.beginPath();
          ctx.moveTo(from.x, from.y);
          ctx.lineTo(to.x, to.y);
          ctx.stroke();
        }
      }
      k += nOut * nIn + nOut;
    }
    ctx.globalAlpha = 1;

    // Nodes: fill = activation (inputs/hidden clamped to [-1, 1]; outputs scaled by max |logit|).
    const outA = acts[L.length - 1];
    let outMax = 1e-9;
    for (const v of outA) outMax = Math.max(outMax, Math.abs(v));
    nodes.forEach((col, l) => col.forEach((n) => {
      const v = !live ? 0 : l === L.length - 1 ? outA[n.i] / outMax : Math.max(-1, Math.min(1, acts[l][n.i]));
      ctx.fillStyle = mix(neutral, v >= 0 ? pos : neg, Math.abs(v));
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = muted;
      ctx.globalAlpha = 0.35;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }));

    // Output labels + chosen-action ring.
    ctx.textBaseline = 'middle';
    nodes[L.length - 1].forEach((n) => {
      const on = n.i === chosen;
      if (on) {
        ctx.strokeStyle = ink;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r + 4, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.fillStyle = on ? ink : muted;
      ctx.font = `${on ? '600 ' : ''}12px ${font}`;
      ctx.textAlign = 'left';
      ctx.fillText(ACTION_NAMES[n.i] || `out ${n.i}`, n.x + 16, n.y - 6);
      ctx.font = `11px ${font}`;
      if (live) ctx.fillText(outA[n.i].toFixed(2), n.x + 16, n.y + 8);
    });

    // Input group labels with a bracket.
    ctx.textAlign = 'right';
    ctx.font = `11px ${font}`;
    ctx.strokeStyle = muted;
    ctx.lineWidth = 1;
    for (const g of GROUPS) {
      const top = nodes[0][g.from].y, bot = nodes[0][g.from + g.names.length - 1].y, bx = nodes[0][0].x - 9;
      ctx.beginPath();
      ctx.moveTo(bx + 3, top);
      ctx.lineTo(bx, top);
      ctx.lineTo(bx, bot);
      ctx.lineTo(bx + 3, bot);
      ctx.stroke();
      ctx.fillStyle = text;
      ctx.fillText(g.name, bx - 5, (top + bot) / 2);
    }
    // Hidden layer captions (shorter when the columns are close together).
    ctx.textAlign = 'center';
    ctx.fillStyle = muted;
    const spacing = nodes[1][0].x - nodes[0][0].x;
    for (let l = 1; l < L.length - 1; l++) {
      const label = [`hidden ${L.length > 3 ? l : ''}`.trim() + ` (${L[l]})`, `h${l} (${L[l]})`, String(L[l])]
        .find(s => ctx.measureText(s).width <= spacing - 6);
      if (label) ctx.fillText(label, nodes[l][0].x, H - 8);
    }
    draws++;
    return true;
  }

  function describe(n) {
    const L = model.layers, a = fwd.acts;
    const val = (v) => live ? v.toFixed(3) : '–';
    if (n.l === 0) {
      const g = GROUPS.find(q => n.i >= q.from && n.i < q.from + q.names.length);
      return `<b>${esc(g.name)} · ${esc(g.names[n.i - g.from])}</b><div class="row">input ${n.i} = ${val(a[0][n.i])}</div>`;
    }
    if (n.l === L.length - 1) return `<b>${esc(ACTION_NAMES[n.i] || 'out ' + n.i)}${n.i === chosen ? ' (chosen)' : ''}</b><div class="row">logit ${val(a[n.l][n.i])}</div>`;
    return `<b>hidden ${L.length > 3 ? n.l + ' ' : ''}unit ${n.i}</b><div class="row">tanh = ${val(a[n.l][n.i])}</div>`;
  }

  const tip = opts.tip;
  if (tip) {
    canvas.addEventListener('pointermove', (ev) => {
      if (!model || !nodes.length) return;
      const r = canvas.getBoundingClientRect(), px = ev.clientX - r.left, py = ev.clientY - r.top;
      let best = null, bd = 12 * 12;
      for (const col of nodes) for (const n of col) {
        const d = (n.x - px) ** 2 + (n.y - py) ** 2;
        if (d < bd) { bd = d; best = n; }
      }
      if (!best) { tip.style.display = 'none'; return; }
      tip.innerHTML = describe(best);
      tip.style.display = 'block';
      const host = canvas.parentElement.getBoundingClientRect();
      tip.style.left = Math.max(0, Math.min(r.left - host.left + best.x + 12, host.width - tip.offsetWidth)) + 'px';
      tip.style.top = (r.top - host.top + best.y + 10) + 'px';
    });
    canvas.addEventListener('pointerleave', () => { tip.style.display = 'none'; });
  }
  if (typeof window !== 'undefined') window.addEventListener('resize', () => { dirty = true; });

  return {
    setModel, clear, idle, capture, draw,
    get model() { return model; },
    get acts() { return fwd ? fwd.acts : null; },
    get chosen() { return chosen; },
    get argmax() { return argmax; },
    get live() { return live; },
    get mismatches() { return mismatches; },
    get decisions() { return decisions; },
    get draws() { return draws; },
  };
}

// A show/hide button for a panel, remembered per browser under `key`.
// onChange(shown) runs once now and after every toggle.
function bindToggle(button, panel, key, onChange) {
  let shown = true;
  try { shown = localStorage.getItem(key) !== 'hidden'; } catch (e) { /* storage unavailable */ }
  const apply = () => {
    panel.hidden = !shown;
    button.textContent = shown ? 'Network: shown' : 'Network: hidden';
    button.setAttribute('aria-expanded', String(shown));
    if (onChange) onChange(shown);
  };
  button.addEventListener('click', () => {
    shown = !shown;
    try { localStorage.setItem(key, shown ? 'shown' : 'hidden'); } catch (e) { /* storage unavailable */ }
    apply();
    button.blur();
  });
  apply();
  return { get shown() { return shown; } };
}

const DinoNetView = { ACTION_NAMES, GROUPS, createForward, createView, bindToggle };

root.DinoNetView = DinoNetView;
if (typeof module !== 'undefined' && module.exports) module.exports = DinoNetView;
})(typeof globalThis !== 'undefined' ? globalThis : this);
