'use strict';
// WebAssembly SIMD forward pass for the policy MLP: the default in every trainer command that
// plays the network (train, assess, bench, finish, session, BC validation) and in the research
// scanner (research/scripts/prereg-bc-robustness/scan.js), off with --no-wasm. Trainer only:
// the browser, eval playback and the dashboard keep policy.js's createNet.
//
// Bit-identical to policy.js createNet, by construction:
//   - Lanes are two adjacent output units of ONE decision (f64x2). Each lane computes
//     bias + w0*x0 + w1*x1 + ... with j ascending, a separate rounded f64x2.mul then
//     f64x2.add per term: the same operations in the same order as policy.js. No relaxed
//     SIMD (relaxed_madd may fuse) and no horizontal sums (they reassociate).
//   - tanh is Math.tanh in a JS loop between the per-layer wasm calls, so it is the very
//     function policy.js uses (a bit-exact port is not practical: see the feasibility study).
//   - argmax uses policy.js's tie-breaking (strict >, lowest index wins).
// test/wasm-net.test.js checks logits bit-for-bit and whole episodes against policy.js.
//
// The module is assembled here from a few opcodes (no toolchain, no npm). It exports
//   layerT(nIn, nPad, w, in, out)
// with byte addresses: w holds the layer's weights input-major, wT[j * nPad + i], then
// nPad biases; nPad = nOut rounded up to even (the pad lane computes 0 and is never read).
// Units go 8 at a time (4 x f64x2), then 2 at a time.
const { FORMAT, OBS_DIM, ACTIONS, observe, numParams } = require('../policy.js');

// --- minimal wasm assembler --------------------------------------------------------
const uleb = (n) => { const o = []; do { let b = n & 127; n >>>= 7; if (n) b |= 128; o.push(b); } while (n); return o; };
const sleb = (n) => {
  const o = [];
  for (;;) {
    const b = n & 127;
    n >>= 7;
    if ((n === 0 && !(b & 64)) || (n === -1 && (b & 64))) { o.push(b); return o; }
    o.push(b | 128);
  }
};
const str = (t) => [...uleb(t.length), ...Array.from(t, (ch) => ch.charCodeAt(0))];
const vec = (items) => [...uleb(items.length), ...items.flat()];
const section = (id, body) => [id, ...uleb(body.length), ...body];
const I32 = 0x7f, V128 = 0x7b;
const op = {
  get: (i) => [0x20, ...uleb(i)], set: (i) => [0x21, ...uleb(i)],
  i32c: (n) => [0x41, ...sleb(n)], add: [0x6a], mul: [0x6c], shl: [0x74], eq: [0x46], gtu: [0x4b],
  block: [0x02, 0x40], loop: [0x03, 0x40], end: [0x0b], br: (d) => [0x0c, ...uleb(d)], brIf: (d) => [0x0d, ...uleb(d)],
  v128load: (o = 0) => [0xfd, ...uleb(0x00), 4, ...uleb(o)],
  v128store: (o = 0) => [0xfd, ...uleb(0x0b), 4, ...uleb(o)],
  load64splat: (o = 0) => [0xfd, ...uleb(0x0a), 3, ...uleb(o)],
  f64x2add: [0xfd, ...uleb(0xf0)], f64x2mul: [0xfd, ...uleb(0xf2)],
};

// layerT body. Params 0..4: nIn, nPad, w, in, out. i32 locals 5..9: i, stride, wp, xp, xend.
// v128 locals 10..13: s0..s3, 14: x.
function layerTBody() {
  const [NIN, NPAD, W, IN, OUT] = [0, 1, 2, 3, 4];
  const [I, STRIDE, WP, XP, XEND, S0, X] = [5, 6, 7, 8, 9, 10, 14];
  const BP = NIN; // nIn is not needed once xend is known; reuse its slot for the bias address
  const c = [];
  c.push(op.get(NPAD), op.i32c(3), op.shl, op.set(STRIDE));
  c.push(op.get(IN), op.get(NIN), op.i32c(3), op.shl, op.add, op.set(XEND));
  c.push(op.get(W), op.get(NIN), op.get(STRIDE), op.mul, op.add, op.set(BP));
  c.push(op.i32c(0), op.set(I));
  // 8 units per block: s_k = bias; for j: s_k = s_k + wT[j][i+2k..] * splat(x[j])
  c.push(op.block, op.loop, op.get(I), op.i32c(8), op.add, op.get(NPAD), op.gtu, op.brIf(1));
  for (let k = 0; k < 4; k++) c.push(op.get(BP), op.get(I), op.i32c(3), op.shl, op.add, op.v128load(16 * k), op.set(S0 + k));
  c.push(op.get(W), op.get(I), op.i32c(3), op.shl, op.add, op.set(WP), op.get(IN), op.set(XP));
  c.push(op.block, op.loop, op.get(XP), op.get(XEND), op.eq, op.brIf(1), op.get(XP), op.load64splat(), op.set(X));
  for (let k = 0; k < 4; k++) c.push(op.get(S0 + k), op.get(WP), op.v128load(16 * k), op.get(X), op.f64x2mul, op.f64x2add, op.set(S0 + k));
  c.push(op.get(WP), op.get(STRIDE), op.add, op.set(WP), op.get(XP), op.i32c(8), op.add, op.set(XP), op.br(0), op.end, op.end);
  for (let k = 0; k < 4; k++) c.push(op.get(OUT), op.get(I), op.i32c(3), op.shl, op.add, op.get(S0 + k), op.v128store(16 * k));
  c.push(op.get(I), op.i32c(8), op.add, op.set(I), op.br(0), op.end, op.end);
  // remaining units, 2 per step
  c.push(op.block, op.loop, op.get(I), op.get(NPAD), op.eq, op.brIf(1));
  c.push(op.get(BP), op.get(I), op.i32c(3), op.shl, op.add, op.v128load(), op.set(S0));
  c.push(op.get(W), op.get(I), op.i32c(3), op.shl, op.add, op.set(WP), op.get(IN), op.set(XP));
  c.push(op.block, op.loop, op.get(XP), op.get(XEND), op.eq, op.brIf(1));
  c.push(op.get(S0), op.get(WP), op.v128load(), op.get(XP), op.load64splat(), op.f64x2mul, op.f64x2add, op.set(S0));
  c.push(op.get(WP), op.get(STRIDE), op.add, op.set(WP), op.get(XP), op.i32c(8), op.add, op.set(XP), op.br(0), op.end, op.end);
  c.push(op.get(OUT), op.get(I), op.i32c(3), op.shl, op.add, op.get(S0), op.v128store());
  c.push(op.get(I), op.i32c(2), op.add, op.set(I), op.br(0), op.end, op.end);
  c.push(op.end);
  return c.flat();
}

function moduleBytes() {
  const types = vec([[0x60, ...vec([[I32], [I32], [I32], [I32], [I32]]), 0]]);
  const imports = vec([[...str('env'), ...str('mem'), 0x02, 0x00, ...uleb(1)]]); // memory, min 1 page
  const funcs = vec([[...uleb(0)]]);
  const exportsSec = vec([[...str('layerT'), 0x00, ...uleb(0)]]);
  const body = [...vec([[5, I32], [5, V128]]), ...layerTBody()];
  const code = vec([[...uleb(body.length), ...body]]);
  return new Uint8Array([0x00, 0x61, 0x73, 0x6d, 1, 0, 0, 0,
    ...section(1, types), ...section(2, imports), ...section(3, funcs), ...section(7, exportsSec), ...section(10, code)]);
}

let compiled = null;
// Compile once per thread. Throws if this runtime has no WebAssembly SIMD.
function wasmModule() {
  if (compiled) return compiled;
  const bytes = moduleBytes();
  if (!simdAvailable(bytes)) throw new Error(NO_SIMD);
  compiled = new WebAssembly.Module(bytes);
  return compiled;
}
// The trainer and the scanner use the kernel unless --no-wasm, so the message names it.
const NO_SIMD = 'this runtime has no WebAssembly SIMD (f64x2); the network runs on the wasm kernel by default, ' +
  'pass --no-wasm to run it in JS (policy.js)';
const simdAvailable = (bytes) => typeof WebAssembly === 'object' && WebAssembly.validate(bytes);
// Throws NO_SIMD at startup rather than in the workers.
const assertAvailable = () => { wasmModule(); };

// --- the network -------------------------------------------------------------------
const align16 = (bytes) => (bytes + 15) & ~15;

// Returns { input, decide, logits }: observe() writes into `input` (a view on the wasm
// memory), decide() returns the action index exactly like policy.js createNet's decide(input).
// Each net owns its own instance and memory (a fixed size, so the views never detach).
function createWasmNet(layers, params) {
  if (params.length !== numParams(layers)) throw new Error(`expected ${numParams(layers)} params, got ${params.length}`);
  const nL = layers.length - 1;
  const meta = [];
  let maxPad = 2, wBytes = 0;
  for (let l = 1; l <= nL; l++) {
    const nIn = layers[l - 1], nOut = layers[l], nPad = (nOut + 1) & ~1;
    meta.push({ nIn, nOut, nPad, w: 0 });
    maxPad = Math.max(maxPad, nPad);
    wBytes += (nIn * nPad + nPad) * 8;
  }
  // Layout: input x | buffer A | buffer B | layer weights. Hidden outputs alternate A, B.
  const X = 0, A = align16(layers[0] * 8), B = A + maxPad * 8, W0 = B + maxPad * 8;
  const pages = Math.max(1, Math.ceil((W0 + wBytes) / 65536));
  const mem = new WebAssembly.Memory({ initial: pages, maximum: pages });
  const { layerT } = new WebAssembly.Instance(wasmModule(), { env: { mem } }).exports;
  const f64 = new Float64Array(mem.buffer);

  // Repack: row-major out x in weights then biases -> input-major, padded, then biases.
  let k = 0, w = W0;
  for (const m of meta) {
    const { nIn, nOut, nPad } = m;
    const base = w / 8, bBase = k + nOut * nIn;
    for (let i = 0; i < nOut; i++) {
      for (let j = 0; j < nIn; j++) f64[base + j * nPad + i] = params[k + i * nIn + j];
      f64[base + nIn * nPad + i] = params[bBase + i];
    }
    m.w = w;
    w += (nIn * nPad + nPad) * 8;
    k = bBase + nOut;
  }

  const nOutLast = layers[nL];
  const outAt = (l) => (l % 2 === 0 ? A : B); // layer l (0-based) writes here
  const outLast = outAt(nL - 1) / 8;
  const input = f64.subarray(X / 8, X / 8 + layers[0]);
  return {
    input,
    decide() {
      let src = X;
      for (let l = 0; l < nL; l++) {
        const m = meta[l], dst = outAt(l);
        layerT(m.nIn, m.nPad, m.w, src, dst);
        if (l < nL - 1) {
          const q0 = dst / 8, q1 = q0 + m.nOut;
          for (let q = q0; q < q1; q++) f64[q] = Math.tanh(f64[q]);
        }
        src = dst;
      }
      let best = 0;
      for (let i = 1; i < nOutLast; i++) if (f64[outLast + i] > f64[outLast + best]) best = i;
      return best;
    },
    // The raw output-layer values of the last decide() (for tests).
    logits: () => f64.slice(outLast, outLast + nOutLast),
  };
}

// The agent loop of policy.js createAgent (same checks, cadence and ACTIONS objects),
// with `net` (a createWasmNet) in place of createNet. Agent state is per call; the net
// may be shared by agents that play one after the other.
function createWasmAgent(model, net = createWasmNet(model.layers, model.params)) {
  if (model.format && model.format !== FORMAT) throw new Error(`unsupported model format ${model.format}`);
  if (model.layers[0] !== OBS_DIM) throw new Error(`model expects ${model.layers[0]} inputs, policy provides ${OBS_DIM}`);
  const NONE = ACTIONS[0], START = ACTIONS[1];
  const repeat = Math.max(1, model.actionRepeat | 0);
  const x = net.input;
  let wait = 0, current = NONE;
  return {
    act(game) {
      if (game.state === 'idle') { wait = 0; current = NONE; return START; }
      if (game.state !== 'running') { wait = 0; current = NONE; return NONE; }
      if (wait <= 0) {
        observe(game, x);
        current = ACTIONS[net.decide()];
        wait = repeat;
      }
      wait--;
      return current;
    },
  };
}

// An agent factory for env.js runEpisode that builds the net once for all episodes of `model`.
function wasmAgents(model) {
  const net = createWasmNet(model.layers, model.params);
  return (m) => {
    if (m !== model) throw new Error('wasmAgents: the factory only plays the model it was built for');
    return createWasmAgent(m, net);
  };
}

module.exports = { assertAvailable, createWasmNet, createWasmAgent, wasmAgents, moduleBytes };
