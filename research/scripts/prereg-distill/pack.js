'use strict';
// Exporter/verifier for pow2-quantized models (research/prereg-distill.md, D2 amended).
// Given a best.json with config.quant.scheme 'pow2' (bc-train / bc-quantize --quant-scheme
// pow2), it checks the claim that EVERY weight and bias is an N-bit integer q in
// [-qmax, qmax] times its layer's 2^-k (trainer/bc.js checkQuantized), writes the packed
// integer form, unpacks it again and checks that the params are identical, and prints the
// storage in bits.
//
// Packed form (.bin, little-endian):
//   'DQP1'                      4 bytes, magic
//   bits                        u8
//   number of layer sizes L     u8
//   layer sizes                 L x u16
//   k per layer                 (L - 1) x i8
//   the integers q              N bits each, two's complement, in the policy.createNet flat
//                               order (per layer: weights row-major, then biases), packed
//                               LSB first; the last byte is zero-padded.
// Storage reported: params N x numParams bits, k 8 bits per layer; their sum is the model's
// payload (the layer sizes and magic are the format's header, reported separately).
//
// Usage: node pack.js --model <best.json> [--out <file.bin>]   (default: next to the model,
//        packed-int<N>.bin). Exit code 1 if the check or the round trip fails.
const fs = require('fs');
const path = require('path');
const assert = require('assert');
const ROOT = path.join(__dirname, '..', '..', '..');
const P = require(path.join(ROOT, 'policy.js'));
const bc = require(path.join(ROOT, 'trainer', 'bc.js'));

const MAGIC = 'DQP1';

// The model's integers (Int32Array) after the check; throws unless the claim holds.
function integersOf(model) {
  const q = model.config && model.config.quant;
  if (!q || q.scheme !== 'pow2') throw new Error('not a pow2-quantized model (config.quant.scheme must be pow2)');
  bc.checkQuantized(model.layers, model.params, q);
  const ints = new Int32Array(model.params.length);
  bc.layerOffsets(model.layers).forEach(({ nout, w, b }, l) => {
    for (let i = w; i < b + nout; i++) ints[i] = model.params[i] * 2 ** q.k[l];
  });
  return { bits: q.bits, k: q.k, ints };
}

function pack(model) {
  const { bits, k, ints } = integersOf(model);
  const L = model.layers;
  if (L.length > 255 || L.some(s => s > 65535)) throw new Error('layer sizes do not fit the header');
  if (k.some(e => e < -128 || e > 127)) throw new Error('k does not fit in 8 bits');
  const head = Buffer.alloc(4 + 1 + 1 + 2 * L.length + (L.length - 1));
  let o = head.write(MAGIC, 0, 'latin1');
  o = head.writeUInt8(bits, o); o = head.writeUInt8(L.length, o);
  for (const s of L) o = head.writeUInt16LE(s, o);
  for (const e of k) o = head.writeInt8(e, o);
  const body = Buffer.alloc(Math.ceil(ints.length * bits / 8));
  const mask = 2 ** bits - 1;
  let bit = 0;
  for (const v of ints) {
    const u = v & mask;
    for (let j = 0; j < bits; j++, bit++) if (u & (1 << j)) body[bit >> 3] |= 1 << (bit & 7);
  }
  return { buf: Buffer.concat([head, body]), headerBytes: head.length, bits, k, n: ints.length };
}

function unpack(buf) {
  if (buf.toString('latin1', 0, 4) !== MAGIC) throw new Error('bad magic');
  let o = 4;
  const bits = buf.readUInt8(o++), nL = buf.readUInt8(o++);
  const layers = [];
  for (let i = 0; i < nL; i++, o += 2) layers.push(buf.readUInt16LE(o));
  const k = [];
  for (let i = 0; i < nL - 1; i++) k.push(buf.readInt8(o++));
  const n = P.numParams(layers), params = new Array(n);
  const offs = bc.layerOffsets(layers);
  let bit = o * 8, l = 0;
  for (let i = 0; i < n; i++) {
    while (i >= offs[l].b + offs[l].nout) l++;
    let u = 0;
    for (let j = 0; j < bits; j++, bit++) if (buf[bit >> 3] & (1 << (bit & 7))) u |= 1 << j;
    const v = u >= 2 ** (bits - 1) ? u - 2 ** bits : u;
    params[i] = v * 2 ** -k[l];
  }
  return { bits, layers, k, params };
}

// Storage of the model's numbers in bits.
const storage = (layers, bits) => {
  const n = P.numParams(layers), kBits = 8 * (layers.length - 1);
  return { params: n, paramBits: n * bits, kBits, totalBits: n * bits + kBits };
};

function main() {
  const args = process.argv.slice(2);
  const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
  const file = opt('--model');
  if (!file) throw new Error('usage: node pack.js --model <best.json> [--out <file.bin>]');
  const model = JSON.parse(fs.readFileSync(path.resolve(file), 'utf8'));
  const p = pack(model);
  const out = path.resolve(opt('--out', path.join(path.dirname(path.resolve(file)), `packed-int${p.bits}.bin`)));
  fs.writeFileSync(out, p.buf);
  const back = unpack(fs.readFileSync(out));
  assert.deepStrictEqual(back.layers, model.layers, 'layers round trip');
  assert.deepStrictEqual(back.k, model.config.quant.k, 'k round trip');
  assert.deepStrictEqual(back.params, model.params, 'params round trip');
  let zeros = 0;
  for (const v of model.params) if (v === 0) zeros++;
  console.log(JSON.stringify({ model: file, out: path.relative(process.cwd(), out), layers: model.layers, bits: p.bits, k: p.k,
    ...storage(model.layers, p.bits), zeroParams: zeros, headerBytes: p.headerBytes, fileBytes: p.buf.length, roundTrip: 'identical' }));
}

module.exports = { pack, unpack, storage, integersOf };
if (require.main === module) {
  try { main(); } catch (e) { console.error(`pack.js: ${e.message}`); process.exit(1); }
}
