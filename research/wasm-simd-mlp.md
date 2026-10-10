# WebAssembly SIMD forward pass for the MLP: feasibility

Date: 2026-10-10. Labels: **STATED** = from a cited source, **INFERENCE** = reasoned, **MEASURED** = measured here (Node v26.9.0, V8 14.6.202.34-node.32, arm64, macOS).

## 1. Verdict

- **Feasible, bit-identical, and worth doing in the trainer only.** A hand-assembled f64x2 SIMD kernel, generic over any hidden-layer shape, gave **0 mismatches in 200,000 random decisions** per variant. That check compared every raw logit bit-for-bit, plus the action, against `policy.js`'s `createNet`, over 9 shapes including odd widths and 2-4 hidden layers. MEASURED.
- **Speed-up of the MLP part: about 1.9x to 2.5x**, measured under heavy machine load (section 5). **Expected end-to-end training speed-up: about 1.5x to 1.9x** (Amdahl, section 7). INFERENCE.
- **`Math.tanh` stays a JS call.** A bit-exact port is not practical (section 3). Tanh becomes the floor: about half of the remaining MLP time.
- **Recommended design:** trainer-only, with the unit-lanes kernel (lanes = adjacent output units within one decision). It needs no lockstep batching, no change to `core.js` or `policy.js`, and so no core-hash change. Details in section 6.

## 2. Q1: V8 / Node 26 SIMD status and fusion

- STATED: fixed-width WebAssembly SIMD ships by default in Chrome 91 and Firefox 89, and is Phase 4 (https://v8.dev/features/simd).
- STATED: in core Wasm all float operators use round-to-nearest-even. `fadd` and `fmul` round each result individually. The only nondeterminism the spec allows is NaN sign and payload (https://webassembly.github.io/spec/core/exec/numerics.html).
- STATED: fused multiply-add exists only as explicit operations. The spec's `fma` rounds once; there is no implicit contraction of `mul` followed by `add`.
- STATED: only the **relaxed SIMD** `f64x2.relaxed_madd` / `relaxed_nmadd` may be fused or unfused, chosen per environment, and "same instruction, same inputs" may differ (https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md).
- INFERENCE: plain `f64x2.add` and `f64x2.mul` are IEEE-754 round-per-operation and are never fused or reassociated. A compiler may only fuse if the module contains the relaxed opcodes, and this prototype emits none. My search for a V8 document confirming "V8 never contracts mul+add" found nothing. That statement rests on the spec, plus the bit-identity measurement.
- Not verified: the V8 version in which relaxed SIMD shipped. I recall Chrome 114 but did not confirm it. The prototype does not use relaxed SIMD.
- MEASURED: `node --v8-options` shows `--wasm-simd-opt` defaults to off. No flag is needed to run SIMD.
- Summation order is under our control: it is the order of the emitted instructions (`s = s + (w * x)`, j ascending). The engine does not reassociate wasm float adds.

## 3. Q2: call costs and tanh

Per-call costs, MEASURED under load (loadavg ~30, so absolute values are inflated):

| Call | Cost |
|---|---|
| JS to wasm, empty export call | ~3 ns |
| wasm to JS import `Math.tanh` (tanh(0) path, so mostly call overhead) | ~6.5-7 ns per element |
| `f64[q] = Math.tanh(f64[q])` in a JS loop (same tanh(0) data, so call overhead cancels) | ~2.2 ns per element |

- Both are cheap per call.
- With realistic inputs, the all-in-wasm import path was slower than the JS-side tanh loop (`import tanh +copy-in` 1589 vs `JS tanh loop +copy-in` 1083 ns/decision, min of trials). So **do the tanh in a JS loop between wasm layer calls**: 3 calls per decision at ~3 ns each.
- The 64 hidden tanh calls per decision are the floor (see section 5 for the matmul-without-tanh figure).

Bit-exact tanh port: **not recommended**.

- STATED: V8's `src/base/ieee754.cc` says "adapted from fdlibm" (https://raw.githubusercontent.com/v8/v8/main/src/base/ieee754.cc). However, on current main `tanh` is a one-line wrapper around `std::tanh`, and `expm1` delegates to LLVM libc's `shared::expm1`. So the implementation is the platform's or a vendored libm, not necessarily a fixed fdlibm.
- MEASURED: a plain-JS transliteration of fdlibm `s_tanh.c` / `s_expm1.c` (`wasm/tanhport.js`) **differs from this Node's `Math.tanh` in the last bit for ~5e-4 of inputs** (2,924 of 6.0M).
- MEASURED: the same port with FMA contraction (a*b+c fused, evaluated with exact BigInt arithmetic, `wasm/tanhfma.js`) gives **0 mismatches in 400,000** inputs. So this arm64 build's `Math.tanh` behaves like fdlibm compiled with fused multiply-add.
- INFERENCE: core Wasm has no fma. Emulating one in software would be slower than the call. `relaxed_madd` is allowed to not fuse, so it cannot be used. A bit-exact in-wasm tanh is therefore not achievable cheaply.
- INFERENCE, a related risk (not new, since the JS path has it too): `Math.tanh` results may differ across CPU architectures or engines (arm64 vs x64, V8 vs SpiderMonkey). Keeping the imported function means the wasm path inherits exactly the JS path's behaviour. Any port would only match one platform.

## 4. Q3: lanes = episodes vs lanes = output units

Summation-order rule: each output unit must compute `bias + w0*x0 + w1*x1 + ...` in index order, with a separate rounded multiply and add. The reference is `policy.js` lines 107-148.

| Scheme | Same order as `policy.js`? | Notes |
|---|---|---|
| Lanes = episodes (2 independent decisions per call) | Yes. Each lane is one decision's unit sum in j order; weights are broadcast. | Needs lockstep batching of 2 games (lane refill, idle lane at tail). Inputs must be written interleaved. Weights stay row-major as in `params`. |
| Lanes = adjacent output units, one decision | Yes. Each lane is a different unit's sum, still j ascending. | No lockstep and one call per decision. Weights must be repacked input-major and padded to an even `nOut` once per theta (cheap). |
| Lanes along j (dot-product plus horizontal add) | **No.** It reassociates the sum. | Rejected. |

- INFERENCE: lanes = output units is the better fit. The trainer loop (`createAgent`, one game per decision) needs no restructuring.
- MEASURED: both kernels pass the bit-identity test.

## 5. Q4: prototype and results

Files (scratchpad only, not in the repo): `/private/tmp/claude-501/-Users-kyle-Desktop-dino-game/a1a47f55-01e9-48b9-afbe-71ea588921bc/scratchpad/wasm/`

- `asm.js`: a small JS wasm assembler (no `wat2wasm`, `wasm-tools` or `wasm-ld` exists; clang has a wasm32 target but nothing to link). It emits the module bytes directly.
- `test.js`: lanes = episodes test.
- `unitlanes.js`: lanes = units kernel plus test.
- `bench.js`, `bench2.js`, `calls.js`: benchmarks.
- `tanhport.js`, `tanhfma.js`: the tanh ports.

Kernels:

- `layer`: lanes = episodes, 4 rows at a time, row-major weights.
- `layerT`: lanes = units, 8 rows at a time, repacked weights.
- Both are fully generic: layer sizes are runtime arguments, with remainder loops.
- The module is ~1 KB, so base64 is trivial.

Correctness (MEASURED): `node test.js 200000` and `node unitlanes.js 200000` each report 0 logit-bit mismatches and 0 action mismatches.

- Shapes: 44-32-32-3, 44-16-3, 44-32-32-32-3, 44-24-24-3, 44-5-3, 44-31-7-3, 44-64-64-3, 44-33-3, and 44-9-11-13-3 for the unit-lanes test.
- Inputs: random with scales 0.5, 2 and 30, with ~20% zeros, so tanh saturates and the near-zero path is covered.
- Weights are perturbed to include large values.
- Every logit was compared with `Object.is` against a verbatim one-row-at-a-time reference, and the action was compared against `createNet`.

Speed, **timings under load**. A trainer job was using most cores, loadavg 26-32 (the machine has 18 cores). One thread, short runs. Results were bimodal, probably P-core vs E-core placement, so the table gives the **minimum over trials**. Absolute numbers are roughly 2x worse than the unloaded 1,164 ns quoted in HANDOFF 2.2. MEASURED:

| 44-32-32-3, ns/decision | Min | Mean |
|---|---|---|
| JS `createNet` | 2075 | 2295 |
| Lanes = units, JS tanh loop, with input copy-in | **1083** | 1194 |
| Lanes = units, import tanh, with input copy-in | 1589 | 1677 |
| Lanes = episodes, import tanh, distinct inputs, no copy-in (`bench.js`, per decision = call time / 2) | ~850-870 | ~1470 |
| Lanes = episodes, JS tanh loop (`bench.js`) | ~850-900 | |
| Lanes = episodes, matmul only, no tanh at all | ~400 | |

| Other shapes (units, JS tanh loop, with copy-in; min) | JS | wasm |
|---|---|---|
| 44-64-64-3 | 5000 | 2500 |
| 44-16-3 | 601 | 295 |

- MEASURED: **the MLP speed-up is ~1.9x to 2.5x under load.** The matmul part alone is ~5x faster; the tanh calls are not.
- Caveat: the "no copy-in" unit-lanes numbers (729 min) repeat the same input every call, so tanh and branch behaviour are unrealistically friendly. They are excluded from the conclusions.
- INFERENCE (scaling the unloaded figures in HANDOFF 2.2): in the unloaded 1,164 ns case, tanh is ~175 ns and the matmul ~990 ns. If the matmul drops ~4-5x, the wasm MLP lands near 350-450 ns/decision, which is about 2.6-3.3x better than 1,164 ns and in line with the scalar C figure (354 ns). This needs an unloaded re-run to confirm.

## 6. Recommended integration design

**Trainer-only, lanes = output units, one wasm instance per worker.**

1. Add the kernel (base64 string plus a small loader) as a new trainer file, e.g. `trainer/wasm-mlp.js`. Do not edit `policy.js` or `core.js`, so the core hash and checkpoints stay valid.
2. Per candidate theta: repack to input-major with even padding, once per generation per candidate (~2.5k doubles, negligible against the millions of frames per episode).
3. Per decision: the observation writes directly into the wasm input region instead of a JS array. The `observe(game, out)` signature already takes an output array, so pass a `Float64Array` view of the wasm memory (the prototype showed a copy-in costs real time).
4. Run `layerT` once per layer, with the JS tanh loop on the hidden outputs between calls. Argmax stays in JS.
5. Trainer workers select the wasm net when it is available and fall back to `createNet` otherwise. The browser, `eval`, the dashboard and the CSS page keep the JS path.

**Why trainer-only is safe given bit-identity.**

- The decisions are provably the same function as `createNet`: same unit sums in the same order (plain f64 `mul` then `add`, no fusion per section 2), and the same imported `Math.tanh`.
- The browser would need no change, and training and playback would agree.
- The gate is `test/net.test.js`, extended with a wasm-vs-JS equality check over random inputs and many shapes. The prototype's test is a ready template.
- Optional later step: share the wasm in the browser too. That changes `policy.js` and so the core hash, which forces `--force` on resume. The gain is small, because the browser runs one agent at a time. Not recommended.
- Lanes = episodes would give higher raw throughput (~850 vs ~1083 ns under load). It requires lockstep pairing of games, and refill within a candidate leaves one lane idle at the tail. Treat it as a second step only if the unit-lanes version is not enough.

## 7. Expected end-to-end speed-up (Amdahl)

Per-frame cost from HANDOFF 2.2: ~1,540 ns, of which MLP ~76-80%, simulation ~16%, observation ~4%.

- If the MLP gets s times faster: speed-up = 1 / ((1-p) + p/s). With p = 0.8:
  - s = 1.9 (measured, units): 1.6x;
  - s = 2.5 (measured, episodes): 1.9x;
  - s = 2.6-3.3 (unloaded estimate): 1.9-2.2x.
- Realistic range: **1.5x to 1.9x** (INFERENCE), consistent with the "1.7-2x" in HANDOFF section 9 item 9.
- With p = 0.76 and s = 2.6, the result is 1.9x.
- Beyond that, tanh limits the gain. Note that in HANDOFF 2.2 the tanh share of the MLP is ~15% when unloaded.

## 8. Risks

- **Tanh cannot be moved into wasm bit-exactly** (section 3). The platform dependence of `Math.tanh` is unchanged from today.
- **Timings here are under load and bimodal.** The estimate needs an unloaded re-run (`node trainer/cli.js bench` with the wasm path wired in).
- Memory-layout bugs: the padded repack. Covered by the equality test.
- **A future relaxed-SIMD or "optimisation" change could fuse operations.** Guard with the equality test in CI and never emit relaxed opcodes.
- Base64 .wasm for a browser path would be a signed-off dependency decision; not needed for the trainer-only design.
- NaN payloads may differ (spec allows it). A NaN input would break the argmax comparison either way and does not occur in normal play.
- Maintenance: a hand-written assembler is brittle. A tiny fixed binary generated once (kept with its generator and test) is simpler.
- Worker startup: each worker instantiates the module (~1 KB); negligible.

## 9. References (accessed 2026-10-10)

- WebAssembly core spec, numerics: https://webassembly.github.io/spec/core/exec/numerics.html
- Relaxed SIMD overview: https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md
- V8 fixed-width SIMD: https://v8.dev/features/simd
- V8 `ieee754.cc`: https://raw.githubusercontent.com/v8/v8/main/src/base/ieee754.cc
- Rust `f64x2_relaxed_madd` docs (one rounding or two): https://doc.rust-lang.org/core/arch/wasm32/fn.f64x2_relaxed_madd.html
- WebKit relaxed madd commit (fused emission): https://lists.webkit.org/pipermail/webkit-changes/2023-June/288057.html
- fdlibm: http://www.netlib.org/fdlibm
