# Pre-registration: lockstep batched play for tier A (and assess / scans)

Session: worktree `agent-a07ee4b4fdb8528bf`, from `main` 7c398cc, 2026-10-12. Machine: the Mac (Apple M5 Pro, 6 P + 12 E cores, node v26.9.0), the reference machine for determinism. User approval: "tier A lockstep prototype: pre-registration plus a scratch prototype of batched games per worker with a wasm SIMD matrix-matrix forward pass; must stay bit-identical."

Written and committed before any measurement. Prototype code goes in `research/scripts/lockstep/` (new files only); nothing in `core.js`, `policy.js`, `test/solver.js` or `trainer/` changes.

## 1. Decision

Whether to implement **lockstep batched play** (K games of one model advanced frame by frame inside one worker, their observations gathered into a K × 44 matrix, one matrix-matrix forward pass per frame) in the trainer's tier A (`scoreSeeds` / `scoreSeedsMany` / `worker.js`), and by extension in `assess` and the scanners.

Options at the end:
- **D1, implement lockstep** in worker.js / pool jobs.
- **D2, widen the existing single-decision kernel** (`trainer/wasm-net.js` `layerT` with more accumulator chains), no lockstep. Cheaper: one file, no job-shape change.
- **D3, do nothing**: tier A stays as it is.

## 2. Where the MLP gain can come from (the mechanism)

- `wasm-net.js` already runs **f64x2**: lanes = 2 adjacent output units of one decision. A lanes = episodes kernel also does one rounded `f64x2.mul` + `f64x2.add` per pair of MACs. **The instruction count per MAC is the same.** SIMD width is not the lever.
- The lever is **instruction-level parallelism**. `layerT` keeps 4 accumulators (8 units) in flight; each accumulator is a chain of nIn dependent `f64x2.add`s. If the add latency is L cycles, a block costs about nIn × L cycles whatever the throughput. Bound for the current kernel (per decision):
  matmul cycles ≈ L × Σ_layers [ceil(nOut / 8) × nIn + (tail units / 2) × nIn ], where the tail loop (nPad mod 8) runs **one** chain per 2 units sequentially.
  For 2×32: 44 × 4 + 32 × 4 + 32 × 2 (the 3-unit output layer: 2 tail chains) = 368 chain steps × L. With L ≈ 3–4 cycles at ≈ 4.5 GHz, ≈ 250–330 ns. The throughput bound (4 FP pipes, 2,528 MACs = 2,528 vector ops) is ≈ 632 cycles ≈ 140 ns. So the current kernel is predicted to be **latency-bound with ≈ 2× headroom** on 2×32 (more for small layers, whose tail chains are short and few).
- A K-lane kernel can keep (units in flight) × (lane pairs in flight) chains, e.g. 4 units × 4 lane pairs = 16, and share each weight splat across lanes and each input load across units. That can approach the throughput bound for **any** shape, including 2×8, 1×6 and every 3-unit output layer, where a single decision has at most 4, 3 and 2 independent chains.
- **Confound (H-alt):** for wide layers (32 units = 16 f64x2 chains) a single-decision kernel with 8 accumulators gets the same ILP with no lockstep. Lockstep's structural advantage is only where unit lanes run out of independent work: narrow layers and the output layer.
- **tanh does not amortize.** Every lane still calls `Math.tanh` once per hidden unit (a JS loop between wasm layer calls). For 2×32 that is 64 calls per game-frame. It is a per-game-frame floor.
- **The scatter.** `observe()` writes 44 contiguous doubles. A lanes = episodes kernel wants inputs interleaved by lane pair ([j][lane]); wasm SIMD has no gather. Either observe into a per-lane scratch and copy-interleave (44 stores per lane per frame), or have the kernel read lane-major rows with scalar splat loads. Either costs something per game-frame.

## 3. Throughput model

Per game-frame cost on one thread, K games in lockstep:

  c(K) = c_sim + c_scatter(K) + c_tanh + c_mm(K) / K

- c_sim: `core.js` step + `observe` + the agent loop; unchanged by lockstep (per lane).
- c_tanh: hidden units × one `Math.tanh` (plus loop), per lane.
- c_mm(K): matmul time of one K-lane call; c_mm(1) is today's `layerT` time.
- Today: c(1) = c_sim + c_tanh + c_mm(1) (no scatter: observe writes straight into the wasm input view).

**Best-case per-thread speedup** S_thread(K) = c(1) / c(K) ≤ (c_sim + c_tanh + c_mm(1)) / (c_sim + c_tanh + c_mm_bound), with c_mm_bound the throughput bound above (optimistic: the K-lane kernel reaches it, scatter free).

Wall time on the pool. P = 18 workers on 6 P + 12 E cores (E per-frame time ≈ ρ × P, ρ to be measured; prior ≈ 1.5–2). F = 1,943,237 frames for every passing rules-v2 game to 1M (prereg-bc-robustness §results), so for passing models the "long-game tail" is **granularity**, not variance: every job of K games costs exactly F × K × c(K) on its core. It becomes variance only for failing or stuck models (lanes drop out; the batch shrinks or refills).

- **Throughput regime** (N games ≫ P·K): wall ≈ N·F / (aggregate lane-frames/s) + tail, tail ≤ one job on the slowest core. Today the tail is one game (F·c_E(1)); lockstep makes it up to one K-game job (F·K·c_E(K)). Speedup ≈ S_thread(K) × (1 − tail share).
- **Latency regime** (N ≤ P): one game per worker today, wall ≈ F·c_E(1). Lockstep can only make it worse (fewer workers, each K·c(K) ≥ c(1) per frame).
- **Regime (a), single-model tier A: 25 games on 18 workers.** Neither clean regime. Today: dynamic dispatch of single games; 18 start, the 7 left go to the first finishers. If ρ < 2 the 7th goes to an E core after its first game, so wall ≈ 2·F·c_E(1); if ρ ≥ 2, ≈ 2·F·c_P(1) on P cores plus E cores finishing at F·c_E(1). Lockstep K = 2 (12 pairs + 1 single on 13 workers): wall ≈ F·2·c_E(2) (a pair on an E core). Predicted speedup ≈ c_E(1)/c_E(2), i.e. S_thread(2), **not** "cannot win", because today's path already pays two sequential games on some core. K ≥ 3 here is worse (wall ≈ F·K·c(K) with fewer workers than games/2).
- **Regime (b), packed tier A: 10 models × 25 seeds = 250 games.** Throughput regime (250/18 ≈ 14 games per worker). Jobs of K same-model games; 25 = 5 × 5 so K = 5 gives 50 jobs (≈ 2.8 per worker); K = 13/12 gives 20 jobs (bad quantization on 18 workers). Predicted wall ≈ today × 1/S_thread(K) + up to one K-job tail; the best K balances the two (prediction: K = 4–5).
- **Regime (c), 1,000-game scan of one model.** Throughput regime; K = 8–16 jobs (125–63 jobs on 18 workers), tail one job. Predicted wall ≈ today / S_thread(K).
- Lanes = candidates (per-lane θ, for ES) is out of scope.

## 4. Hypotheses and predictions

- **H-sim (sim dominates with wasm).** With `--wasm`, the MLP is no longer 76–80 % of the frame (that was JS). Prediction: MLP share ≈ 55–65 % for 2×32, ≈ 25–40 % for 2×8, ≈ 15–30 % for 1×6.
- **H-lat (current kernel latency-bound).** Matmul-only time of `layerT` for 2×32 is ≥ 1.5× the throughput bound (≈ 140 ns). Prediction ≈ 250–330 ns.
- **H-tanh.** tanh is 30–50 % of today's wasm MLP time for 2×32.
- **H-lock (the proposal).** A K-lane kernel at K ≥ 8 brings matmul per decision within 1.3× of the throughput bound for every shape; total per-thread speedup S_thread(8) is **1.25–1.45× for 2×32, 1.05–1.2× for 2×8, ≤ 1.1× for 1×6.**
- **H-alt (wider unit lanes suffice).** An 8-accumulator `layerT` (16 units in flight, single decision) captures ≥ 70 % of the K-lane kernel's per-thread gain on 2×32. Prediction: true for 2×32, false for 2×8 / 1×6 (but there the total gain is small anyway).
- **H-null.** Best-case speedup < 1.2× in every regime the user runs (tier A single and packed, scans) for every shape: lockstep is not worth implementing.

## 5. Premises (checked from existing data or seconds-scale probes)

- P1. Every 2×32 / 2×8 v2 clone used for timing passes 25/25 validation (best.json `validation`), so every game is F frames. From best.json: `runs/exp-bc-duck-stuck/v2-bc-duck-d2w32-s3`…`s12` are 25/25 (s16, s20, s2-ctrl are not; excluded). CHECKED.
- P2. actionRepeat = 1 for every bundled model (models.js). CHECKED. The driver stays generic over actionRepeat anyway.
- P3. `wasm-net.js` is bit-identical to `createNet` (test/wasm-net.test.js), so `createWasmNet(...).logits()` is a valid logit oracle; argmax is checked against `createNet` directly.
- P4. Nothing else CPU-heavy runs on the Mac during timings (`uptime` load ≈ 2.3 at start). Re-check before each multi-worker timing.
- P5. `scoreSeedsMany` and `createPool` are importable (`trainer/cli.js` runs `main()` only when it is the entry point). CHECKED by reading.

## 6. Positive controls

- PC1 (Stage 2): the K-lane kernel run on the same inputs as `createWasmNet` must give bit-identical logits (raw 64-bit patterns, so −0 and NaN payloads count) and the same argmax as `createNet`, for every lane, for K ∈ {1, 2, 3, 8, 16, 25}: random θ (several shapes ≤ 16 × 32, odd widths) × random inputs (scales 0.5, 2, 30, ≈ 20 % zeros), and every bundled model on recorded observations. A non-identical kernel is never accepted.
- PC2 (Stage 3): the lockstep driver reproduces, per seed, the score and an action-sequence hash (FNV over every frame's action index, idle START included) of `runEpisode` with the wasm agent (hash taken through a wrapping `makeAgent`, no trainer edit): 25 validation seeds to 20,000 points, and 3 seeds to 1M, for a 2×8 and a 2×32 v2 clone and the v3 clone `runs/exp-v3-clones/v3-bc-duck-d2w32-s1`.
- PC3 (timing sanity): today's measured single-thread frames/s reproduces the loaded-pool aggregate within the P/E band (scan: 1,000 × F in 211–222 s on 18 workers ≈ 9.0 M frames/s aggregate).

## 7. Stages, horizons and stop rules

- **Stage 1, bound (seconds).** Single thread, unloaded, min of ≥ 5 trials, rules v2, a real game played to 1M by a 25/25 clone of each shape (2×32, 2×8, 1×6): (i) full wasm episode frames/s; (ii) sim alone (replay the recorded actions with `observe`, no net); (iii) `decide` alone on ≥ 200k recorded observations sampled over the game; (iv) tanh alone and matmul alone (strip tanh). Also P vs E: (i) under `taskpolicy -c background` (E cores) for ρ. Then S_thread bound per shape and the regime projections. **Stop rule:** if the optimistic bound is < 1.2× for every regime the user runs, stop and report (conditional null).
- **Stage 2, kernel (minutes).** PC1, then ns/decision per K and shape against `wasm-net.js` single decide, plus the H-alt arm (8-accumulator unit lanes, same bit-identity check). If bit-identity fails: fix or stop.
- **Stage 3, lockstep play (minutes).** PC2, then single-thread frames/s vs K (1, 2, 4, 5, 8, 16).
- **Stage 4, wall time.** Today's path measured: `node trainer/cli.js assess --model <best.json> --target 1000000 --seeds 25` (a `--model` path writes no assess.jsonl) for single model; a `scoreSeedsMany` harness for 10 × 25 (s3–s12). Lockstep projected from Stage 3 numbers and the §3 model; measured with a small multi-worker harness under `research/scripts/lockstep/` only if cheap (each multi-worker run ≤ a few minutes).

Seeds/horizons: timings are deterministic workloads (fixed seeds), so the variance is machine noise; report min and median of trials and treat differences < 10 % in wall time as noise (the §4 benchmark's observed 5–10 %).

## 8. How each result is read

| Result | Reading | Decision |
|---|---|---|
| Stage 1 bound < 1.2× in all regimes | H-null holds within scope (this Mac, these shapes, wasm-net as baseline) | D3; stop |
| Bound ≥ 1.2× but the K-lane kernel fails bit-identity after a fix attempt | method blocked, not a null on the idea | stop; report the failing case |
| H-alt holds (8-acc unit lanes ≥ 70 % of the gain) and narrow shapes gain < 1.1× | the gain is ILP, not batching | D2 (widen `layerT`), no lockstep |
| K-lane kernel + lockstep driver ≥ 1.2× projected wall gain in (b) or (c), H-alt fails | lockstep pays where games ≫ workers | D1 for packed tier A and scans; single-model tier A K = 2 only if ≥ 1.1× |
| Gains only in (c) | scans benefit, tier A does not | D1 for scans only (or D2), tier A unchanged |

## 9. What would reopen a null

- A different baseline: if tier A moves off wasm-net (e.g. back to JS) or to a machine where the add latency / FP pipe count differs (x64 desktops: AVX2 f64x4 is not reachable from wasm SIMD, so the reading may differ).
- Larger shapes than 2×32 (not allowed under the shape cap) or deeper nets where the MLP share grows.
- A cheaper `core.js` step (the sim share bounds everything here).
- A bit-exact in-wasm tanh (rejected in wasm-simd-mlp.md §3), which would remove the per-lane JS call floor.

## 10. Results (Stage 1; the stop rule fired)

Commit 89243be (prereg) + `research/scripts/lockstep/{common.js,bound.js}`. Run 2026-10-12 on the Mac, otherwise idle (load ≈ 2.4, the desktop session only), single thread, rules v2, validation seed 0 played to 1M by each clone (every game 1,943,237 frames, score 1,000,000). Raw rows: `research/scripts/lockstep/out/stage1-pcore.jsonl` (5 trials, min; scheduled on a P core) and `stage1-ecore.jsonl` (3 trials, under `taskpolicy -c background`, which pins to E cores **and** lowers their clock, so it is a pessimistic E figure, about 4.4× the P times). Command: `node research/scripts/lockstep/bound.js --trials 5 --cap 1000000`.

### 10.1 Per-frame split, ns (P core, min of trials)

| | 2×32 (`exp-bc-duck/v2-bc-duck-d2w32-s1`) | 2×8 (`exp-bc-duck/v2-bc-duck-d2w8-s1`) | 1×6 (`exp-distill/v2-bc-duck-d1w6-s1`) |
|---|---|---|---|
| Full wasm episode (runEpisode + wasmAgents) | **881** | **515** | **453** |
| Sim: `core.js` step | 255 | 258 | 256 |
| Sim: `observe` | 80 | 80 | 76 |
| Net decide (copy-in subtracted) | 522 | 174 | 105 |
| — of which `Math.tanh` (5.3 ns × hidden units) | **337** (64 calls) | 85 (16) | 33 (6) |
| — of which matmul (`layerT`, tanh off) | 204 | 56 | 63 |
| — argmax, loop and call overhead (rest) | ≈ 0 (−19, noise) | 33 | 9 |
| Sum of parts / full episode | 858 / 881 (97 %) | 512 / 515 | 437 / 453 |
| MLP share of the frame | 61 % | 34 % | 24 % |
| Matmul throughput bound (§2: f64x2 mul + add per MAC pair, 4 vector ops/cycle, 4.5 GHz) | 142 | 25 | 16 |
| Matmul share of the frame | **24 %** | 11 % | 14 % |

E cores (background QoS): the same shares within a few points (2×32: matmul 851 of 3,901 ns = 22 %, tanh 1,561 = 40 %, sim 1,301 = 33 %; 2×8 and 1×6 matmul 10 % and 6 %). So the reading does not depend on which core class a worker lands on.

### 10.2 Best-case per-thread speedup S_thread = c(1) / c(K)

Sim, tanh and the rest unchanged (§2: per lane, they do not amortize); scatter assumed free.

| | 2×32 | 2×8 | 1×6 |
|---|---|---|---|
| Matmul to the throughput bound (the pre-registered optimistic factor) | **1.08×** | **1.06×** | **1.12×** |
| Matmul to zero (impossible ceiling) | 1.31× | 1.12× | 1.17× |
| Same, E core: bound (scaled by the measured E/P matmul ratio) / zero | 1.07× / 1.29× | 1.06× / 1.11× | 1.05× / 1.07× |

The E-core bound scales the P-core arithmetic floor by the measured E/P matmul time (2×32: 851/204 = 4.2×, so a 593 ns floor); applying the unscaled P floor to E would read 1.23× and is not meaningful.

Regime factors (§3) can only lower these: (a) single-model tier A gains at most S_thread(2); (b) packed tier A and (c) scans gain at most S_thread(K) minus the K-game tail. With the pre-registered factor, every regime is **≤ 1.08× for 2×32, ≤ 1.06× for 2×8, ≤ 1.12× for 1×6**, all under 1.2×. In wall time that is at most ≈ 3–4 s off the ≈ 46 s packed tier A (10 × 2×32) and ≈ 16 s off a ≈ 215 s 1,000-game 2×32 scan, before the scatter (the measured 44-double copy-in alone is 26 ns, 3 % of a 2×32 frame) and the tail.

Stages 2–4 were not run: §7's stop rule fired at Stage 1.

### 10.3 Hypotheses

- **H-sim: holds.** MLP share 61 / 34 / 24 % (predicted 55–65 / 25–40 / 15–30).
- **H-lat: not supported.** The 2×32 matmul is 204 ns, 1.44× the throughput bound (predicted ≥ 1.5×, 250–330 ns). The current `layerT` is already close to the bound for wide layers; the narrow shapes have more relative headroom (2.2× for 2×8, 3.9× for 1×6) but a matmul that is 11–14 % of the frame.
- **H-tanh: holds, more strongly than predicted.** `Math.tanh` is 65 % of the 2×32 wasm MLP (predicted 30–50 %) and the single largest item of a 2×32 frame (337 of 881 ns, 38 %), larger than the `core.js` step (29 %) and the matmul (23 %).
- **H-lock: refuted on the bound.** Predicted S_thread(8) 1.25–1.45× for 2×32; the bound is 1.08× (1.31× even with a free matmul).
- **H-alt: moot.** Widening `layerT` (8 accumulators) can save at most 204 − 142 = 62 ns of an 881 ns 2×32 frame (≤ 1.08×), so D2 is not worth doing either.
- **H-null: holds** within scope.

### 10.4 Reading and decision

Row 1 of §8: **D3, do nothing.** Lockstep batching (and a wider single-decision kernel) cannot reach 1.2× in any regime the user runs, because the per-lane costs that lockstep does not touch, the `core.js` step + `observe` (≈ 335 ns) and the per-unit `Math.tanh` calls (5.3 ns each), are 62 % (2×32) to 86 % (1×6) of a frame.

The null does not hinge on the 4.5 GHz / 4-pipe assumption behind the throughput bound: to reach 1.2× on 2×32, c(K) must be ≤ 858 / 1.2 ≈ 715 ns, so the K-lane matmul would have to cost ≤ 715 − (858 − 204) ≈ **61 ns per decision**, under half the 142 ns arithmetic floor. For 2×8 and 1×6 the required matmul is negative: 1.2× is out of reach even with a free matmul.

Scope of the null: this Mac (M5 Pro, node v26.9.0), shapes 2×32 / 2×8 / 1×6 (the largest allowed width at depth 2 and the shapes the user trains), rules v2 to 1M, `wasm-net.js` at 7c398cc as the baseline, the stop rule's optimistic factor (matmul to the f64x2 throughput bound). Validity: the bound is generous (free scatter, the matmul at the arithmetic limit); a positive result was possible and would have shown as S_thread ≥ 1.2×, as it nearly does for the impossible free-matmul case on 2×32 (1.31×).

What would reopen it (beyond §9):
- **tanh.** The one cost that matters and that lockstep could amortize *if* it ran in wasm SIMD across lanes. That needs a tanh bit-identical to V8's `Math.tanh` on arm64 (fdlibm; prereg-x86-determinism §10 found V8's libm differs between arm64 and x64 by 1 ulp on about 1/3,000 inputs, so the arm64 build's exact operation sequence would have to be reproduced). Even with tanh free and the matmul at its bound, the ceiling would be 881 / (335 + 142) ≈ 1.85× for 2×32 and 515 / (338 + 25) ≈ 1.4× for 2×8, before the cost of the vector tanh itself. This is the only route with a large payoff, and it is a separate feasibility question (wasm-simd-mlp.md §3 judged it impractical).
- **Rules v3** was not measured (the v3 clone was a Stage 3 check that never ran). The sim cost at v3 speeds could differ; the MLP share is unlikely to rise, since the network is the same size.
- **The `core.js` step** (≈ 255 ns, 29–56 % of a frame). Out of scope here (core edits move the core hash and the parity gates).
