# Plan: GPU-accelerated, multi-model BC trainer for Apple Silicon

Authored 2026-10-10 by session mac-6, at the request of orchestrator mac-5, on behalf of the user. The user asked:

> elaborate a GPU accelerated trainer for this Apple Silicon device. If it can do multiple training in parallel, that would also be nice.

**Scope of this document.** It is a planning-only deliverable. Its existence does not authorize carrying out the steps below; execution depends on the user's later instructions and current permissions. Commands in it are for a future executor.

Handoff workflow: assessed applicable (four sequential phases, each verifiable on its own); excluded at the user's request.

Baseline: `main` at `c87d861`. Read `AGENTS.md` in full before starting, especially "Behaviour cloning", "Invariants to preserve" (dashboard visibility, core hash) and "Research method". Reconfirm before Phase 1 that `trainer/bc_train.py`, `trainer/bc.js` (`trainPython`, `assembleBest`), `trainer/cli.js` (`bcTrainCmd`) and `trainer/dashboard.js` (`trainLineIn`, `runKind`) still match the description in §2.

---

## 1. Decisions already made

These come from the user's answers in the planning Q&A on 2026-10-10. Don't reopen them.

| # | Decision | Source |
|---|---|---|
| D1 | Framework: **MLX** (`mlx==0.29.3`, the last release that supports the repo `.venv`'s Python 3.9.6), installed **into the repo `.venv`** next to numpy | user (Q&A) |
| D2 | Many-model training is a **new subcommand, `bc-train-batch`**. Each model still gets its own normal run directory | user (Q&A) |
| D3 | Scope is **BC training only**. Evaluation (tier A, tier B, screens, `assess`) and ES stay on the CPU. The reasons are in §6 | user (Q&A) |
| D4 | No HANDOFF.md workflow | user (Q&A) |
| D5 | The node and numpy backends are unchanged. The numpy backend must keep reproducing the shipped clone's params exactly | AGENTS.md |
| D6 | Shape cap: ≤ 16 hidden layers and ≤ 32 units per layer | AGENTS.md (user order) |

The user approved the `.venv` install as part of this plan. The executor still installs it only once the user has authorized implementation (Phase 0).

## 2. Current state (inspected at `c87d861`)

- **`trainer/bc_train.py`** (numpy, float32 throughout) is a script with top-level code, not an importable module. It has the following parts:
  - RNG: `rng = np.random.default_rng(seed)` draws, in this order: the init `rng.normal` once per layer, then once per epoch `rng.permutation(N)`, then `rng.choice(N, 200000, replace=False)` for the recall subsample.
  - Features: distillation (`--teacher-logits`, soft or hard), QAT (`--quant-bits`, `--quant-scheme row|layer|pow2`; float64 quantization on the CPU), `--snapshot-dir`, `--extra-demos` and `--transition-weight`.
  - Output: one JSON line per epoch on stdout, and `{layers, params, log[, scales|k, transitions]}` in `--out`.
- **`trainer/bc.js` `trainPython`** spawns `bc_train.py` and streams its JSON lines to `onEpoch`. `assembleBest` builds `best.json`.
- **`trainer/cli.js` `bcTrainCmd`** (about lines 938–1095) handles one run directory. It:
  - validates the options, checks the demos (`checkDemos`, core hash) and calls `acquireLock(dir)`;
  - writes the start line `bc-train: dir=… backend=… layers=… epochs …` to stdout (the user redirects stdout to `<dir>/train.log`);
  - writes the `bc-metrics.jsonl` rows;
  - trains;
  - runs tier A (the 25 `validationSeeds(25)` full games on a worker pool);
  - writes `best.json` and releases the lock.
- **Dashboard** (`trainer/dashboard.js`):
  - A directory is a BC run if it has `bc-metrics.jsonl`, or `best.json` with a BC `config.method`, or `train.lock` plus a log containing `bc-train: dir=`.
  - `trainLineIn` reads the newest `*.log` in the directory, or `<group>/<run>.log` for a nested run, and parses `backend=(\S+)`, `layers=`, ` epochs N`, ` demos …`.
  - `running` means `train.lock` holds a live pid. `acquireLock` (`trainer/store.js`) writes `{pid, startedAt}`, so one process can hold the locks of many directories.
- **Precedent for "two backends that don't match bitwise"**: `research/prereg-bc-pipeline-e2e.md` accepted the node backend against the python backend with these criteria:
  - per-epoch loss within about 1e-3 of the reference;
  - tier A 25/25;
  - tier B 0/1,000 on the same 1,000 seeds (`py-s1` 176 s, `node-s1` 170 s).

  §5 reuses that bar.

### Measured costs (where time goes today)

| Item | Cost | Source |
|---|---|---|
| `bc-train` d2w32, python, rules v2 (2.06M rows) | 19 s wall: 14.3 s training (≈ 0.95 s/epoch ≈ 0.47 ms/step over 2,014 steps), 4.2–5.0 s tier A | `research/prereg-teacher-duck-stuck.md` §cost table |
| `bc-train` d2w8 | 15 s: 12.3 s training, 2.8 s tier A | same |
| Opening scan per model | 5–13 s | `research/prereg-v3-clones.md` §9 |
| Tier B per model, rules v2, 18 workers | 2×32 ≈ 175–295 s (v3: 82–137 s at 0.465× the frames) | same |
| A whole 10-seed arm (train, tier A, screens, ≤ 2 tier B, duck share) | ≈ 15–25 min, of which `bc-train` is ≈ 3–5 min | same, arms A and B |
| Per-frame cost of a game with a 32,32 MLP (JS) | ≈ 1,540 ns, of which the MLP is 76–80 % | `research/wasm-simd-mlp.md` |

### Probes run for this plan (2026-10-10)

The Mac was otherwise idle: the rules-v3 runner was paused by mac-5 at the user's order. All probes used random data and did timing only. The scripts are in the session scratchpad (not durable), so the commands are given here so that Phase 4 can reproduce them.

**Probe 1: MLX 0.29.3, scratchpad venv, Python 3.9.6, macOS 27.0.1, Apple M5 Pro, 48 GB.**
- Setup: N models stacked as `(N, out, in)` weights, batch 1024 per model, a fwd+bwd+Adam step in `mx.compile`, `mx.eval` once per step, and 200 timed steps.
- Result: MLX installs and runs on the GPU (`Device(gpu, 0)`).

| Shape | numpy, 1 model (same step, random batches) | MLX N=1 | MLX N=10 | MLX N=40 | MLX N=100 |
|---|---|---|---|---|---|
| 2×32 | 0.27 ms/step | 0.50 | 0.51 (0.051/model) | 1.15 (0.029/model) | 2.46 (0.025/model) |
| 2×8 | 0.23 | 0.47 | 0.42 (0.042/model) | 0.76 (0.019/model) | 1.51 (0.015/model) |
| 16×32 | 1.20 | 0.65 | 1.59 (0.159/model) | 4.99 (0.125/model) | 13.3 (0.133/model) |

- **Reading.** One model on the GPU is about 2× slower than numpy for small shapes, because launch and sync overhead dominate. Stacking N models is close to free up to N ≈ 10, and the cost per model flattens by N ≈ 40.
- Aggregate speedup over sequential numpy: ≈ 5× at N=10 and ≈ 9–15× at N=40–100. Deep 16×32 nets gain ≈ 2× even alone.

**Probe 2: parallel numpy (the no-new-dependency competitor).**
- Setup: K concurrent copies of the same numpy step, using permutation batches, with `VECLIB_MAXIMUM_THREADS=1`. The M5 Pro has 6 performance and 12 efficiency cores.
- Result (2×32):
  - single process: 0.287 ms/step;
  - K=6: 0.31 ms/step each, ≈ 5.6× aggregate;
  - K=10: 0.35–0.36 each, **≈ 8× aggregate**;
  - K=18: 1.2–1.8 each, ≈ 3× aggregate (worse: the efficiency cores and contention).
- **Reading.** For a 10-seed arm of small nets, **10 parallel numpy processes are about as fast as the GPU** (≈ 8× against ≈ 5× at N=10). The GPU wins on three counts:
  - larger N (≥ 40 models, ≈ 10–15×);
  - deep nets;
  - leaving the CPU free, so the 18 CPU workers can screen arm k while the GPU trains arm k+1.

  The parallel-numpy mode must therefore be part of the design and the benchmark (§7, H-cpu). It isn't a footnote.

## 3. Framework choice (D1), with the rejected alternatives

**MLX.**
- It is Apple's array framework, built for unified memory.
- `mx.compile` fuses a whole training step, which matters because this workload is bound by launches, not compute.
- Batched matmul over a leading model axis is native.
- The install is about 30 MB, and the probe confirmed that it installs and runs on this machine's Python 3.9.6.
- The pin to 0.29.3 is forced by Python 3.9. A newer MLX needs a newer `.venv` interpreter, which is out of scope.

**Rejected:**
- **PyTorch MPS.**
  - Python 3.9 caps it at torch 2.8.0.
  - The install is large (hundreds of MB).
  - Import takes about 2–3 s against a 15 s job.
  - The per-op MPS dispatch overhead is higher than MLX's compiled graphs.
  - MPS has documented nondeterministic kernels.
  - It offers nothing this workload needs that MLX lacks.
- **WebGPU or Metal from Node.** Both need a native addon (Dawn bindings or a Metal bridge), which breaks the repo's no-npm-dependency rule for `trainer/`. A hand-written Metal/WGSL MLP plus Adam would also be a third implementation of the algorithm to keep in sync.
- **A custom Swift/Metal binary.** It is a build step, and the repo has none. Its maintenance cost is out of proportion to a 15 s job.

The no-npm rule is unaffected, because MLX lives behind the existing python-backend boundary (`trainer/bc_train*.py`, spawned by Node, with the `PYTHON` env var override). The game, browser and CSS paths never touch it.

## 4. Design

### 4.1 Ownership

These decisions follow the repo's conceptual-first rule.

- **One algorithm definition, three numeric engines.** The training algorithm (sqrt-balanced class weights, transition weights, distillation targets, QAT rules, Adam, cosine LR, RNG call order, the recall subsample, the output schema) is defined today by `bc_train.py`, with `bc.js` as a port.
- **Host-side helpers move into `trainer/bc_common.py`.** The MLX engine needs every host-side piece of `bc_train.py` unchanged: argument parsing, row loading and `--extra-demos`, class and transition weights, distillation target preparation, the per-seed RNG streams, `quantize`, `pow2_exponent`, `quantize_pow2`, `flat_params`, `write_json`, the snapshot writer, and the epoch-row and result formats. The move is gated on byte-identical output (Phase 1). The numpy step code stays in `bc_train.py`, which becomes `main()` plus the numpy step and imports the helpers.
- **The MLX step lives in `trainer/bc_train_mlx.py`.** It always trains a **stack of N ≥ 1 models of one shape**. `bc-train --backend mlx` is the case N = 1, and `bc-train-batch` passes N per shape group. There is only one MLX implementation.
- **Per-run finishing moves into one Node function.** The steps after training (`quantRecord`, tier A, the `config` object, `assembleBest`, writing `best.json`, releasing the lock, the final log line) move out of `bcTrainCmd` into `finishBcRun(dir, result, ctx)` in `trainer/cli.js`, and both commands call it. The same goes for the start-line formatter (`bcStartLine(...)`), so `bc-train-batch` writes exactly the line the dashboard parses.
- **State lifetime.**
  - Per run directory: the lock, `train.log`, `bc-metrics.jsonl`, `snapshots/` and `best.json`. These are never shared across models.
  - Per batch process: the demo arrays on the GPU, shared read-only by all models in the process.
  - Per model: its own `default_rng(seed)`, params, Adam state and permutation.

### 4.2 Seed semantics (what makes correctness testable)

Each model k with seed s_k keeps **its own numpy `default_rng(s_k)` on the CPU, with exactly the numpy backend's call order**:
1. the init normals, layer by layer;
2. per epoch, `permutation(N)`;
3. after each epoch, `choice(N, 200000, replace=False)`.

The init weights, permutations and recall indices are uploaded to the GPU. The result is that **`--seed k` means the same initial params and the same batch order on every python-family backend**. The only remaining difference is float32 accumulation order on the GPU, which makes the equivalence test in §5 tight.

Host cost: one `permutation(2.06M)` is about 20 ms per model per epoch. For N = 40 that is ≈ 0.8 s/epoch, comparable to the GPU epoch (≈ 2.3 s). Generate epoch e+1's permutations on a background thread while the GPU runs epoch e. Upload them as int32 `(N, steps·B)`.

### 4.3 The MLX step (stacked N)

- **Params.** `W_i` has shape `(N, out_i, in_i)` and `b_i` has shape `(N, 1, out_i)`, both float32. Adam `m` and `v` are float32 with the same shapes.
- **Batch gather.** `xb = X[idx]` gives `(N, B, 44)`, from `X` resident on the GPU (shared). `yb` and `wb` are gathered the same way. Distillation `Q[idx]` gives `(N, B, 3)`.
- **Forward and backward.** Write the gradient **by hand as in `bc_train.py`**: `g = p − onehot`, scaled by `wb / wb.sum()`, then `gW = gᵀ h` and the tanh backprop through the fake-quantized `Wf`. Don't use autodiff. That keeps the arithmetic expression for expression comparable with numpy, makes the QAT straight-through estimator exact by construction, and avoids a sum-over-models loss.
- **Step function.** The whole step (gather, forward, backward, Adam with the cosine LR computed on the host as float32 scalars `lr_t`, `1−0.9^t`, `1−0.999^t`) is one `mx.compile` function, with one `mx.eval` per step, as in Probe 1.
- **Loss and recall.** The epoch loss sums accumulate on the device and are read once per epoch. Recall is a forward pass on the 200k subsample per model on the GPU, then argmax.
- **QAT** (`--quant-bits`):
  - The per-step fake quantization runs on the GPU in float32: `row` and `layer` scales are `max|W| / qmax`, and `pow2` uses `k = floor(log2(qmax/m))` with the same ±1 correction loop expressed as two vectorized `where` checks. Multiplying by a power of two is exact in float32, so given the same `m`, `k` matches.
  - This is training-only arithmetic and may differ from float64 at rounding ties.
  - **The saved params, `scales` and `k` are computed on the CPU by the shared float64 `flat_params` and `quantize*` from the final float32 weights**, so `bc.quantRecord`'s "exactly on the grid" check still holds.
- **Distillation.**
  - Hard labels (`TL.argmax(1)`, first index wins) are computed on the CPU from the float64 logits file, as now.
  - Soft targets `Q` are computed on the CPU (float64 to float32, as now) and uploaded.
  - `distillLoss` is accumulated like `loss`.
- **`--snapshots`.** After each epoch's last step, pull the N models' params to the host (≈ 70 KB per model at 16×32) and write `epoch-NN.json` per model with the shared writer. It draws no RNG.
- **`--extra-demos` and `--transition-weight`.** These are host-side row preparation (shared helpers), so they are unchanged.
- **Output.**
  - Each stdout JSON line gains `"model": k` (0-based within the process). The single-model path (N = 1) still prints `"model": 0`, and Node ignores the field for `bc-train`.
  - `--out` becomes `--out-dir`, written as `model-<k>.json` with the existing result schema.
  - The result gains `engine: {name: 'mlx', version, device}`.

### 4.4 Memory

All of this is unified memory (48 GB):

| Item | Size |
|---|---|
| Demos `X`, rules v2 (2.06M × 44 float32) | 363 MB (+ 16 MB labels and weights) |
| Demos `X`, rules v3 (≈ 3.6M rows) | ≈ 630 MB |
| Distillation `Q` | ≈ 25–43 MB |
| Permutations (N × rows × int32), double-buffered | N = 40: 330 MB each |
| Activations, N = 40, 16×32, B = 1024 (forward plus backward intermediates) | ≈ 250 MB |

**Cap a group at N = 40.** Probe 1 shows no gain per model past about 40, and larger groups only add memory and latency to the first result. A group larger than 40 is split into chunks of up to 40.

### 4.5 Shape grouping

- Models are grouped by **exact hidden list**, for example `32,32` or `6`. Each group is one compiled function with stacked weights.
- Groups run **one after another** on the GPU. Different depths cannot share a batched matmul chain.
- Don't pad narrower nets to a common width. Padding would waste 4–16× compute on 1×4 against 2×32 and would add masks to the Adam state and QAT scales, which is a new source of bugs. Revisit only if a sweep has many tiny groups and Phase 4 shows that per-group launch overhead dominates.
- **Overlap.** Once group g finishes training, its tier A runs on the CPU worker pool while group g+1 trains on the GPU. The rest stays sequential.

### 4.6 Node integration

**`trainer/bc.js`.**
- Generalize `trainPython(binFile, outFile, o, root, onLine)` to `trainPythonEngine(script, …)`, or add a `script` field to `o`, so that one spawn and stream function serves `bc_train.py` and `bc_train_mlx.py`.
- `pythonPath()` is unchanged.
- Add `engineAvailable('mlx')`: run `python -c "import mlx.core"` once and give a clear error (`install mlx==0.29.3 into .venv, or set PYTHON`).

**`bc-train --backend mlx`** (`cli.js` `bcTrainCmd`).
- Accept `mlx` next to `node|python`.
- Every python-only flag (`--snapshots`, `--extra-demos`, `--transition-weight`) is also allowed for `mlx`.
- The `config` gains `backend: 'mlx'` and `engine` (MLX version, device).
- Everything else is the shared `finishBcRun`.

**`bc-train-batch`** (new subcommand).

```text
node trainer/cli.js bc-train-batch --dir-pattern 'runs/exp-x/v3-bc-duck-d{depth}w{width}-s{seed}'
     --shapes 32,32/6 --seeds 1-10 --demos <dir> [--backend mlx|python] [--jobs K]
     [every bc-train option except --dir/--hidden/--seed: --epochs --batch --lr --target --workers --wasm
      --force --overwrite --snapshots --distill-from … --quant-bits … --extra-demos … --transition-weight]
```

- **Shapes.** `--shapes` is `/`-separated hidden lists. `{depth}` and `{width}` are taken from each list; a non-uniform list uses `{hidden}`, with `-` joining the widths.
- **Grid.** The cross product shapes × seeds names the run directories. Shared options apply to every model; only shape and seed vary.
- **Start.**
  - Refuse if any directory already has `best.json` (unless `--overwrite`) or a live lock.
  - Check the demos once.
  - Lock **every** directory with this pid.
  - Write `<dir>/train.log` with the exact `bc-train: dir=… backend=…` start line, plus ` batch=<i>/<n>`.
  - Then train.
- **During training.**
  - Each epoch row is routed by `model` to its directory's `bc-metrics.jsonl`, as `{backend, ...row}` without the `model` field, so the dashboard sees the usual row.
  - The matching `epoch N:` line goes to that directory's `train.log`.
  - Each model's `train.log` gets the validation and final lines from `finishBcRun`.
- **`--backend mlx`** (default): one `bc_train_mlx.py` process per shape group (chunks of ≤ 40).
- **`--backend python --jobs K`**: K concurrent `bc_train.py` processes with `VECLIB_MAXIMUM_THREADS=1` and `OMP_NUM_THREADS=1`, default K = 6 (the performance cores). This is the parallel-CPU path from Probe 2. It needs no new dependency, and its output per model is byte-identical to `bc-train --backend python`, because the same script runs with the same arguments and the thread count doesn't change Accelerate's float32 results.

  Verify that last claim in Phase 1 (V1.3). If it fails, `--jobs` keeps the default threading, and the byte-identity claim is dropped from the docs.
- **Failure.**
  - If an engine process exits non-zero, the models it was training get no `best.json`. Their directories keep the partial `bc-metrics.jsonl` and a `train.log` line `bc-train-batch: engine failed (exit N)`, and their locks are released.
  - Other groups continue.
  - The command exits non-zero, listing the failed directories.
  - On SIGINT, kill the engine children and release every lock.
- **`config`** adds `batch: {id: <ISO start time + pid>, group: <hidden>, index, size}`. `trainer` is `trainer/cli.js bc-train-batch --backend <b>`.

**Dashboard.**
- No layout change: every model is an ordinary BC run directory.
- `trainLineIn` already parses `backend=mlx` and finds `<dir>/train.log`, `running` already comes from the lock pid, and the playback reads `best.json` and `snapshots/` as before.
- Required in the same change:
  - extend `test/dashboard-check.js`'s temp `runs/` fixture with (a) a finished `backend=mlx` run and (b) a batch in progress: three directories locked by one live pid, each with a `train.log` start line and a partial `bc-metrics.jsonl`;
  - assert that all three are listed as running BC runs with their backend.
- Also fix the hint string in `trainer/dashboard.html` ("bc-train --snapshots, python backend") to say "python or mlx backend".

**Docs.** Update `AGENTS.md` (Commands, the BC bullets, the backend description in the model-provenance table) and the `trainer/cli.js` header in the same change.

**Untouched.** `core.js`, `policy.js` (so the **core hash is unchanged**), the node backend, the numpy numerics, `assess`, `scan.js` and the ES trainer.

## 5. Correctness: what replaces bit-identity

The numpy backend stays the reference, and it must keep reproducing the shipped clone exactly (D5). MLX can't be bit-identical: GPU float32 reductions use a different order, and the GPU stream has no float64. The python backend is already float32, so float32 is not a downgrade. The equivalence is tiered, from cheap to expensive, and every tier is a gate.

| Tier | Check | Bar | Why this bar |
|---|---|---|---|
| **E0, one step** | For each feature combination (plain; soft distillation τ=2 α=0.5; hard distillation; QAT row/layer/pow2 at 4 bits; transition weight 3), the same init and the same batch: MLX gradients and post-Adam params against a **float64 numpy recomputation of the same step** | max relative error ≤ 1e-4 per tensor | catches logic bugs (wrong transpose, missing weight, STE) independently of chaotic drift |
| **E1, epoch 1** | `bc_train.py` against `bc_train_mlx.py`, N=1, same seed, 1 epoch, shipped demos `runs/exp-bc-e2e/demos.json`, 2×32 | loss Δ ≤ 1e-3 (the e2e precedent); max \|ΔW\| / max\|W\| ≤ 1e-3 | same init and batch order (§4.2), so only accumulation order differs |
| **E2, stacking** | Model k of an N=10 stack against an MLX N=1 run with seed k, 1 epoch | as E1. Bitwise equality is recorded if it happens, but not required | stacking changes kernels, not math |
| **E3, determinism** | The same MLX run twice | **recorded, not gated**: bitwise equal or not. If not equal, the AGENTS.md text says "MLX: not bitwise reproducible run to run; the float32 numerics are verified by E0–E2" | the repo claims per-seed determinism for node and python; be truthful about MLX |
| **E4, behaviour** | The pre-registered §7 benchmark arms: 10 seeds 2×32 `teacher-jump` (`runs/exp-bc-e2e/demos.json`, rules v2), numpy against MLX | MLX: tier A 25/25 on every seed where numpy is 25/25; final-loss distributions overlap (\|mean Δ\| ≤ 2 SD of numpy's seed spread); **same-seed decision agreement on held-out demo rows (`runs/exp-bc-duck/v2-demos-duck-s41-60`, labels ignored) ≥ the numpy s1-vs-s2 agreement** (the seed-noise floor); 2 MLX models tier B 0/1,000 on seeds 30,002,000–30,002,999 (as e2e) | per-seed outcomes are chaotic over 30k steps, so equivalence is distributional, with the seed noise floor as the yardstick |
| **E5, features end to end** | `--quant-bits 4 --quant-scheme pow2` and `--distill-from` with MLX, 1 seed each | `bc.quantRecord` accepts (on the grid); best.json loads in `assess`; `--snapshots` writes 15 files that the dashboard plays | integration, not numerics |

Note: E4 does **not** require MLX s_k to match numpy s_k per seed. For example, numpy's `v2-bc-duck-d2w32-s2` being the stuck-duck seed says nothing about MLX s2.

## 6. What the GPU does not help, with the reasons

**Evaluation (tier A, tier B, opening and post-step screens, `assess`, duck share).**
1. **Latency.** The game is `core.js` in JS, and each frame's action feeds the next frame. A batched GPU policy would need one GPU round trip per lockstep frame. A rules-v2 game is ≈ 1.94M frames, so even all 1,000 tier-B games in lockstep take ≈ 1.94M round trips. At ≈ 50–100 µs per dispatch and sync, that is **≈ 100–200 s of sync latency alone**, against 175–295 s for the whole scan on 18 CPU workers today. That is no gain, before counting the JS↔python bridge.
2. **Fidelity.** The browser, `model.js`, `assess` and every certification decide with `policy.js` in float64 with `Math.tanh`. The GPU is float32 only. Clone margins as small as 5.8e-5–2.7e-4 have been measured, so a float32 evaluator would sometimes take a different action than the shipped page. It would be a screen at best, never a certification, and the existing CPU screens already cost seconds to minutes.
3. **The CPU route, for the record (not part of this plan).** A sizable eval speedup would come from lockstep batching of games **within each CPU worker**, with a WASM SIMD matrix-matrix forward pass. `research/wasm-simd-mlp.md` already sketches "lanes = episodes". It keeps float64 decisions bit-identical to `policy.js` and is a separate proposal.

**ES (`trainer/es.js`).** The work is the same per-frame, branchy game simulation with a different θ per candidate, so it has the same lockstep latency problem as point 1 and the same float64-selection concern as point 2. The ES update (noise regeneration, ranks, Adam over ≤ 17k params) is microseconds to milliseconds and isn't the bottleneck. **Verdict: no GPU for ES.**

### Expected end-to-end speedup for a typical 10-seed arm

Example: 2×32, rules v2, the stuck-duck/v3-clones procedure.

| Stage | Today (sequential `bc-train` python) | GPU `bc-train-batch` (N=10) | `bc-train-batch --backend python --jobs 6` |
|---|---|---|---|
| Training, 10 models × 15 epochs | 10 × 14.3 s = **143 s** | 15 × (2,014 steps × 0.51 ms + ≈ 0.2 s permutations) ≈ **18 s** (Probe 1) | ≈ 143 s / 5.6 ≈ **26 s** (Probe 2, K=6) |
| Tier A, 10 × 25 games | 10 × 4.5 s = 45 s | 45 s (CPU; overlappable only across shape groups) | 45 s |
| `bc-train` stage total | **≈ 190 s** | **≈ 63 s (≈ 3.0×)** | **≈ 71 s (≈ 2.7×)** |
| Rest of the arm: openings, ≤ 2 tier B, duck share, margins, agreement | ≈ 12–20 min | unchanged | unchanged |
| **Whole arm** | ≈ 15–25 min | **≈ 1.1–1.15× faster** | ≈ 1.1× |

- **The bottleneck is tier B** and the other game-playing screens: JS simulation on the CPU, ≈ 1.94M frames per game × 1,000 games.
- **Amdahl's bound.** Training is ≈ 10–20 % of an arm's wall time, so even infinitely fast training gives ≤ ≈ 1.25× end to end.
- **Where the GPU does pay off:**
  1. **Training-heavy studies:** shape sweeps, snapshot or QAT studies, many seeds with cheap screens. 40 models train in ≈ 35 s instead of ≈ 10 min (≈ 16× against today's measured 0.95 s/epoch).
  2. **Overlap:** the GPU trains arm k+1 while the CPU screens arm k. This needs the orchestrator to schedule two jobs (GPU training plus CPU screens) as compatible, which loosens the current "one heavy job" rule to "one CPU-heavy job plus one GPU job". That is the user's call; see §9.
  3. **Deep nets** (≥ 8 layers): ≈ 2× even for a single model.

## 7. Pre-registered benchmark (Phase 4)

**Decision informed.**
- (a) Whether `bc-train-batch` defaults to `--backend mlx` or `--backend python --jobs K`.
- (b) Whether studies should plan "GPU training overlapped with CPU screens".
- (c) Whether the MLX backend is accepted as equivalent for research runs, per §5 E4.

**Hypotheses and predictions.**
- **H-gpu:** MLX N=10 trains a 10-seed 2×32 arm (15 epochs, 2.06M rows) in ≤ 30 s wall (≥ 4.7× against sequential numpy's measured 143 s), and N=40 in ≤ 60 s.
- **H-cpu:** `--backend python --jobs 6` trains the same arm within 1.5× of MLX's wall time. If so, MLX's advantage is limited to N ≥ 40, deep nets and freeing the CPU.
- **H-host:** the host side (permutations, gather, the snapshot and recall pulls) dominates MLX's epoch time. It would show as < 50 % GPU utilization in `powermetrics --samplers gpu_power` (or Activity Monitor's GPU history), and as wall time not improving from N=10 to N=40 per model.
- **H-overlap:** while MLX trains N=40, a concurrent opening scan (`opening.js`, 1,000 games) takes ≤ 1.15× its solo wall time. That means the GPU job doesn't steal meaningful CPU.

**Premises**, checked before running and each a seconds-scale check:
- the Mac is otherwise idle: ask mac-5 to pause other jobs;
- Probe 1 and Probe 2 reproduce within 20 %;
- the E0–E2 gates have passed.

**Positive controls.**
- **PC1:** sequential `bc-train --backend python` d2w32 seed 1 reproduces its committed params sha (or the shipped clone on its demos with `--seed 1`, as AGENTS.md states). The baseline then stays the baseline.
- **PC2:** the MLX N=1 timing reproduces Probe 1's 0.50 ms/step ± 20 %.

**Arms**: wall time measured by `/usr/bin/time -l`; GPU utilization sampled.

| Arm | Command (run dir under `runs/exp-gpu-bench/`) |
|---|---|
| A0 | sequential: 10 × `bc-train --backend python --hidden 32,32 --seed k` (k = 1–10) |
| A1 | `bc-train-batch --backend python --jobs 6 --shapes 32,32 --seeds 1-10` |
| A2 | `bc-train-batch --backend mlx --shapes 32,32 --seeds 1-10` |
| A3 | `bc-train-batch --backend mlx --shapes 32,32 --seeds 1-40` |
| A4 | `bc-train-batch --backend mlx --shapes 6/8,8/32,32/32×16 --seeds 1-10` (mixed shapes, grouping and overlap) |
| A5 | A3 concurrently with `opening.js` on the shipped clone, against `opening.js` solo |

**Variance and power.**
- Wall-time noise from probe repeats was ≤ 5 %, so one run per arm suffices for the predicted 3–10× effects.
- Repeat an arm (n = 3) only if it lands within 20 % of a decision threshold.
- E4's behavioural arms (10 seeds each) come from A0 and A2's models, so they cost no extra training. Tier B covers 2 MLX models only (≈ 6–10 min).

**Total machine time:** ≈ 25–35 min, most of it in tier A and the two tier B scans.

**Reading.**

| Outcome | Reading and action |
|---|---|
| H-gpu holds, H-cpu fails (A1 > 1.5 × A2) | default `--backend mlx`; plan overlapped studies (if the user approves, §9) |
| H-gpu holds, H-cpu holds | default `--backend mlx` only if H-overlap holds (it frees the CPU); otherwise default `python --jobs 6`, and MLX is kept for N ≥ 40 and deep nets |
| H-gpu fails, and H-host is shown | fix the host side (thread the permutations, upload once per epoch) and re-measure once. If it still fails, default `python --jobs 6` and keep MLX as an opt-in |
| E4 fails | MLX is not accepted for research runs. Diagnose from E0–E2 (the bug is in the step) before any further use |

**Scope of any null.** One machine (M5 Pro, 48 GB, macOS 27.0.1), MLX 0.29.3, Python 3.9.6, rules-v2 demos of 2.06M rows, the shapes in A0–A4. It reopens on a new MLX or Python, or on a much larger demo set.

## 8. Phases

Each phase ends with its verification passing and a commit on the implementation branch. Don't merge to `main` (the orchestrator merges).

### Phase 0: environment
1. Confirm that the user has authorized implementation.
2. `.venv/bin/python -m pip install mlx==0.29.3` (user-approved per D1).
3. Check: `.venv/bin/python -c "import mlx.core as mx; print(mx.__version__, mx.default_device())"` prints `0.29.3 Device(gpu, 0)`, and `.venv/bin/python -c "import numpy; print(numpy.__version__)"` is still `2.0.2`.
4. If pip tries to change numpy, stop and ask the user.

### Phase 1: extract `bc_common.py` (numpy output byte-identical)
1. Before editing, record reference outputs with the current `bc_train.py`, called directly so that no Node is involved. Use `runs/exp-bc-e2e/demos.bin` (regenerate it with `bc-gen` per `research/prereg-bc-pipeline-e2e.md` if it is gitignored and missing). Save each `--out` JSON's sha256 for these cases:
   - V1.1: `--hidden 32,32 --epochs 2 --seed 1`;
   - the same with `--snapshot-dir`;
   - `--quant-bits 4` with `row`, `layer` and `pow2` (`--epochs 1`);
   - `--teacher-logits` from the shipped clone (written by `bc.writeLogits` via a 5-line node snippet), soft τ=2 α=0.5 and hard (`--epochs 1`);
   - `--transition-weight 3` on duck demos `runs/exp-bc-duck/v2-demos-duck-s41-60` (`--epochs 1`).

   That directory's `demos.bin` is gitignored and was absent on 2026-10-10. Regenerate it with `node trainer/cli.js bc-gen --dir runs/exp-bc-duck/v2-demos-duck-s41-60 --teacher duck --seeds 41-60 --overwrite` and check that its sha256 equals the committed `demos.json`'s `814c0686…`. The same applies to E4's held-out rows. `runs/exp-bc-e2e/demos.bin` was present (385,713,312 bytes).
2. Move the host-side helpers (§4.1) into `trainer/bc_common.py`. `bc_train.py` keeps its CLI and step and calls `main()`.
3. V1.2: every reference sha is identical after the refactor. `node --test test/` passes.
4. V1.3: `VECLIB_MAXIMUM_THREADS=1` gives the same sha for V1.1. This decides the byte-identity claim for `--jobs`.

### Phase 2: `bc_train_mlx.py`
1. Implement §4.2–4.3 for N ≥ 1, with every feature flag.
2. Add `test/bc-mlx-check.py`. It is not part of `node --test`, because it needs MLX. It runs E0, E1, E2 and E3 and prints a pass/fail table. Add `test/bc-mlx.test.js`, which skips with a message when `import mlx.core` fails and otherwise runs E0 only, which takes seconds.
3. Gate: E0–E2 pass. E3 is recorded.

### Phase 3: Node integration and dashboard
1. Generalize `trainPython` (§4.6), add `bc-train --backend mlx`, extract `finishBcRun` and `bcStartLine`, and add `bc-train-batch` with both backends.
2. Extend `test/dashboard-check.js` (§4.6) and fix the dashboard hint string.
3. Update the `trainer/cli.js` header and `AGENTS.md`.
4. Verify:
   - `node --test test/` passes;
   - `node test/dashboard-check.js` passes;
   - `node test/smoke-trainer.js` passes (unaffected paths);
   - a 2-epoch `bc-train-batch --backend mlx --shapes 8,8/6 --seeds 1-2 --epochs 2` into `runs/exp-gpu-smoke/` produces 4 run directories, each with `best.json`, `bc-metrics.jsonl` (2 rows) and `train.log`, all listed on the running dashboard while they train and after they finish;
   - the same with `--backend python --jobs 2`, whose per-directory params sha equals a plain `bc-train --backend python` for that seed (if V1.3 held);
   - E5 passes;
   - SIGINT during the smoke batch releases every lock (no `train.lock` left).

### Phase 4: benchmark
Run §7 exactly. Write `research/gpu-trainer-results.md`: the measurements, the readings applied verbatim, the deviations, and the E3/E4 results. Record the chosen default in `AGENTS.md`'s decision log.

**Acceptance for the whole plan.**
- Phases 0–3 verifications pass.
- §7 is run and read.
- The numpy backend still reproduces the shipped clone.
- The dashboard lists every new run layout.
- The core hash is unchanged.
- The docs are updated.

**Non-goals.** GPU evaluation or simulation; ES on the GPU; changes to `core.js`, `policy.js`, the node backend or the numpy numerics; padding across shapes; upgrading the `.venv` Python; a new run-directory layout.

## 9. Open choices for the user

1. **Scheduling rule.** Should the orchestrator treat "one GPU training job plus one CPU-heavy job" as allowed at the same time (needed for the overlap gain, §6 point 2), pending H-overlap in §7?
2. **Default backend for `bc-train-batch`.** Pre-registered to be decided by §7. The user may prefer to fix `python --jobs` as the default regardless, to avoid the MLX dependency in the default path.
