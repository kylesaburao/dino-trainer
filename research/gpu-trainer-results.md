# GPU trainer: implementation gates and §7 benchmark results

Results for `research/plan-gpu-trainer.md` (the plan; read it first). Branch `wt/gpu-batch` from `main` 24419c0, 2026-10-11, session mac-6. Machine: Apple M5 Pro (6 performance + 12 efficiency cores, 20-core GPU), 48 GB, macOS 27.0.1. The rules-v3 runner had finished (00:30:39Z) and the CPU was otherwise idle during A0–A3 and A5, except for the ≈1-minute MLX checks (E1–E3) noted below.

## 1. What was built

| Piece | File | Status |
|---|---|---|
| Host-side helpers shared by every python engine | `trainer/bc_common.py` | done; numpy output byte-identical |
| `bc-train` split into shared stages | `trainer/cli.js` (`prepareBcTrain`, `bcStartLine`, `bcTeacherLogits`, `bcEpochWriter`, `trainBcModel`, `finishBcRun`) | done; `bc-train` output identical |
| `bc-train-batch --backend python --jobs K` | `trainer/cli.js` | done |
| MLX engine (N stacked models of one shape) | `trainer/bc_train_mlx.py`, `trainer/bc.js` `trainMlx`, `mlxPythonPath` | done |
| `bc-train --backend mlx`, `bc-train-batch --backend mlx` | `trainer/cli.js` | done |
| Dashboard: batch members (queued, training, finished), `mlx` backend | `trainer/dashboard.js`, `dashboard.html`, `test/dashboard-check.js` | done, 49/49 |
| Checks | `test/bc.test.js` (2 new tests), `test/bc-mlx-check.py` (E0), `test/rng-streams.py` (V0.2) | done |

Phase order followed the user's choice: parallel numpy first, then MLX.

## 2. Environment (Phase 0)

The user approved this on 2026-10-11 via AskUserQuestion ("Yes, 3.14.8 + fallback ok"). PyPI was re-checked that day: mlx 0.32.3 is still the newest, with wheels for cp310–cp314 and none for 3.15.

- `pyenv install 3.14.8`, then `.venv-mlx` at the repo root with `mlx==0.32.3 numpy==2.5.4`. The global pyenv version is still `system`, and `.venv` is untouched (Python 3.9.6, numpy 2.0.2).
- **V0.1:** `.venv-mlx` prints `3.14.8 0.32.3 Device(gpu, 0) 2.5.4`.
- **V0.2, RNG streams:** `test/rng-streams.py` under `.venv` and under `.venv-mlx` produced identical sha256s for every draw: seeds 1, 2 and 7; layers 44-32-32-3 and 44-6-3; the init, `permutation(2,062,427)` twice, and `choice(…, 200000)` twice. No fallback was needed.
- **V0.3:** `flat_params` and `quant_extras` (row, layer, pow2 at 8 and 3 bits) give byte-identical output under both interpreters.

## 3. Correctness gates

### Phase 1: `bc_common.py` extraction

The pre-change `bc_train.py` (from 24419c0) and the new one were run on 300k-row slices of the jump demos (`runs/exp-bc-e2e`) and the duck demos. These cases were compared:
- plain 2×32 (2 epochs);
- snapshots;
- QAT row, layer and pow2;
- soft distillation (τ 2, α 0.5) and hard distillation, with the shipped jump clone's logits;
- transition weight 3;
- extra demos.

The result JSON, the snapshots and stdout (wall `seconds` excluded) have the same sha256 in all 9 cases. Two reference runs are identical to each other, and `VECLIB_MAXIMUM_THREADS=1` gives the same bytes (V1.3).

### `bc-train` after the stage split

The 24419c0 tree was run against this one in 5 cases:
- node;
- python;
- python with snapshots and pow2 QAT;
- python with soft distillation;
- python with transition weight and extra demos.

`best.json` (with `trainSeconds` and the ROOT-relative paths normalised), `bc-metrics.jsonl`, the snapshots and the log lines are identical. `node test/bit-identity.js` against 24419c0 reports IDENTICAL: 7 scenarios, 48,629 values.

### `bc-train-batch --backend python`

Each member's params, validation, bc-metrics and snapshots equal those of `bc-train --backend python` with the same shape and seed. This holds on the tiny demos (`test/bc.test.js`) and **on the full 2.06M-row demos over 15 epochs**: arms A1 and A1b are byte-identical to A0 for 10/10 seeds each.

The failure paths were exercised:
- A member that fails gets no best.json, a "failed" log line and an empty bc-metrics, so it stays listed. The others finish, and the command exits 1.
- SIGINT exits 130 with every lock released and no temp files left.

### MLX engine, §5 tiers

The first E0 run failed every mode, with relative gradient error ≈ 2e-3 against numpy float32's 1e-6. A probe isolated the cause: **MLX's GPU float32 matmul defaults to TF32**. It shows 9e-4 relative error against float64, while MLX on the CPU and numpy are at 3e-7. Elementwise ops (tanh, exp, log) are fine. `MLX_ENABLE_TF32=0` gives 2.5e-7. The engine now sets it before importing mlx, and `config.engine.tf32` records `false`.

| Tier | Result | Bar |
|---|---|---|
| E0, one step, 10 modes: plain 2×32 and 16×8, soft τ2 α.5, soft τ1 α0, hard, QAT int4 row/layer/pow2, int8 pow2 2×32, duck + transition weight 3 (10 weighted rows in the batch) | **10/10 PASS**. Max relative gradient error 0.79–1.17e-6; numpy float32's own error is 0.80–1.41e-6. Loss rel error ≤ 1.2e-7 | ≤ 1e-4, loss ≤ 1e-5 |
| E1, numpy against MLX N=1, 300k-row slice, 2×32, seed 1 | 1 epoch: max \|ΔW\|/max\|W\| 2.2e-7, loss Δ 3e-8. 3 epochs: 7.9e-7, loss Δ ≤ 4e-8; recall identical | loss Δ ≤ 1e-3, params ≤ 1e-3 |
| E2, model 0 of an N=3 stack against N=1, same seed | 1.8e-7; not bitwise equal | as E1 |
| E3, the same MLX run twice | **bitwise equal** (recorded, not gated) | — |
| E4 (from A0 and A2, below) | see §4 | see §4 |
| E5 | `bc-train` and `bc-train-batch --backend mlx` with snapshots: run dirs, `config.engine`, `config.batchRun`; params within 1e-4 of numpy and the same validation (`test/bc.test.js`). Distillation and QAT on MLX via E0 and the batch smoke run (quantRecord accepted pow2 `k`) | — |

## 4. §7 benchmark

Common setup: 2×32, 15 epochs, `runs/exp-bc-e2e` jump demos (2,096,268 rows, rules v2, pre-v3 core hash, so `--force` as AGENTS.md allows), tier A to 1M on the 25 validation seeds with `--wasm`, and the default 18-worker pool. Outputs are in `runs/exp-gpu-bench/` (gitignored). Wall times are from the scripts' `date` stamps; `trainSeconds` comes from each best.json.

The training span runs from the first "training started" to the last "epoch 14" line in the members' train.logs. In A1 it includes the first wave's tier A, which overlaps the second wave.

| Arm | What | Wall | Training | Tier A |
|---|---|---|---|---|
| A0 | 10 × `bc-train --backend python`, sequential | **201 s** | 145.8 s in total (14.6 s/model) | 25/25 × 10 |
| A1 | `bc-train-batch --backend python --jobs 6` | **91 s** (2.2×) | span 72.4 s (2.0×), two waves; 22.5 s/model in contention (max 27.7) | 25/25 × 10 |
| A1b | the same with `--jobs 10` | 107 s (1.9×) | span **32.1 s** (4.5×), one wave; 31.9 s/model; then all 10 tier A runs come after, none overlapped | 25/25 × 10 |
| A2 | `bc-train-batch --backend mlx`, 10 seeds | **84 s** (2.4×) | **20.9 s for all 10** (1.39 s/epoch; 7.0× A0's training) | 25/25 × 10 |
| A3 | `bc-train-batch --backend mlx`, 40 seeds | 312 s | **58.8 s for all 40** (3.92 s/epoch; ≈ 9.9× against 40 × 14.6 s) | 25/25 × 40 |
| A5 | `opening.js` (3,000 games to 2,000; one thread), solo against during MLX training of 40 × 2×32 | solo 38.03 / 38.46 s; during 39.46 s (fully overlapped) and 40.39 s (partly) | MLX 52.4 s for 15 epochs under that load | — |

E4, behavioural equivalence (A0 numpy against A2 MLX, seeds 1–10):
- **Tier A:** 25/25 on every seed in both arms.
- **Final train loss:** same-seed differences are |Δ| ≤ 1.6e-7, against a numpy seed-to-seed SD of 7.8e-4.
- **Decision agreement:** measured on the held-out jump demos (`runs/exp-bc-sweep/heldout-demos`, `bc-gen` seeds 41–60, 1,048,674 rows), in float64 with policy.js argmax.
  - MLX s_k against numpy s_k: ≥ 0.999999 on every seed. One row in about 10.5M differs.
  - The noise floor, the minimum pairwise agreement among the 10 numpy models (45 pairs), is 0.998537 (median 0.999171).
  - **PASS.**
- **Tier B:** `scan.js --base 30002000 --n 1000 --target 1000000 --wasm --expect-hash 65f3546c` on A2 s1 and s2: **0/1,000 and 0/1,000** (222 s and 211 s, seeds contiguous). That is the same bar as the e2e precedent.

### Readings

These apply plan §7's table verbatim.
- **H-gpu holds.** N=10 trained in 20.9 s (≤ 30 s predicted), N=40 in 58.8 s (≤ 60 s).
- **H-cpu holds, end to end.** A1 (91 s) is within 1.5× of A2 (84 s), because both are dominated by tier A, not training. On training alone MLX is far ahead: 20.9 s for 10 models, against 22–28 s **per model** for the 6 parallel numpy processes. Those processes slowed 1.5× per model, not the ≈ 1.07× of Probe 2, because tier A's 18 workers run at the same time.
- **H-overlap holds** (1.03× ≤ 1.15×), within this scope: one single-threaded CPU screen. An 18-worker screen alongside MLX is untested. MLX's host side (permutations, the per-step dispatch) uses about one core.
- **H-host is not shown.** Per-model GPU time keeps falling from N=10 (0.139 s/epoch per model) to N=40 (0.098).
- **Table row "H-gpu holds, H-cpu holds"** reads: default `--backend mlx` only if H-overlap holds, which it does. So the pre-registered reading is to make **MLX the default for `bc-train-batch`**.
  - *Not applied in code:* the default is still `python`.
  - On this machine the two backends differ by 8 % end to end for a 10-seed stage (91 s against 84 s). That is within the 5–10 % wall noise, so the stage time doesn't decide it. Training alone does favour MLX: 20.9 s against a 32.1 s span at its best.
  - The deciding factor is portability. An MLX default fails on a machine without `.venv-mlx` (the WSL desktops). That is the user's decision (§6).
- **E4 passes**, so MLX is accepted as equivalent for research runs.

### Deviations

- **A4** (mixed shapes in one batch) was not run. The grouping and per-group overlap code path is covered only by the tiny mixed-shape tests (8x2/6).
- **Tier A wall times** include the 18-worker pool's warm-up for each model.
- The ≈1-minute MLX E1–E3 checks ran while A0 trained, which may have slowed A0 by a few seconds. A0's per-model training times (14.4–15.1 s) match earlier measurements (14.3 s).

## 5. Where the time goes now

Training is no longer the bottleneck in a `bc-train` stage. For 10 seeds, A2 spends about 21 s training and about 60 s on tier A. For 40 seeds, A3 spends about 59 s training and about 250 s on tier A. Tier A runs one model at a time on the pool: 25 games on 18 workers is two waves with 11 workers idle in the second.

**The next cheap win** is to pack several models' tier A into one pool run. 250 games on 18 workers is about 14 full waves instead of 10 half-empty pairs, an estimated 30–40 % less tier-A time. That is CPU-side and independent of the GPU work. Not done here.

As the plan said, the whole-arm speedup stays bounded by the CPU screens (opening, tier B, duck share). The GPU's added value is that training no longer competes for the CPU: H-overlap holds.

## 6. Open for the user or orchestrator

1. **Default backend for `bc-train-batch`.** The pre-registered reading says `mlx`, and the code default is `python`. The options are to switch it, or to switch only when `.venv-mlx` exists.
2. **Tier A packing** (§5): whether to implement it as a follow-up.
3. **A4 and an 18-worker overlap test,** if mixed-shape sweeps or GPU training next to tier B scans are planned.
