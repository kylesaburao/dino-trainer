# Pre-registration: behaviour-cloning sweep over depth × width: model efficiency and training effort

Status: **registered.** Author: mac-3, 2026-10-10, on main `4808925`. It supersedes `research/prereg-bc-shapes.md`. That study's Step 0 checks, its gate and its certification rung are carried over in §6. Its extra 44-4-4-3 shape is dropped, because the user set the grid.

**User's request (2026-10-10):** "training done on models of different configurations of layer count and layer sizes with a report on the efficiency of the model, the amount of effort required to train the model. Ideally the evaluation between models is time-independent so they can be trained in parallel. Operate layer counts [1,64], layer sizes [8, 64] (in powers of 2)."

The user then chose powers of 2 on both axes, trained by behaviour cloning. An earlier answer the same day made a **smaller / faster model** the goal of clone research.

## 1. Decisions
- **(S1) Report:** for each configuration, how efficient is the model (inference cost, size, reliability), and how much effort does it take to train? This is a deliverable for the user. Every number in it must be time-independent (§4).
- **(S2) Ship candidate:** is there a configuration smaller than the shipped 44-32-32-3 that is as reliable? If so, it goes to the user with evidence (§6). The user decides whether to ship it.
- **(S3) Depth:** up to what depth does a plain tanh MLP clone train at the shipped hyperparameters? Out of scope: residual connections and normalisation layers. The model format is `dino-mlp-v1`, a plain MLP, and any change there is a separate decision.

## 2. Grid
- **Depth** (hidden layers) d ∈ {1, 2, 4, 8, 16, 32, 64}.
- **Width** w ∈ {8, 16, 32, 64}, with every hidden layer the same width.
- That gives 28 configurations: `44-[w×d]-3`.
- The shipped model is d = 2, w = 32.
- MACs per decision = 44w + (d−1)w² + 3w. Parameters = MACs + d·w + 3.
  - For example: d1 w8 = 376 MACs; d2 w32 = 2,528; d64 w64 = 261,056.
  - Exact values come from the script.

## 3. Hypotheses
- **H-width:** reliability saturates at a small width. Probe B's 8-8 nets reached 1M on all their games. Prediction: w8 and w16 at d1–d2 pass the game bars.
- **H-depth:** with N(0, 1/fan_in) init, tanh and lr 2e-3, deep plain nets train worse: signals and gradients shrink with depth. Prediction: at some d ≥ 16 the held-out agreement and the game pass rates fall, and the drop starts earlier at small w.
- **H-null:** every configuration is equivalent on all game bars. Then effort and inference cost alone rank them.

## 4. Time-independent measurements
Every primary metric is a count or a deterministic function of (configuration, training seed, fixed data, fixed game seeds). None of them depends on wall-clock time or machine load. **That is why training and evaluation jobs may run in parallel.**

**Model efficiency (inference):**
- parameters;
- MACs per decision;
- `model.js` bytes;
- `model.css` bytes, when the CSS generator handles the shape (Step 0).

**Training effort:**
- training FLOPs = 6 × MACs × (samples processed). The 6 is 2 for the forward pass and 4 for the backward pass, a standard approximation; state it as one.
- samples processed = steps × batch;
- **effort-to-criterion:** the first epoch, and the FLOPs up to it, at which the snapshot's **held-out demonstration agreement** reaches T1 and T2 (below).

**Held-out demonstration agreement:**
- The data comes from `bc-gen` over **seeds 41–60** with the default levels and spans: 20 seeds, disjoint from the training demos' 1–40.
- The metric is the weighted fraction of frames where argmax(model) equals the solver's label. Also report recall for "none" and "jump" separately.
- Compute it on each epoch's snapshot (§5). A numpy evaluator is fine, but spot-check it against `policy.js` argmax on a few thousand rows per configuration.

**Game reliability (fixed seed sets):**
- **Tier A:** validation, 25 full v2 games to 1M (bc-train's own validation).
- **Tier B:** the 1,000-game scan (§5).

**Effort-to-criterion is measured within one 15-epoch cosine schedule, not as a minimal budget.** A snapshot at epoch 4 is a model mid-schedule, at high learning rate.

**Thresholds T1 and T2:** T1 = 99.5% and T2 = 99.8% held-out agreement.
- **Positive control for the thresholds:** the shipped configuration (d2 w32, seed 1) must reach both within its 15 epochs.
- If it doesn't, rebase T1 and T2 to its agreement at epoch 4 and at epoch 15. That fallback is fixed now, so the thresholds can't be tuned after seeing results.

**Secondary (time-dependent; labelled as such):**
- training CPU-seconds, as process CPU time, not wall time;
- JS decisions per second from `bench`, run single-threaded, alone, after all training finishes;
- the CSS per-decision recalc time (Step 0, a few shapes only).

These are reported but never used to rank configurations or to make decisions.

## 5. Procedure
- **Step 0 (premises):**
  1. Check that `bc-train --backend python`, `export --out <scratch>` and `trainer/css-model.js` handle d = 1 and d = 64 at w = 8 on tiny demos, and that `css-nn.js` evaluates the CSS output to the same decisions as `policy.js`. Report anything shape-specific; don't patch the CSS code here.
  2. Measure the CSS recalc time per decision for d2 w32 against d2 w8 (`prereg-bc-shapes.md` §3.2 method).
  3. Time one epoch of d64 w64 to size the parallelism.
- **Snapshots** (a trainer change: implemented, reviewed and merged on its own branch **before** the sweep runs from main): add an opt-in `--snapshots` flag to `bc-train --backend python`. It writes each epoch's params to `<dir>/snapshots/epoch-NN.json`.
  - Off by default.
  - The Node backend rejects it with a clear error.
  - It must not change training: the d2 w32 seed-1 params with `--snapshots` must equal `runs/exp-bc-e2e/py-s1`'s params exactly. **That is the harness positive control.**
  - `node --test test/` must pass, with a test that the flag leaves the params unchanged.
- **Training:**
  - all 28 configurations with seed 1;
  - `--demos runs/exp-bc-e2e` (the validated `bc-gen` output);
  - the Python backend, 15 epochs, batch 1024, lr 2e-3 (the shipped hyperparameters);
  - `OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2` for every job, so results don't depend on the thread count;
  - run dirs `runs/exp-bc-sweep/d<d>-w<w>-s<seed>`.

  - Pass `--wasm` to every `bc-train`, so its validation uses the wasm path.
  - Run small configurations in parallel with `--workers 4` for their validation. Train the heavy ones (w64 at d ≥ 8, w32 at d ≥ 16, w16 at d64) serially and last, with the full pool.

  Run several jobs in parallel, at most `floor(cores / 2)` at a time, because the measurements are time-independent. Run each job's tier-A validation with few workers, or after training. CPU-seconds will be inflated under load, which is why they are secondary.
- **Tier B (scan):** 1,000 full v2 games to 1M on **seeds 30,003,000–30,003,999** (`research/scripts/prereg-bc-robustness/scan.js --wasm`). Scan every configuration that:
  - passes tier A at 25/25;
  - **and** has ≤ 2× the shipped model's MACs (≤ 5,056).

  A full game costs about MACs × 1.94M, so bigger nets are scanned only on request; each would take hours. Train and scan seed 2 **only for the S2 candidates**: configurations in tier B with fewer MACs than d2 w32 (< 2,528), i.e. w8 at all depths, w16 at d ≤ 4 and w32 at d1. Only the ship decision needs the replicate. For the other configurations, the tier-B figure is one seed and is labelled that way.
  - **Tier-B positive control:** the shipped configuration's seed-1 model from this sweep (equal to `py-s1`) scores 0/1,000 here.
- **CSS sizes:** generate `model.css` for every configuration to record its bytes. d64 w64 is about 10 MB. Load it in a browser only for the Step 0 shapes (d1 and d64 at w8, plus d2 w32 and d2 w8 for timing).
- **Report:** a standalone `research/bc-sweep-report.md`. It opens with the time-independence statement, and this file's "Results" section points to it. Contents:
  - one table with every metric in §4 per configuration;
  - plots or tables of agreement against FLOPs per configuration (learning curves on a FLOP axis);
  - the Pareto frontier of reliability against MACs per decision, and of reliability against training FLOPs to T2.

## 6. Readings (fixed)
| Result | Reading |
|---|---|
| The snapshot control's params differ from `py-s1` | STOP: `--snapshots` changes training. Fix it before any sweep result is read |
| d2 w32 s1 fails any tier-B game | STOP: the harness or the seed range is suspect |
| A configuration fails tier A, or doesn't reach T1 within 15 epochs | **Untrainable at the shipped hyperparameters** (scoped: plain tanh MLP, this init, 15 epochs, lr 2e-3, seed 1). Not "too small" or "too deep" in general. Reopen with a depth- or width-scaled schedule as a separately registered step |
| Tier A passes at d ≥ 16 but agreement or tier B degrades with depth | H-depth supported at that depth, at these hyperparameters |
| **S2:** some configurations with fewer MACs than d2 w32 have 0/1,000 on both seeds | **The smallest-MACs one is the ship candidate.** Break ties by the larger min top-2 logit margin over 600k frames of its own play. Then apply `prereg-bc-shapes.md`: its gate (a speed gain, or ≤ half the `model.css` bytes); its certification on held-out offsets 325–474, paired with the shipped clone (the next fresh offset is then 475); its CSS agreement check on a scratch page; and its readings |
| A configuration passes tier B but never reaches T2 | Demonstration agreement is not the reliability bottleneck at that size. Report its effort-to-T1 only. This is not a failure |
| No smaller configuration passes tier B on both seeds | The shipped shape stays. Record the smallest passing shapes and their failures |

**Power and scope.** 0/1,000 bounds a model's per-game failure rate below 0.3% (one-sided 95%), and 0/2,000 over two seeds below 0.15%. Tier A (25 games) bounds it only below about 11%. Large nets that pass only tier A are reported as "tier A only".

Scope: v2, the 44 features, the demos from seeds 1–40 with default `bc-gen` settings, the Python backend and the shipped hyperparameters. Effort is measured at one learning-rate schedule. A configuration that needs a different schedule will look more expensive than it is.

## 7. Cost estimate (to be replaced by Step 0's timing)
- **Training:** the total training FLOPs over the grid are dominated by w64 at d32–64. At 15 epochs × 2.1M rows, d64 w64 costs about 6 × 261k × 31.4M ≈ 4.9e13 FLOPs. That is about 10–20 min per job at numpy speed with 2 threads. The other 27 configurations together cost roughly as much again. With parallel jobs, about 30–60 min.
- **Tier A:** games are about MACs × 1.94M multiply-adds. d64 w64 is about 5e11 per game, so 25 games on 18 workers take minutes to tens of minutes. The largest few dominate.
- **Tier B:** if every capped configuration passes tier A, that is 16 configurations at seed 1 plus 11 at seed 2 plus the control, about 28 scans. At 3–5 min each (d64 w8 is about 1.7× the shipped cost per frame), **about 1.5–2.5 h**.

## Results (PARTIAL: stopped by the user, 2026-10-10 about 19:30 UTC; mac-3 executor, branch `wt/bc-sweep` from main `9992a55`)
The full results are in **`research/bc-sweep-report.md`**. In summary:
- **Controls.** Both STOP controls passed: the `--snapshots` params are identical to `py-s1`, and d2 w32 s1 scored 0/1,000 on tier B. T1 and T2 stay at 99.5% and 99.8%.
- **Training.** All 28 configurations were trained at seed 1, and the 9 S2 candidates at seed 2.
- **Tier B, seed 1.** All 14 eligible configurations that passed tier A were scanned. All scored 0/1,000 except d8 w8, at 2/1,000.
- **Untrainable at the shipped hyperparameters (§6):** d ≥ 32 at w8, w16 and w32, and d ≥ 16 at w64. These nets collapse to the constant "none" predictor.
- **Not reached:** S2. Seed-2 scans were done only for d1 w8 and d2 w8, both 0/1,000. There were no gate or certification steps.
- **No held-out offsets were spent; the next fresh offset is still 325.**
