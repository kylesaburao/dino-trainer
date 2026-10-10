# Pre-registration: does the first-class BC pipeline, end to end, reproduce a passing clone?

Status: **registered.** Author: mac-3, 2026-10-10, on main `f040434`.

## 1. Decision
Is `bc-gen` → `bc-train` (both backends), as merged, a valid base for all further clone research (shapes, data size, and anything else)?

Every positive control so far trained on the *prototype's* demonstrations, converted:
- control (b): Python reproduces `m_base_s1` exactly;
- control (c): the Node clone reaches 25/25;
- the robustness scan: C1, C2 and N1 each 0/2,000.

`bc-gen`'s own output differs from those demonstrations in one deliberate way: the boring-frame counter resets every episode, so a few down-weighted rows differ (control (a) showed the weight-1 rows are byte-identical). No model has yet been trained on `bc-gen` output and played. Future clone research will use `bc-gen`, so this is a necessary-condition test.

## 2. Hypotheses
- **H0:** the counter change is immaterial. Clones trained on `bc-gen` demonstrations play like C1, C2 and N1.
- **H1:** something in the `bc-gen` path breaks training: row order, weights, manifest, or the hand-off to the backends. Prediction: a validation shortfall or scan failures.

## 3. Design
1. `node trainer/cli.js bc-gen --dir runs/exp-bc-e2e` with the defaults: seeds 1–40 and the default levels. Record the row counts and the sha256.
   - Compare row counts and class counts with the converted prototype demos (`/Users/kyle/Desktop/dino-game-bc/runs/exp-bc-repro/converted`, if still present). Expect equal weight-1 rows, and boring rows within ±1 per episode.
2. `bc-train --backend python --seed 1 --hidden 32,32` → `runs/exp-bc-e2e/py-s1`, with `PYTHON=/Users/kyle/Desktop/dino-game/.venv/bin/python`. Then `bc-train --backend node --seed 1 --hidden 32,32` → `runs/exp-bc-e2e/node-s1`. Both use `--demos runs/exp-bc-e2e` and `--wasm`.
3. Scan both clones on **seeds 30,002,000–30,002,999** (1,000 fresh non-held-out seeds, past the robustness scan's 30,000,000–30,001,999) with `research/scripts/prereg-bc-robustness/scan.js`: v2, full games to 1M.

- **Positive control:** the robustness scan's C1 under the same harness (0/2,000). The known-good clone is not re-run here.
- **Cost:** about 20 s for gen, about 15 s plus about 160 s for training, two validations of about 10 s, and 2 × about 3 min of scan. About 10 min in total.

## 4. Readings
| Result | Reading |
|---|---|
| Both validate 25/25 and scan 0/1,000 | **H0: the pipeline is validated** as the base for clone research (scope: 44-32-32-3, v2, the default gen settings, seed 1) |
| Any validation < 25/25, or ≥ 1 scan failure | **H1 candidate.** Stop clone research. Diagnose the failure (`diag.js`), and compare the demo files row by row before any new experiment |
| Python's per-epoch loss differs from `m_base_s1`'s log by more than about 1e-3 | Note it. Expected to be small: only the down-weighted rows differ |

One failure in 1,000 would **not** be statistically significant against C1/C2/N1's pooled 0/6,000 (Fisher one-sided p ≈ 0.14). The gate is deliberately strict anyway: a clone death is cheap to diagnose, and the robustness tooling's drift/fatal-frame split has never been tested on a real clone death (robustness results, deviation 2).

## Results (run 2026-10-10, mac-3 executor, branch `wt/bc-e2e` from main `da74010`)

Run in the §3 order: gen → Python train + validation → Node train + validation → scan Python clone → scan Node clone. Everything ran from the main checkout at `da74010` (core hash `0f329d4e3493…`), one job at a time, 18 workers. Raw files are in `runs/exp-bc-e2e/` (gitignored): `demos.{bin,json}`, `gen.log`, `py-s1/` and `node-s1/` (`best.json`, `bc-metrics.jsonl`), `py-s1.log`, `node-s1.log`, and the scans `py-s1.jsonl` and `node-s1.jsonl` with their logs. Scripts are in `research/scripts/prereg-bc-pipeline-e2e/`: `democounts.js` gives row/class counts by weight plus the sha256, and `analyze.js` gives the scan CP bounds, seed and frame checks, and the loss comparison. The scan runner is unchanged: `research/scripts/prereg-bc-robustness/scan.js --base 30002000 --n 1000 --target 1000000 --wasm`.

### Step 1: `bc-gen` (defaults: seeds 1–40, levels 0,2000,50000,300000,900000, span0 3000, span 5000, keep-boring 10, v2)

| Demos | Rows | Weight-1 none | Weight-1 jump | Boring (weight 10, none) | Weighted none | Duck | sha256 |
|---|---|---|---|---|---|---|---|
| `bc-gen` (`runs/exp-bc-e2e`) | 2,096,268 | 1,848,555 | 239,599 | 8,114 | 1,929,695 | 0 | `3a67bfd4e109d6bb…` |
| converted prototype (`dino-game-bc/runs/exp-bc-repro/converted`) | 2,096,181 | 1,848,555 | 239,599 | 8,027 | 1,928,825 | 0 | `8aa43c055b2d6273…` |
| Δ | +87 | 0 | 0 | +87 | +870 (= 87 × 10) | 0 | — |

- The weight-1 rows are equal in both classes, as expected.
- The boring rows differ by +87 over 200 episodes (40 seeds × 5 levels), so |Δ| ≤ 200 holds in aggregate. The direction is the expected one: with a reset per episode, each episode keeps its first boring frame.
- **Limitation:** the converted prototype file has no episode boundaries, so the "±1 per episode" expectation could only be checked in aggregate, not per episode.
- Solver crashes: 0. Wall time: 0.55 s (the §3 cost estimate was about 20 s).

### Step 2: `bc-train`, seed 1, 44-32-32-3, `--demos runs/exp-bc-e2e`, `--wasm`

| Backend | Epoch time (15 epochs) | Train time | Final loss (ep 14) | Final recall none / jump | Validation (25 seeds to 1M) | Validation wall | Job wall |
|---|---|---|---|---|---|---|---|
| Python (`.venv`) | 0.94–0.96 s (sum 14.1 s) | 14.6 s | 0.004369 | 0.9984 / 0.9983 | **25/25** (median = min = 1,000,000) | 4.4 s | 19 s |
| Node | 10.27–10.45 s (sum 155.8 s) | 155.9 s | 0.005291 | 0.9978 / 0.9975 | **25/25** (median = min = 1,000,000) | 4.6 s | 161 s |

Recall is train-set recall on a subsample, as `bc-train` reports it.

**Python per-epoch loss against `runs/exp-bc-clones/log_base_s1.txt` (`m_base_s1`).** The prototype log has 4 decimals, so the Python loss was rounded to 1e-4 before differencing.

| Epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m_base_s1 | .0583 | .0149 | .0125 | .0110 | .0109 | .0100 | .0090 | .0077 | .0070 | .0062 | .0058 | .0052 | .0048 | .0045 | .0044 |
| e2e Python | .0579 | .0148 | .0124 | .0117 | .0109 | .0096 | .0086 | .0077 | .0069 | .0062 | .0058 | .0051 | .0047 | .0045 | .0044 |
| \|Δ\| | 4e-4 | 1e-4 | 1e-4 | 7e-4 | 0 | 4e-4 | 4e-4 | 0 | 1e-4 | 0 | 0 | 1e-4 | 1e-4 | 0 | 0 |

The largest difference is **7e-4** (epoch 3), below the "about 1e-3" note threshold. The final loss is the same at 4 decimals: 0.0044.

### Step 3: scans, seeds 30,002,000–30,002,999 (v2, full games 0 → 1,000,000, `--wasm`)

| Clone | Failures / games | Two-sided 95% CP | One-sided 95% upper | Wall time |
|---|---|---|---|---|
| `py-s1` (Python backend) | **0 / 1,000** | 0 – 0.368% | 0.299% | 176 s |
| `node-s1` (Node backend) | **0 / 1,000** | 0 – 0.368% | 0.299% | 170 s |

- Both scan files cover exactly seeds 30,002,000..30,002,999: 1,000 seeds, contiguous.
- Each header records layers 44-32-32-3, v2, target 1,000,000, wasm, and core `0f329d4e3493`.
- Every game ran the full 1,943,237 frames, the same count as the robustness scan's passing games.
- `diag.js` was not run, because nothing failed.

### Readings (applied verbatim from §4)
- Both validate 25/25 and scan 0/1,000, so the first row applies: **"H0: the pipeline is validated as the base for clone research (scope: 44-32-32-3, v2, the default gen settings, seed 1)"**.
- The loss row ("Python's per-epoch loss differs from `m_base_s1`'s log by more than about 1e-3") does not apply. The largest |Δ| was 7e-4, small as expected, since only the down-weighted rows differ.
- The H1-candidate row does not apply.

### Deviations from the pre-registration
- None in steps, order, seeds, models or readings.
- The "±1 boring row per episode" expectation was checked only in aggregate (+87 over 200 episodes), because the converted prototype demos have no episode boundaries.
- The loss comparison is at the prototype log's 4-decimal precision.
- Wall times came in well under the §3 estimate: gen 0.55 s, against about 20 s. The whole run took about 9 min.
- `analyze.js` here is a new script. Its CP functions are copied from `prereg-bc-robustness/analyze.js`, which is hard-wired to the arm names P, C1, N1 and C2.
- No McNemar test was computed, because none was pre-registered.
- **This is not a certification:** the seeds are non-held-out.
