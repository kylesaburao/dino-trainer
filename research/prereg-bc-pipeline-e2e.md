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
