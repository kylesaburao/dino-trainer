# Pre-registration: certification of the rules-v3 2×8 clone `v3-bc-duck-d2w8-s2`

Status: **registered 2026-10-12, before any assessment ran.** Main at `5fbe42d`, core hash `65f3546c…`. Author: the Mac orchestrator. The user ordered it the same day: "Certify then ship v3 2×8", so that the game pages can always play the latest rules, which are v3.

## 1. Decision
Is `runs/exp-v3-clones/v3-bc-duck-d2w8-s2` (44-8-8-3, `teacher-duck`, trained for rules v3) reliable to 1M on rules v3 on fresh held-out seeds, and does its `model.css` agree with its `model.js` on rules v3? This informs **whether it is put to the user as the ship candidate** that replaces the rules-v2 `model.js`/`model.css`. Shipping needs the user's approval and a decision-log entry, and this registration ships nothing.

## 2. Evidence so far (not held out; `research/prereg-v3-clones.md` Results)
- Tier A 25/25, opening screen 1,000/1,000 and post-step screen 1,000/1,000, tier B 0/1,000 failures.
- Duck share 1.000, min top-2 margin 8.1e-4: the largest of the three candidates, and above the shipped v2 2×8's 5.6e-4.
- **Baseline (P2 there):** v2 models on rules v3 score 0/150; every game crashes between 100,016 and 101,555.

## 3. Arms and seeds
| Arm | Command | Prediction |
|---|---|---|
| **Candidate** | `node trainer/cli.js assess --dir runs/exp-v3-clones/v3-bc-duck-d2w8-s2 --which best --target 1000000 --seed-offset 775 --seeds 150` | 150/150 |
| **Detection control** (the shipped v2 model on rules v3, a cross-rules assessment) | `node trainer/cli.js assess --model model.js --game v3 --target 1000000 --seed-offset 775 --seeds 150` | 0/150, every crash in 100,000–102,000. Otherwise the harness does not reach the step as in P2: stop and investigate before reading the candidate |

- Both arms run on the wasm kernel (the default, bit-identical to policy.js), on the Mac, the reference machine, one at a time.
- The candidate's record is appended to its `assess.jsonl`. The control uses `--model`, so it writes no record; its output is quoted here.
- **Seeds:** held-out indices 775–924 (`evalSeeds(925).slice(775)`). Premise check: a scan of every `assess.jsonl`/`evals.jsonl` under `runs/`, `research/` and the worktrees found the highest spent range ending at 775 (offsets 625–774, `prereg-cert-duck-1x6.md`). **This spends 775–924; the next fresh offset is 925.**

## 4. Statistics (fixed now)
- **Reliability bar:** 150/150. Report the two-sided 95% Clopper–Pearson interval, and for 0 failures the one-sided 95% upper bound on the failure rate, 1.98%.
- **Power:** at a true failure rate of 2%, P(≥ 1 failure in 150) = 95%.
- **Comparison with the baseline:** descriptive. 150 vs 0 needs no test.

## 5. CSS gate (only if 150/150)
- `node test/css-agreement.js --dir runs/exp-v3-clones/v3-bc-duck-d2w8-s2 --start-score 98000`, which crosses the step at 100k. **Pass:** 0 CSS/JS disagreements, the script's duck and score targets reached, exit 0.
- Report the frames compared, the duck frames and the smallest margin seen.

## 6. Reading
- **150/150 and the CSS gate passes:** a ship candidate. Ask the user for approval. If approved: `export --dir … --which best` (root `model.js` + `model.css`), move `shipped: true` in `BUNDLED_MODELS`, run `export-models`, `test/browser-check.js` and the test suite. Then merge together with the page change that forces the latest rules.
- **Any failure:** not a candidate. Report the failing scores and class them as step (100–102k), late (above 130k) or other. The next step would be certifying `v3-bc-duck-d2w32-s5` on offset 925, the user's choice.
- **CSS disagreement:** not a candidate as is. Report the frames; reopen with the margin analysis.

## 7. Results (2026-10-12, Mac, main `001cf90`; both arms and the gate as registered)
- **Detection control** (v2 `model.js` on rules v3, offsets 775–924): **0/150**. Median 100,150; every crash between 100,016 and 101,747, all inside 100,000–102,000, as predicted. The harness reaches the step.
- **Candidate** `v3-bc-duck-d2w8-s2`: **150/150** to 1M on rules v3, 95% CP [0.9757, 1.0000], one-sided 95% upper bound on the failure rate 1.98%. It took 6.6 s on the 18-worker pool with wasm. Record: `runs/exp-v3-clones/v3-bc-duck-d2w8-s2/assess.jsonl`.
- **CSS gate** (`test/css-agreement.js --start-score 98000`): **PASS**.
  - 11,854 decisions compared over 98,000 → 107,874 (one game, through the step), with **0 disagreements** and 0 argmax mismatches.
  - Agreement per class: 10,386 none, 1,225 jump, 243 duck.
  - Mid pterodactyls: 54 ducked, 0 jumped, 0 crashed.
  - Smallest top-2 margin 0.0126.
  - Scope: the gate covers the step and the 10k after it, not the late game to 1M, which `assess` covers in JS/wasm.
- **Reading (§6): a ship candidate.** Shipping awaits the user's approval.
- Offsets 775–924 are spent; the next fresh offset is 925.
