# Pre-registration (DRAFT): Run B′ curriculum vs hygiene at the second wall

Status: **draft for mac-2 review. Nothing has been run except the static premise probe in §2.** Author: desktop-1, 2026-10-10, on main `322fcaf`.
Process: retrospective `handoff/2026-10-10T1635Z-retrospective.md` §4 and §6.

## 1. Decision this informs

Register row (retrospective §6.4): **"Curriculum targeting the 400–800-point zone — OPEN — gated curriculum merged, untrained."**

The decision: does the merged gated curriculum (Run B′, transfer handoff §7.3) become the default pure-ES recipe for getting an escaped policy through the second wall, or is this design pruned?

Possible register outcomes:
- **promising, confirmation pending.** B′ clears stuck θs that hygiene doesn't (§5 reading A). That leads to a confirmation run (§4.5), not straight to adoption.
- **pruned (this design), row stays OPEN for a targeted variant.** No large effect detected (reading B), or B′ is worse (reading C).
- **inconclusive.** Reading D.

**Naming caveat, found in the premise check.** As merged, B′ does **not** put any training episodes in the 400–800-point zone:
- Late starts must be 0 or ≥ ~2000 (`trainer/env.js`: below that, speed is still ramping and a late start wouldn't match natural play), and `--start-min` defaults to 2000.
- With `--start-dist points`, late starts are uniform on [2000, 1M), so they fall almost entirely in the max-difficulty late game.
- The 400–800 zone is reached only by start-0 episodes. With `--episodes 8 --start-zero-frac 0.5` there are 4 per candidate, the **same count as the hygiene arm's 4 from-0 episodes**.
- So B′ can affect the wall only through (i) the extra late windows, which contain the same pterodactyl types at higher density (§2), and (ii) reshaped fitness (progress to a 50k cap, geomean).

A curriculum that really starts episodes inside 400–800 needs a code change (natural-ramp starts below 2000) and is **not** tested here. That is the row's reopen trigger if B′ is pruned.

## 2. Premise check (static probe, done; `research/scripts/prereg-es-curriculum/late-probe.js` (usage: `node <script> <checkpoint.json> <repo root>`), single-threaded, ~1 min)

Setup: the stuck centre θs (pre-resume checkpoints, now frozen as `runs/exp-bp-start/ckpt-s{9,7,5}.json`, from ee6995c; `core.js` and `policy.js` are unchanged since, so the core hash still matches). 40 episodes per condition. Late starts drawn as B′ draws them (uniform on [2000, 1M)).

| θ | Condition | Median progress | Killers (of 40) | Mean frames/ep | Pteros seen per 1000 frames (low/mid/high) |
|---|---|---|---|---|---|
| s9 | from 0, cap 50k | 0.011 | ptero-low 40 | 3097 | 0.32/0.44/0.36 |
| s9 | late, window 5000 | 0.025 | ptero-low 25, ptero-high 15 | 273 | 2.29/2.29/1.38 |
| s7 | from 0, cap 50k | 0.017 | ptero-mid 29, cactus 11 | 3847 | 0.59/0.70/0.70 |
| s7 | late, window 5000 | 0.032 | ptero-low 19, ptero-mid 19, other 2 | 339 | 2.73/2.29/1.92 |
| s5 | from 0, cap 50k | 0.024 | ptero-mid 20, cactus 17, ptero-high 3 | 5122 | 0.93/0.96/1.03 |
| s5 | late, window 5000 | 0.067 | ptero-mid 21, cactus 13, ptero-high 6 | 754 | 4.14/3.42/3.18 |

(Window 25000 gives the same killers and densities, with progress ÷5.)

What it establishes:
1. In late windows the stuck θs die to the **same pterodactyl types** that stop them from 0. They see those types at **4–7× the density per frame**, in episodes 5–11× cheaper. So the mechanism by which B′ could help (dense, cheap exposure to the failing skill) is plausible. Whether skill learned at late speeds (13–22) transfers to the wall's speeds (8.5–10) is the open question.
2. The window will sit at its 5000 floor for the whole test. Start-0 episodes capped at 50k are never passed by a stuck θ (progress ≈ 0.01–0.02), so the gate's start-0 pass rate stays below 30% and `--window-adapt` only shrinks.
3. For s9, late-window deaths vary (low vs high pteros, different distances), so B′'s fitness is **not** tied the way s9's from-0 fitness is (tiedPairs/64 ≈ 0.9 in the resumed hygiene run). B′ breaks s9's ties by construction. Whether the gradient that results points at the low-ptero skill is the question.
4. Fitness scale: geomean ranks candidates by Σ ln f_j. A from-0 death at 520 vs 1000 points is Δln ≈ 0.65. A late-window death after 3 vs 8 obstacles is a similar Δln. **The late windows carry roughly half the ranking signal**, and none of it comes from the 400–800 zone itself.

## 3. Competing hypotheses and predictions

Failure types, from the desktop-1 second-wall report (main `322fcaf`):
- **Type T (s9):** blind to low pteros (dies standing still, 50/50), with population ties (tiedPairs/64 ≈ 0.9).
- **Type M (s5, s7, s8):** jumps into mid pteros and lands on wide cactus clusters, with about 0 ties, and climbs slowly.

| Hypothesis | Mechanism | B′ vs hygiene on type M (s5, s7, s8) | B′ vs hygiene on type T (s9) | Positive controls (s4, s12) |
|---|---|---|---|---|
| H1 — **slow search, curriculum irrelevant** | The wall is just a slow ES climb on from-0 episodes. B′'s extra late-window signal is unrelated noise. | No faster in generations; **slower** in frames (extra episodes) | Censored in both arms | Both arms clear; B′ the same or slower |
| H2 — **transfer from dense late exposure** | Late windows train ducking under mid pteros and jumping low ones at high speed; this transfers down to speeds 8.5–10 | **B′ faster** (fewer frames to event) | B′ faster, or clears where hygiene is censored | Both clear; B′ no slower |
| H3 — **tie-breaking only** | B′ helps only by adding non-tied late-window signal; the from-0 gradient is unchanged | No difference (type M has no ties) | Ties fall in B′ (tiedPairs/64 well below 0.5), but **no faster to the event** unless H2 also holds | Unaffected |
| H4 — **harm / trade-off** (Run B's failure mode) | The geomean and late signal pull θ towards late-game behaviour at the cost of the opening | B′ **slower or loses ground**: valMedian falls below 200 at any validation after +100 gens | Same | B′ positive controls delayed > 2× or lost |

Which comparisons separate which:
- H1 vs H2: type M frames to event.
- H2 vs H3: whether s9's ties falling is accompanied by s9 clearing.
- H4: the opening-loss check on every B′ run.

## 4. Design

### 4.1 Paired θs, not seeds (the plateau lottery is taken out of the measurement)

Both arms resume from **the same pre-resume checkpoint** (θ, Adam state, RNG state, generation). The checkpoint is copied into a fresh run directory per arm.
- Start checkpoints are frozen in `runs/exp-bp-start/ckpt-sN.json` (gitignored, desktop host), with `SHA256SUMS`. The `runs/exp-surv-*` directories now hold *post*-resume checkpoints, so don't use them.
- Each run directory gets `cp runs/exp-bp-start/ckpt-sN.json runs/exp-bp-sN-<arm>/checkpoint.json` and a sha256 check before launch.
- **Launch check:** the checkpoints come from ee6995c and carry its config keys (`tieTarget`, `tieGain`, `sigmaMax`). On main, resume spreads `ck.config` and backfills new keys, and `st.window` takes `--window` when it is given on the CLI (`trainer/cli.js` ~281–293). That reads as fine, but before the batch, run a zero-generation resume (`--generations G0`) of one scratch copy per arm to confirm it starts. Needs mac-2's OK, since it is a `train` invocation.
- **Stuck θs (test set, n = 4):** s5 (gen 1435), s7 (1500), s8 (1500), s9 (1017). Each escaped, then stalled below about 1100 in ≥ 730 further hygiene generations.
- **Positive controls (n = 2):** s4 (gen 1500) and s12 (gen 1500). In the hygiene resume they reached fitMean > 2000 at +505 and +65 generations.
- Pairing is **of starting state only**. B′ draws 8 episodes per candidate against hygiene's 4, so the trainer RNG streams diverge from the first generation. This is not common-random-numbers pairing.

### 4.2 Arms (both `--wasm`, both with `--eval-every 10 --val-seeds 12`)

Validation doesn't affect the θ trajectory; it is only there to measure the event the same way in both arms.
- **hyg:**
  `train --resume --dir runs/exp-bp-sS-hyg --wasm --generations G0+2000 --time-budget 30 --eval-every 10 --val-seeds 12`
  The run's own config (mid + decoupled 0.01, pop 128, σ 0.1, lr 0.03, 4 episodes, no curriculum) is unchanged. For the first ~730–976 generations this replays the earlier `exp-surv` resumes, which doubles as a determinism check: diff with `tools/diff-metrics.js`.
- **cur (B′):**
  `train --resume --dir runs/exp-bp-sS-cur --wasm --generations G0+2000 --time-budget 30 --eval-every 10 --val-seeds 12 --window 25000 --episodes 8 --start-dist points --start-zero-frac 0.5 --start-zero-cap 50000 --window-adapt --fitness-agg geomean`
- Left out of both arms: `--select count --stop-rule allpass:59 --fold-validation`. They don't move θ. `allpass` would spend held-out seeds 0..58, and per-generation validation after the event is expensive. No `session`, `finish` or `--auto-export`.

### 4.3 Primary metric (time-to-event, censored)

- **Event:** **two consecutive validations** (every 10 generations on the centre θ, 12 validation seeds from score 0, capped at 1M) with **valMedian ≥ 2000**. The event is timed at the first of the pair. A single reading isn't enough, because the 12-seed median is noisy at the wall and s8 had a one-generation blip.
  - fitMean can't be compared across arms: the curriculum's fitness is progress in [0, 1].
  - A watcher (a small script to write in `tools/`) sends SIGINT to the run once the second confirming reading is logged. SIGINT saves a checkpoint and releases the lock, so post-wall generations, which are expensive, aren't paid for.
- **Primary outcome: training frames from the resume point to the event**, Σ `frames` over the logged generations.
  - It is deterministic, unaffected by host contention, and charges B′ for its extra episodes.
  - Censored at G0 + 2000 generations. The 30-minute session budget is a safety cap only and shouldn't bind (§6).
- **Secondary outcomes:**
  - generations to the event;
  - wall time (cumulativeMs) to the event;
  - mean tiedPairs/64 over the run;
  - the window trajectory (expected to sit at 5000);
  - **opening check:** the lowest valMedian after +100 generations (H4 if < 200).

### 4.4 Horizon (from the measured 200→2000 numbers)

- Hygiene from these θs: the positive controls reached the event at +65 and +505 generations (fitMean > 2000; the validation event may lag a little). The four stuck θs were censored at +730, +782, +976 and +1013 generations.
- **H = 2000 generations after the resume point.** That is 4× the slower positive control and about 2× the longest censored stretch already observed. A B′ effect that clears the wall in under about 4× the slower control's time is inside the window.

### 4.5 Sample size and what it can detect

- Measured spread of 200→2000 under hygiene: 207, 524, >867, >1049, >1087, >1442 generations. sd(ln T) is well above 0.7 once censoring is accounted for.
- For a 2× paired effect with sd ≈ 0.8 and between-arm correlation about 0.3 from the shared start, n ≈ 11 θs would be needed (one-sided α 0.05, power 0.8; about 15 two-sided). That isn't feasible here.
- The positive controls are single runs. With this much chaos, one run is more than 2× slower than its pair by luck about 20% of the time. So the controls are judged only on reaching the event within H, and how much slower they were is reported as a secondary result.
- **With n = 4 stuck θs this is a screen for large, qualitative effects**: clears vs doesn't clear within H, or about ≥ 4–5× in frames. A sign test on 4/4 gives p = 0.0625. A null means "no effect larger than ~4× detected in this scope", not "no effect".
- **Confirmation step if the screen is positive:**
  1. Generate about 10 more escaped-and-stuck θs: hygiene survey seeds 13–30 under `--wasm`, about 1–2 min each to gen 1500.
  2. Keep those that escape and stay below 1100.
  3. Rerun both arms on them.
  4. Adopt only if B′ wins the paired primary outcome on every θ, or by a sign test at α 0.05 (n ≥ 6 → 6/6).
  5. Then certify the resulting model against hygiene on fresh offsets **from 325**.

## 5. Readings (fixed before the run)

| Outcome | Reading | Register action / next step |
|---|---|---|
| **A** — B′ event on ≥ 3 of 4 stuck θs, each with fewer frames than its hyg pair (hyg censored counts as more), AND both B′ positive controls reach the event within H, AND no B′ opening loss | Supports H2 (look at the type split: if s9 clears too, H3 alone is excluded) | "promising, confirmation pending" → run the §4.5 confirmation |
| **B** — B′ ≤ 1 of 4 stuck θs and hyg ≤ 1 of 4; hyg positive controls reach the event; no B′ harm | H1 (or H3 with no transfer); no effect > ~4× detected | Prune **this design (B′ flags)**, scope = 32,32, these 6 θs, H 2000, ee6995c-equivalent code with `--wasm`. Row stays OPEN for a true 400–800-start curriculum (code change) |
| **C** — any B′ opening loss (valMedian < 200 at two consecutive validations after +100 gens on any θ), OR a B′ positive control fails to reach the event within H, OR hyg beats B′ on ≥ 3 of 4 stuck θs | H4 | Prune B′ as **harmful in this scope**. Reopen if a curriculum keeps the opening (e.g. lexicographic ranks, archived HANDOFF §9 fix 4) |
| **D** — anything else (e.g. 2 of 4 vs 1 of 4, or type-split results such as B′ clearing s9 but no type-M θ) | Inconclusive at n = 4 | Report the split by failure type. Decide with mac-2 whether the §4.5 confirmation is worth it |
| **Invalid** — a hyg positive control (s4 or s12) doesn't reach the event within H | The event definition or horizon is wrong; the test can't show success | Fix the event (e.g. val cap or threshold) and rerun the controls before reading anything |

## 6. Compute (desktop host, Ryzen 7 9700X, 16 workers, `--wasm`)

- **Per-generation cost while stuck**, from the hyg resumes without wasm: s9 about 0.37 s; type M (s5, s7, s8) about 0.36–0.49 s, which is 730–1013 generations in 6 min. desktop-2 measured a 1.65–1.70× speed-up on x86, so about 0.22–0.30 s with `--wasm`. Generations slow down as θ improves (longer episodes).
- **B′ adds** 4 late episodes of 270–830 frames each to the 4 from-0 episodes of 3,100–5,100 frames: about +10–25% frames per generation. So 2000 generations take about 10–12.5 min per run, under the 30-min safety cap.
- **Worst case, all censored:** 6 θs × 2 arms ≈ **100–110 min**. Expected is lower: the positive controls stop at their event, and so does any B′ success. Validation (12 seeds every 10 generations) costs under 1% before the wall.
- One trainer at a time. desktop-2 gets "start" and "batch done" messages, as before.

## 7. Scope and caveats recorded with the result

- Shape 32,32; v2; pop 128; σ 0.1; lr 0.03; mid ranks; decoupled wd 0.01.
- θs from survey seeds 4, 5, 7, 8, 9 and 12 (pre-resume, ee6995c); code at main with `--wasm`.
- The θs were **selected** for being stuck under hygiene, so regression to the mean affects both arms equally (paired). Results don't generalise to fresh-seed training.
- B′ also changes the episode count (8 vs 4). With from-0 exposure equal, the comparison is "B′ as specified" vs hygiene, not a pure curriculum effect.
