# Pre-registration: does dense late exposure transfer down to the second wall? (Run B′ vs hygiene)

**This experiment does not test the register row "curriculum targeting the 400–800-point zone". That row stays OPEN whatever the result.** B′ never starts an episode in that zone (§1). A curriculum that really does (natural-ramp starts below 2000) is being scoped by desktop-2 and will get its own pre-registration once this one reads out.

Status: **amended per mac-2's review (option (a)); runs gated on the reproduction check (§4.6).** Author: desktop-1, 2026-10-10, on main `322fcaf`.
Process: retrospective `handoff/2026-10-10T1635Z-retrospective.md` §4 and §6.

## 1. Decision this informs

**New register row registered by this experiment: "Dense late exposure transfers down to the second wall (Run B′ flags as merged)".** The existing row "curriculum targeting the 400–800-point zone" stays OPEN regardless of the outcome.

The decision: does the merged gated curriculum (Run B′, transfer handoff §7.3) become the default pure-ES recipe for getting an escaped policy through the second wall, or is this design pruned?

Possible register outcomes:
- **promising, confirmation pending.** B′ clears stuck θs that hygiene doesn't (§5 reading A). That leads to a confirmation run (§4.5), not straight to adoption.
- **pruned (this design).** No large effect detected (reading B), or B′ is worse (reading C).
- **inconclusive.** Reading D.

**Naming caveat, found in the premise check.** As merged, B′ does **not** put any training episodes in the 400–800-point zone:
- Late starts must be 0 or ≥ ~2000 (`trainer/env.js`: below that, speed is still ramping and a late start wouldn't match natural play), and `--start-min` defaults to 2000.
- With `--start-dist points`, late starts are uniform on [2000, 1M), so they fall almost entirely in the max-difficulty late game.
- The 400–800 zone is reached only by start-0 episodes. With `--episodes 8 --start-zero-frac 0.5` there are 4 per candidate, the **same count as the hygiene arm's 4 from-0 episodes**.
- So B′ can affect the wall only through (i) the extra late windows, which contain the same pterodactyl types at higher density (§2), and (ii) reshaped fitness (progress to a 50k cap, geomean).

A curriculum that really starts episodes inside 400–800 needs a code change (natural-ramp starts below 2000) and is **not** tested here.

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

### 2.1 Further premises (checked before the batch; results appended in §8)
1. **The positive control reproduces on main with `--wasm`** (§4.6).
2. **`--select count`, `--fold-validation` and `--stop-rule allpass:59` don't move θ.** These are left out of both arms (§4.2). Check: a 5-generation resume of frozen s12, with and without those flags; θ must be bit-identical (checkpoint `theta` arrays compared element-wise).
3. **Validation cadence (`--eval-every`, `--val-seeds`) doesn't move θ.** This is covered by check 1: the reference resume used `--eval-every 50`, the reproduction uses 10.

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
- **Primary reading: the paired time-to-event ratio per θ, R = B′ frames / hyg frames.** Censoring is handled as bounds:
  - both clear: R is exact;
  - B′ clears, hyg censored: R ≤ F_B′ / F_hyg(H), an upper bound;
  - hyg clears, B′ censored: R ≥ F_B′(H) / F_hyg, a lower bound;
  - both censored: R unknown, read as "no difference within H".
  - **Promising at n = 4:** R ≤ 1/4 (B′ at least ~4× fewer frames, using the bound when censored) on ≥ 3 of 4 stuck θs, or B′ clears where hyg is censored on ≥ 3 of 4. Smaller ratios can't be told apart from chaos at n = 4 (§4.5).
  - **Harmful:** R ≥ 4 on ≥ 3 of 4 stuck θs, by the same rule.
  - The count readings in §5 are the coarse layer. The horizon is 1.4–2.3× the budget under which these θs were labelled "stuck", so both arms may clear, and then the ratio decides.
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

### 4.6 Launch gate: reproduction check (replaces the zero-generation resume)
- Resume frozen `ckpt-s12.json` in the **hyg** configuration on main with `--wasm --eval-every 10 --val-seeds 12`, for 100 generations (gen 1500 → 1600).
- It must replay the earlier ee6995c resume `runs/exp-surv-32x32-s12` **exactly**: fitMean, fitMax and thetaNorm to 1e-9 for every generation 1501–1600 (`tools/diff-metrics.js`), including the crossing of fitMean 2000 at gen 1565.
- **If it doesn't reproduce: STOP and report.** The positive control would then be invalid.
- This run doubles as the hyg-s12 arm only if it also reaches the validation event. Otherwise the hyg-s12 arm is rerun in full.

## 5. Readings (fixed before the run)

| Outcome | Reading | Register action / next step |
|---|---|---|
| **A** — B′ event on ≥ 3 of 4 stuck θs with R ≤ 1/4 (or hyg censored) on each of them, AND both B′ positive controls reach the event within H, AND no B′ opening loss | Supports H2 (look at the type split: if s9 clears too, H3 alone is excluded) | "promising, confirmation pending" → run the §4.5 confirmation |
| **B** — B′ ≤ 1 of 4 stuck θs and hyg ≤ 1 of 4; hyg positive controls reach the event; no B′ harm | H1 (or H3 with no transfer); no effect > ~4× detected | Prune **this design (B′ flags)**, scope = 32,32, these 6 θs, H 2000, ee6995c-equivalent code with `--wasm`. Row stays OPEN for a true 400–800-start curriculum (code change) |
| **C** — any B′ opening loss (valMedian < 200 at two consecutive validations after +100 gens on any θ), OR a B′ positive control fails to reach the event within H, OR hyg beats B′ on ≥ 3 of 4 stuck θs | H4 | Prune B′ as **harmful in this scope**. Reopen if a curriculum keeps the opening (e.g. lexicographic ranks, archived HANDOFF §9 fix 4) |
| **E** — both arms clear on ≥ 3 of 4 stuck θs | "Stuck" was a budget artefact; the counts don't discriminate | Read the per-θ ratios R: promising if R ≤ 1/4 on ≥ 3 of 4, harmful if R ≥ 4 on ≥ 3 of 4, otherwise no effect > ~4× detected (prune this design, as in B) |
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

## 8. Premise and launch-gate results

1. **Reproduction gate: PASS.**
   - Run: frozen `ckpt-s12.json` (sha256 checked) resumed in the hyg configuration on main `f39302c`, `--wasm --eval-every 10 --val-seeds 12`, gens 1501–1600, in `runs/exp-bp-repro-s12`.
   - fitMean, fitMax, thetaNorm and tiedPairs match the ee6995c resume `runs/exp-surv-32x32-s12` to 1e-9 on all 100 generations (0 mismatches). fitMean first exceeds 2000 at gen 1565, as before.
   - This also confirms premise 3 (validation cadence 50 → 10 doesn't move θ) and that `--wasm` is bit-identical on this path.
   - Validation medians: 1510–1590 between 695 and 1360, then 88,532 at gen 1600. **The two-reading validation event isn't reached by gen 1600** (the confirming reading would come at 1610), so per §4.6 the hyg-s12 arm is rerun in full in the batch.
   - Cost note: once past the wall, a generation costs about 4.8 s (17.7M frames). This is why the watcher stops runs at the event.
2. **Selection/validation flags don't move θ: PASS.** Frozen `ckpt-s9.json` resumed for 5 generations (1017 → 1022) with `--wasm`, with and without `--select count --fold-validation --stop-rule allpass:59 --val-seeds 12`. θ (0/2595 elements differ), Adam m and the RNG state are bit-identical.

**Deviation 1 (confirmed by mac-3).** The trainer's default stop rule (`median`) ends a run when a validation median reaches `--target` (1M) and the held-out median on stop-check seeds 0–24 does too. That happened before the watcher's second reading on s12-cur, s4-cur and s9-cur.
- Treatment, applied identically in both arms: the **event** is the first validation with valMedian ≥ 2000, and the held-out stop counts only as its **confirmation**.
- Marked * in the table. No reruns with `allpass`, and no code changes.

## 9. Results (batch finished 18:50Z, main `b6c170e` code, `--wasm`, desktop host)

Tools: `tools/run-bp.sh` (batch), `tools/event-watch.js` (SIGINT at the confirmed event) and `tools/bp-table.js` (this table). Run directories are `runs/exp-bp-s{4,5,7,8,9,12}-{hyg,cur}` (gitignored).

(* = confirmed by the trainer's held-out stop-rule check instead of a second validation)
| θ | Role | G0 | hyg: to event | cur (B′): to event | R = cur/hyg frames | hyg tied / min val after +100 | cur tied / min val after +100 / window |
|---|---|---|---|---|---|---|---|
| s12 | control | 1500 | 100 gens, 522M fr, 114 s | 40* gens, 239M fr, 42 s | 0.46 | 0.00 / 1000000 | 0.00 / - / win 25000→16667→11111→7407→5000 |
| s4 | control | 1500 | 500 gens, 539M fr, 94 s | 560* gens, 3643M fr, 654 s | 6.76 | 0.00 / 477 | 0.00 / 477 / win 25000→16667→11111→7407→5000 |
| s5 | stuck | 1435 | >2000 gens, 4414M fr, 708 s | 855 gens, 5248M fr, 842 s | ≤ 1.19 | 0.12 / 1151.5 | 0.00 / 1023.5 / win 25000→16667→11111→7407→5000 |
| s7 | stuck | 1500 | 1100 gens, 2234M fr, 360 s | 210 gens, 988M fr, 168 s | 0.44 | 0.02 / 518 | 0.00 / 495 / win 25000→16667→11111→7407→5000 |
| s8 | stuck | 1500 | >2000 gens, 3876M fr, 624 s | 1340 gens, 8170M fr, 1300 s | ≤ 2.11 | 0.04 / 223 | 0.01 / 205.5 / win 25000→16667→11111→7407→5000 |
| s9 | stuck | 1017 | >2000 gens, 3052M fr, 486 s | 33* gens, 219M fr, 37 s | ≤ 0.07 | 0.91 / 580 | 0.19 / - / win 25000→16667→11111→7407→5000 |

Notes:
- "Frames" is training frames from the resume point to the event, or to censoring. The hygiene arms on s5, s8 and s9 reached the +2000-generation horizon without the event.
- Last validation medians of the censored hygiene arms: s5 1297, s8 1421 (best 1622), s9 580 (unchanged since gen 900, with 59/64 pairs tied at the end).
- No opening loss in any run: the lowest valMedian after +100 generations was ≥ 205 everywhere.
- The B′ window sat at its 5000 floor after about 20 generations, as §2 predicted.
- The 30-minute safety cap never bound. The longest run was s8-cur at 1300 s.

### 9.1 Readings, applied as pre-registered

**Counts (coarse layer):**
- B′ reaches the event on **4 of 4** stuck θs (s5 +855, s7 +210, s8 +1340, s9 +33 generations).
- Hygiene reaches it on **1 of 4** (s7 +1100).
- Both positive controls reach the event in both arms within H.
- No B′ opening loss.

**Reading A** (B′ event on ≥ 3 of 4 stuck θs with R ≤ 1/4 or hygiene censored, controls within H, no opening loss): **met.**
- s9 qualifies with R ≤ 0.07. s5 and s8 qualify because hygiene was censored. s7 doesn't qualify (R = 0.44).
- **Register: B′ as specified → promising, confirmation pending** (§4.5). The attribution to late exposure (the row's name) is untested; see Hypotheses below.
- Reading E doesn't apply: hygiene cleared only 1 of 4 stuck θs.

**Limitation: the primary metric (frames).** This doesn't change the verdict above; the confirmation round fixes it.
- The horizon was set in **generations**. B′ generations cost more frames: about 2–6× more after the wall starts to give, because the late windows lengthen as the policy improves.
- On s5 and s8, B′ used **more frames to clear (5.2B, 8.2B) than hygiene had spent when it was censored (4.4B, 3.9B)**. Their ratios are only bounded by R ≤ 1.19 and R ≤ 2.11. So on s5 and s8 B′ is shown to clear in generations, **not** shown to be cheaper in compute.
- On the frames ratio alone, only **s9 (R ≤ 0.07)** is a decisive win. s7 (0.44) and s12 (0.46) favour B′ within chaos, and the **s4 control goes against it (R = 6.76)**.

**Hypotheses:**
- B′ changes three things at once: the late windows, 8 episodes per candidate instead of 4, and progress/geomean fitness. This design can't separate them.
- **H1** (curriculum irrelevant) is contradicted **in generations** (4/4 vs 1/4 under paired starts). It is **not excluded per unit of compute** on type M (s5, s8).
- **H3 alone** (tie-breaking without transfer) is excluded by s9. Ties fell from 0.91 to 0.19, but s9 also **cleared the wall in 33 generations** after 1000 + 2000 hygiene generations stuck at 580.
- **H4** (opening loss) is not seen.
- **H2** (dense late exposure transfers) is **not established.** On type M (s5, s7 and s8 had about 0 ties), less noisy fitness from 8 episodes, or the geomean aggregation, explains the gains as well as late exposure does. Only s9 points specifically at the late windows (its from-0 ties were broken by late-window variation), and that is n = 1.

### 9.2 Recommended confirmation (§4.5), amended by this result
- Use about 10 more escaped-and-stuck θs (hygiene survey seeds 13–30, `--wasm`), paired as here.
- **Censor both arms on a matched frames budget**, not on generations, so the primary metric compares equal compute.
  - Size it at ≥ 1.25× the largest B′ spend seen here (s8-cur 8.2B), so **≥ ~10B training frames**.
  - At the hygiene arm's measured 3.9–4.4B frames per 2000 stuck generations (about 10–12 min), a hygiene arm censored at 10B takes about 25–30 min on this host. Raise `--time-budget` to 45.
- **Add an ablation arm, hyg + `--episodes 8`** (no curriculum), to separate the episode count from the late windows and the fitness reshaping. If it matches B′ on type M, the gain there isn't late exposure.
- Report the split by failure type (T vs M) using `tools/wall-diag.js` on each start θ.

## 10. Confirmation pre-registration (approved by mac-3 with amendments; written before any confirmation-arm run)

Code: main `b1fbab8` plus this commit, with `--wasm`. The core hash is unchanged from the §9 runs (`0f329d4e3493`). The question has narrowed after §9: **does B′'s gain come from its late windows and fitness reshaping, or only from playing 8 episodes per candidate?**

### 10.1 Start θs
- **Source:** hygiene survey seeds 13–30 (`tools/run-survey.sh 13 30`: mid + decoupled 0.01, pop 128, σ 0.1, lr 0.03, 4 episodes, 32,32, `--generations 1500 --time-budget 5 --eval-every 50 --wasm`). These were started before this section was written: mac-3 approved generating them immediately. Their selection rule below is fixed before any arm runs.
- **Qualifies as stuck when**, at the run's last generation (1500, or earlier if the budget cut it):
  - it has escaped (10 consecutive generations with fitMean > 200);
  - the mean fitMean over the last 50 generations is < 1100;
  - fitMean never exceeded 2000;
  - the last validation median is < 2000.
  - Its start θ is that final checkpoint, frozen into `runs/exp-bp2-start/ckpt-sN.json` with `SHA256SUMS`.
- **Type**, from `tools/wall-diag.js` on the start θ (50 seeds from 900000) plus the run's last 100 generations:
  - **T (tie-bound, s9-like):** ≥ 60% of deaths are `ptero-low` with the dino on the ground, AND mean tiedPairs/64 over the last 100 generations ≥ 0.5.
  - **M:** everything else. A θ that meets only one of the two T conditions is reported as "T-partial" and counted as M.
- **Choosing n = 6:** all type-T θs first (up to 3), then type M by ascending seed, up to 6 in total. The **extension set** is the next 4 by the same rule (more T first if any remain). If fewer than 10 θs qualify, survey seeds 31+ are run with the same script before the extension.
- **If fewer than 2 type-T θs qualify,** the late-exposure question for the tie-bound mode can't be reached in this round. The type-T reading is then "not reached", not "null".

### 10.2 Arms (paired on each start θ; resumed from the same frozen checkpoint)
1. **abl** (the ablation, mandatory): hyg + `--episodes 8`, no curriculum.
2. **cur:** B′, with exactly the §4.2 flags (`--window 25000 --episodes 8 --start-dist points --start-zero-frac 0.5 --start-zero-cap 50000 --window-adapt --fitness-agg geomean`).
3. **hyg** (optional, secondary): plain hygiene with 4 episodes. Run only after arms 1–2 have finished on all θs, if the host is free.
4. **ramp** (optional): hyg + ep8 + desktop-2's natural-ramp flags. Included only if desktop-2's exact flags arrive before the θ classification is done; otherwise this round runs without it.

Common to every arm: `--wasm --eval-every 10 --val-seeds 12 --time-budget 45 --generations G0+20000`. The generation cap is deliberately non-binding; the frame budget below censors.

Order: positive controls first (s12, s4, frozen in `runs/exp-bp-start`), then the 6 θs in ascending seed, each θ abl then cur.

### 10.3 Event and censoring
- **Event** as in §4.3 and deviation 1: the first validation with valMedian ≥ 2000, confirmed by the next validation or by the trainer's held-out stop.
- **Frame-matched censoring:** each arm stops at **10B training frames since the resume point** (`tools/event-watch.js --max-frames 1e10`). That is 1.25× the largest B′ spend in §9 (s8-cur, 8.2B).
  - A first reading ≥ 2000 whose confirmation comes after the budget counts as censored.
  - If the 45-minute cap binds before 10B frames, that arm is "censored at budget" and reported separately. It shouldn't happen: about 6M training frames/s gives about 28 min.

### 10.4 Primary analysis (fixed in advance)
- **Per θ: R = cur frames to event / abl frames to event.** Bounds under censoring as in §4.3. Both censored → R undetermined, which counts as **not** a B′ win.
- **A B′ win on a θ:** R < 1, including "cur clears, abl censored".
- **Stage 1 (n = 6):** paired sign test on B′ wins, one-sided.
  - **6/6 wins → promote** (p = 0.016).
  - **4/6 or 5/6 → extend** with 4 more θs (n = 10). Promote only if the total is **≥ 9/10 wins** and **R ≤ 0.5 on ≥ 2/3 of all θs**. A stage-1 5/6 needs all 4 extension θs to win.
  - **≤ 3/6 → stop, not promoted.**
  - Overall one-sided α of the two-stage rule: P(6/6) + P(5/6)·P(4/4) = 1/64 + (6/64)(1/16) ≈ 0.022.
- **Promotion criterion (share of θs with R ≤ threshold):** besides the sign-test count, **R ≤ 0.5 (B′ at least 2× cheaper) on ≥ 2/3 of the θs.** A 6/6 sign-test result with fewer than 4 θs at R ≤ 0.5 is reported as "B′ consistently but only marginally cheaper", and not promoted.
- **Ablation-wins reading:** abl wins (R > 1, or abl clears and cur is censored) on ≥ 5/6 → the §9 gain is attributed to the episode count, and B′'s late windows and fitness reshaping are pruned in this scope.

### 10.5 Controls
s12 and s4 (frozen `runs/exp-bp-start`) run in abl and cur only. They are read **only** as "both arms clear within the 10B budget", never as ratio evidence (in §9, s4 went 6.76× against B′). If either arm fails to clear on a control, the round is **invalid** for that arm.

### 10.6 Readings split by type
- Every reading above is reported separately for type T and type M, as well as pooled. The sign test is computed on the pooled set.
- **Type T:** the late-exposure question (does B′'s late-window variation break the ties the ablation keeps) is answered only if ≥ 2 type-T θs are in the set:
  - if cur clears and abl stays tied (mean tiedPairs/64 ≥ 0.5) on both → late windows matter for the tie-bound mode;
  - if abl also clears → the episode count is enough.
- **Type M:** the pooled sign test is the evidence. A split such as "T wins, M ties" is reported as such, not averaged away.

### 10.7 Cost
- **Per censored arm:** about 10B frames at about 6M/s ≈ 28 min. Arms that clear stop sooner.
- **Primary arms:** 6 θs × 2 arms ≲ 5.6 h, plus controls (about 0.5 h).
- **Optional arms:** plain hyg adds ≲ 2.8 h; a ramp arm adds ≲ 2.8 h.
- **Extension:** 4 θs × 2 arms ≲ 3.7 h.
- One trainer at a time on the desktop host. desktop-2 is told at the start and end.

### 10.8 Amendment (mac-3, before any round-2 arm run): attribution test on known θs

**Survey finding** (hygiene, `--wasm`, seeds 13–30, gen cap 1500; `tools/classify-stuck.js`):
- Only **s15** qualifies as stuck (type M; escaped at 1488, median diag death 396).
- 4 seeds never escaped (s16, s18, s19, s20).
- 13 seeds passed fitMean 2000 at gens 188–1345. Their runs were stopped there, since that disqualifies them under §10.1. Their centre θs are genuinely past the wall: wall-diag medians up to about 20k, mostly dying to high pteros at speed 15–17.
- A fresh seed-9 run on main `b1fbab8` with `--wasm` matches the ee6995c survey run to 1e-9 for 600 generations, so this isn't a code change.
- **Pooled over hygiene seeds 4–30:** about 15 of 27 passed the wall by 1500, 7 were stuck after escaping (s4, 5, 7, 8, 9, 12, 15; about 7/22 of escapers), and 5 never escaped. **The second wall is a minority outcome under hygiene.** That lowers the value of a fresh-θ confirmation; mac-3 decides on it after this round.

**Reframed: this round is an ATTRIBUTION test on known θs, not a confirmation.**
- **θs (n = 5):** s5, s7, s8, s9 (frozen in `runs/exp-bp-start`) and s15 (`runs/exp-bp2-start`). Type T: s9 only, so **type T is "not reached"** (n = 1). The two-stage sign test of §10.4 is suspended for this round.
- **abl** (hyg + ep8): new on all 5 θs, plus the controls s12 and s4 (abl only, read as "clears within 10B").
- **cur** (B′):
  - Reused from round 1 on s5, s8 and s9. Runs are deterministic from the frozen checkpoints with the same flags, and all three cleared under 10B frames (max 8.2B), so they are valid under the 10B budget.
  - **Rerun on s7 as a replication gate.** It must match round 1's `runs/exp-bp-s7-cur` metrics exactly (fitMean, fitMax, thetaNorm and tiedPairs to 1e-9 through the event), or STOP and report.
  - New run on s15.
  - Round-1 directories are linked as `runs/exp-bp2-sS-cur` for the table.
- **ramp:** desktop-2's arm (`research/prereg-es-ramp.md`), new on all 5, only if its gate (a) shows bit-identity. Paired with abl.
- **Order:** the cur-s7 replication gate first (about 3 min; it validates reusing the round-1 cur results, so it runs before anything that depends on it). Then abl on s5, s7, s8, s9, s15, s12, s4. Then cur on s15. Then ramp.

**Readings (cur vs abl, frames to event with 10B censoring, R = cur/abl):**
- cur beats abl in frames (R < 1, including "cur clears, abl censored") on **≥ 4/5** → late windows and/or fitness reshaping contribute beyond the episode count.
- abl ≤ cur (R ≥ 1, including "abl clears, cur censored") on **≥ 4/5** → the round-1 gain is attributed to the episode count, and B′'s late windows are pruned at this scope.
- Otherwise → unresolved.
- **The register row stays "B′ as specified: promising, confirmation pending" whatever the result.** Promotion needs fresh θs, which is out of scope this round.

### 10.9 Stopped by the user before completion (time); no reading applied

At 19:23Z on 2026-10-10 the user stopped the round-2 attribution batch: the project moves to the ducking solver (teacher-duck). The running trainer got SIGINT and saved its checkpoint. No further arm was started. **No reading in §10.8 (or in `research/prereg-es-ramp.md`) is applied.**

Arms that ran. This is **partial, and NOT read against the registered thresholds**:

| Run | Status | Generations | Training frames | Note |
|---|---|---|---|---|
| exp-bp2-s7-cur | complete (replication gate) | 1500 → 1710 event (confirmed 1720) | 988M | Bit-identical to round 1's exp-bp-s7-cur on all 220 gens (6 fields, 0 mismatches) |
| exp-bp2-s5-abl | **interrupted** (SIGINT) | 1435 → 1543 (108 gens) | ~0.4B | No event; checkpoint at gen 1543 kept |

Not run: abl on s7, s8, s9, s15, s12 and s4; cur on s15; ramp on all θs and controls. cur on s5, s8 and s9 is round 1's, linked as `runs/exp-bp2-sS-cur`.

**Register:** "B′ as specified: promising, confirmation pending; attribution round incomplete."

**To resume later:**
- Checkpoints and start θs stay in place: `runs/exp-bp-start`, `runs/exp-bp2-start` and `runs/exp-bp2-s5-abl` on the desktop host.
- `tools/run-bp2.sh` skips any directory that already has a checkpoint, so continue s5-abl by hand: `train --resume --dir runs/exp-bp2-s5-abl` with the same flags plus `event-watch --after 1435 --max-frames 1e10`. The frame budget counts from 1435.
- Then rerun the runner for the remaining arms.
