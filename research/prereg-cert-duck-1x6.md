# Pre-registration: certification of a rules-v2 1×6 duck clone, paired with the shipped `v2-bc-duck-d2w8-s1`

Status: **registered (pending orchestrator review), 2026-10-10; not run.** Author: a mac subagent, branch `wt/prereg-v3-clones` from main `63dcfa1` (core hash `65f3546c…`). The user chose "Certify 1×6 later". The template is `research/prereg-teacher-duck-cert.md`. The evidence is distillation Stage 0 (`research/prereg-distill.md` Results, on branch `wt/exp-distill2`, not yet on main).

## 1. Decision
Is a rules-v2 1×6 duck clone (44-6-3, `teacher-duck`) **at least as reliable as the shipped `v2-bc-duck-d2w8-s1`** (44-8-8-3) on fresh held-out seeds to 1M? Does its `model.css` agree with its `model.js` through the duck phase?

This informs one decision only: whether the 1×6 clone is put to the user as a **replacement** for the shipped model. It would be the same behaviour (it ducks), with a smaller net: 291 params against 459, and 282 MACs per decision against 440, about 0.64× the CSS cost per decision. Shipping needs the user's approval and a decision-log entry. This step ships nothing and never commits the root `model.js`/`model.css`.

## 2. Candidate: selection rule (fixed now, before anything is read for this purpose)
**Rule:** among the 1×6 Stage-0 seeds that are **clean** (tier A 25/25, opening 1,000/1,000) **and** have **tier B 0/1,000**, take the one with the **largest min top-2 margin** (`margin.js`, 600k frames). Ties go to the lower seed.
- Eligible set: only s1 and s2 have tier B, because of Stage 0's cap of 2 tier-B scans per arm. s3–s10 are clean on tier A + the opening only, so they are not eligible.
- Read from `runs/exp-distill/stage0-summary.json` (branch `wt/exp-distill2`):

  | Seed | Tier A | Opening | Tier B | Min margin | Final loss |
  |---|---|---|---|---|---|
  | **s1** | 25/25 | 1,000/1,000 | 0/1,000 | **1.56e-4** | 0.01972 |
  | s2 | 25/25 | 1,000/1,000 | 0/1,000 | 8.6e-5 | 0.01975 |

- **Candidate: `runs/exp-distill/v2-bc-duck-d1w6-s1`.** Duck share 1.000 (Stage 0).
- **Margin note (CSS fidelity):** 1.56e-4 is above the 5.8e-5 flag but 3.6× below the shipped model's 5.61e-4 (`prereg-teacher-duck-cert.md` §2). That is why §4's CSS gate goes through the duck phase.

**Precondition: merge first.** The candidate's `best.json` exists only on `wt/exp-distill2`. The orchestrator merges that branch (or at least `runs/exp-distill/v2-bc-duck-d1w6-s1/`, `stage0-summary.json` and the margin file) to main before execution. The executor then checks:
- the params sha256 of the merged `best.json` (computed as in `prereg-teacher-duck-stuck.md` §4.2: sha256 of `JSON.stringify(b.params)`) equals **`eb89e31f5c5d02e5e11519f1f9966179c3bf073074673877030127052c73070e`**. That value was read at registration from `wt/exp-distill2` (file committed in `cfed47d`; whole-file sha256 `b9bf4181…`);
- `gameVersion` v2, `layers` [44, 6, 3], `config.teacherName` duck.

**Core hash.** Both the candidate and the shipped model carry `coreHash` `0f329d4e…` (trained before rules v3); main is `65f3546c…`. `assess` does not refuse this: `game-v3.md` §9.2 assessed the shipped model on the current core. It is justified for rules-v2 play by AGENTS.md ("Core hash", rules-v3 bullet): the v1 and v2 goldens, the regenerated demo sha256 values, `net.test.js` and `bit-identity.js --core-changed` prove that v2 games and decisions are unchanged. Recorded, not a STOP.

## 3. Arms (same seeds, paired)
| Arm | Run dir | Shape | Evidence so far (not held out, except the shipped model) |
|---|---|---|---|
| **candidate** | `runs/exp-distill/v2-bc-duck-d1w6-s1` | 44-6-3 | tier A 25/25, opening 1,000/1,000, tier B 0/1,000, duck share 1.000, min margin 1.56e-4 |
| **shipped (control)** | `runs/exp-bc-duck/v2-bc-duck-d2w8-s1` | 44-8-8-3 | 150/150 on offsets 325–474; tier B 0/1,000; min margin 5.61e-4 |

**Commands, one at a time** (the only heavy job on the Mac; order: candidate, then shipped):
```sh
node trainer/cli.js assess --dir runs/exp-distill/v2-bc-duck-d1w6-s1 --which best --target 1000000 --seed-offset 625 --seeds 150 --wasm
node trainer/cli.js assess --dir runs/exp-bc-duck/v2-bc-duck-d2w8-s1 --which best --target 1000000 --seed-offset 625 --seeds 150 --wasm
```
Both use rules v2, the models' own rules, so no `--game`. Each appends one record to `<run>/assess.jsonl`; nothing is hand-edited. The per-seed `scores` in the records give the pairs.

**Seeds.** Held-out indices 625–774 of `trainer/env.js` `evalSeeds`. They are disjoint from the demo seeds (1–60), the validation seeds and the tier-B seeds (30,003,000–30,003,999). The executor records the first and the last seed.

**Offsets.** **625–774 become spent; the next fresh offset is 775.**
- **Precondition:** scan every `assess.jsonl` and `evals.jsonl` under the main checkout and every worktree for a `seedOffset` + `seeds` range reaching 625 or above. If one exists, STOP and report.
- Known spent ranges: 0–24, 100–174 (ES), 175–324, 325–474, 475–624 (the rules-v3 cross-rules assessment).
- If the rules-v3 clone study (`prereg-v3-clones.md`) is certified later, it starts at 775 or above.

**Positive control.** The shipped arm: expected 150/150, as on 325–474. It also checks the harness on these seeds.

## 4. CSS gate (candidate only, and only if it passes 150/150)
`node test/css-agreement.js --dir runs/exp-distill/v2-bc-duck-d1w6-s1` with the defaults: `--min-ducks 50 --min-score 3000 --seed 12345 --speed 2048`, timeout 900 s.
- It exports the candidate into a scratch directory with the real export path, and never reads or writes the root `model.js`/`model.css`.
- It compares the CSS decision with the JS decision on every frame, through the pterodactyl and duck phase.
- **Bar (the same as the shipped model's export check, AGENTS.md decision log):** **0 disagreements**, ≥ 50 frames whose JS action is duck, and the play target (3,000 points) reached, i.e. exit 0. The shipped model's record: 0 disagreements over 25,446 frames, 421 of them duck frames.
- Recorded: frames, disagreements, duck frames, ducked mid pterodactyls, score, the smallest margin among compared frames, and the `model.css` byte size against the shipped `model.css`.

`test/browser-check.js` (root model only) is **not** run here. It belongs to the ship step, after export, with the `net.test.js` update.

## 5. Statistics (fixed now; as `prereg-teacher-duck-cert.md` §3)
- **Pass rate:** passed/150 with the two-sided 95 % Clopper–Pearson interval (`passCI95`, as `assess` writes it). For 0 failures the one-sided 95 % upper bound on the failure rate is **1.98 %** (two-sided upper end 2.43 %; pass-rate interval 0.9757–1.000).
- **Paired comparison:** exact McNemar, **one-sided**, H1 "the candidate fails more than the shipped model".
  - b = seeds where only the candidate fails; c = seeds where only the shipped model fails; p = P(X ≥ b), X ~ Binomial(b + c, ½). Concordant failures do not enter.
  - With c = 0, b = 5 is the smallest significant count (p = 0.031); b = 4 gives 0.0625; b = c = 0 gives p = 1.
  - The other direction (the candidate fails less) is reported with the same formula, for information.
- **Power** (probability of ≥ 1 / ≥ 5 failures in 150 games at a true per-game failure rate p): 0.5 %: 53 % / 0.1 %; 1 %: 78 % / 1.8 %; 2 %: 95 % / 18 %; 3 %: 99 % / 47 %; 5 %: 100 % / 87 %. So the 0-failure bar detects a 2 % failure rate with 95 % probability, but a significant McNemar result needs about 3–5 %.
- **Reliability bar:** **150/150.** One failure fails the bar even when it is not significant.
- **A tie is not a win.** 150 pairs cannot show equivalence. b = c = 0 excludes only failure rates above about 2 % per arm.

## 6. Readings (fixed in advance)
| Result | Reading | Next |
|---|---|---|
| Precondition fails (not merged, sha mismatch, offsets ≥ 625 already spent) | not executable as registered | STOP; report |
| **Both 150/150, CSS gate passes** | **Equal reliability** at 150 seeds (McNemar b = c = 0, p = 1; each failure rate ≤ 1.98 % one-sided). **A tie: a behaviour and size change only.** Same behaviour (both duck), smaller net (291 vs 459 params, 282 vs 440 MACs, about 0.64× CSS cost per decision), smaller min margin (1.56e-4 vs 5.61e-4) | **The user decides** whether to replace the shipped model. If yes, a separate ship step: export, `export-css`, `browser-check.js`, `net.test.js` pin update and a decision-log entry |
| Both 150/150, CSS gate fails (≥ 1 disagreement, < 50 duck frames, or target not reached) | the assessment stands; **the candidate is not shippable to the CSS page** as it is | Report the frames and margins. CSS fidelity work is a separate step |
| Candidate fails ≥ 1 seed, shipped 150/150 | **Reliability bar not met.** b ≥ 5: significantly worse (p ≤ 0.031); b = 1–4: not significant, but the bar fails | Not a replacement. Report the failure scores. A diagnosis (`diag.js --teacher duck`) is a separate step |
| Shipped fails ≥ 1 seed | **The first held-out failure of the shipped model**, which bears on its own decision-log claim. The general McNemar form applies (c > 0) | Report to the user first |
| Both fail the same seed(s) | concordant failures; they do not enter McNemar | Report the seeds; check them against the solver before reading anything |
| Candidate 150/150 and shipped fails, with McNemar c ≥ 5 | the candidate is significantly more reliable | The user decides; recorded as a reliability gain, not just a size change |

## 7. Scope, and what this does not do
- Rules v2, target 1M, full games from 0, `--wasm` (bit-identical to JS), held-out offsets 625–774, this one candidate seed. The other 1×6 seeds and 1×4 (also PASS in Stage 0, margins 3.3e-4 and 1.6e-4) are not certified here.
- No change to `core.js`, `policy.js`, `test/solver.js` or `trainer/` between registration and results.
- Nothing is shipped. The root `model.js`/`model.css` are never committed. No `session`, `finish` or `--auto-export`.
- A null (a tie) is conditional on 150 seeds. It reopens with more fresh offsets (from 775).
- **Cost:** two `assess` runs of 150 full v2 games. The shipped 2×8 took 12.4 s on 325–474; the 1×6 should take a similar or shorter time. The CSS gate takes about 1–15 min (headless Chrome, until 50 ducks and 3,000 points). Under 20 min of Mac time in all.
- **Commits:** both `assess.jsonl` (`git add -f` under `runs/exp-*`), the CSS-agreement JSON summary, and the results appended here with the code commit and the core hash.

## 8. Open choices for the orchestrator
- **O1, the selection rule:** the largest margin among tier-B-clean seeds (proposed; picks s1). The alternative, the lowest clean seed, also picks s1.
- **O2, 1×4 instead of, or alongside, 1×6:** 1×4 is smaller (195 params, 188 MACs) and its s1 margin (3.3e-4) is larger than 1×6-s1's. The user named 1×6, so 1×4 is not registered.
- **O3, timing:** before or after the rules-v3 clone study. They share no seeds or offsets, but both need the Mac.
