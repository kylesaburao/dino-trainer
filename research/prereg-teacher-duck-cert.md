# Pre-registration: certification of the duck-generation 2×8 clones (`d2w8-s1`, `d2w8-s2`), paired with the shipped jump clone

Status: **registered 2026-10-10, before any eval ran.** Author: a mac subagent, branch `wt/cert-duck-d2w8`, created from main `8e98dfa` (core hash `0f329d4e3493f794daf9cf2d013bd7927b51ff848efed768922a10b582c7410e`).
Continues `research/prereg-teacher-duck-step2.md` (its §10 "Shipping is out of scope" list, item 2: certification on fresh held-out offsets, paired with the shipped jump-generation clone). AGENTS.md "Research method" and the decision log's certification table are the template.

## 1. Decision
Is a duck-generation 2×8 clone (44-8-8-3, `teacher-duck`) **at least as reliable as the shipped jump clone** on fresh held-out seeds to 1M, and does its `model.css` agree with its `model.js`? This informs one decision only: **whether a duck clone is put to the user as a ship candidate.** Shipping itself needs the user's approval and a decision-log entry; this step ships nothing, and never writes the root `model.js`/`model.css` into a commit.

Arm choice (the orchestrator's): d2w8 (arm B of step 2), both training seeds. Step 2 named d1w8 the smallest reliable-and-ducking arm, but d1w8-s2 carries the "third class thins margins" flag (7.57e-5 < 0.5× its twin); d1w8 is not certified here.

## 2. Arms (all on the same seeds, so the comparison is paired)
| Arm | Run dir | Shape | Teacher | Step-2 evidence (not held out) |
|---|---|---|---|---|
| duck s1 | `runs/exp-bc-duck/d2w8-s1` | 44-8-8-3 | duck | tier A 25/25, tier B 0/1,000, duck share 1.000, min margin 5.61e-4 |
| duck s2 | `runs/exp-bc-duck/d2w8-s2` | 44-8-8-3 | duck | tier A 25/25, tier B 0/1,000, duck share 1.000, min margin 3.58e-4 |
| jump (control) | `runs/bc-v2-h32x32` (shipped) | 44-32-32-3 | jump | 150/150 on offsets 175–324, 25/25 on 0–24 |

All three `best.json` have `gameVersion` v2, target 1,000,000, and core hash `0f329d4e3493…`, equal to the current `coreHash()` (checked before registration).

**Command, one at a time** (the only heavy job on the Mac; the only other node processes are two dashboards):
```sh
node trainer/cli.js eval --dir <run> --which best --target 1000000 --seed-offset 325 --seeds 150 --wasm
```
Order: duck s1, duck s2, jump. `eval` appends one record to `<run>/evals.jsonl`; nothing is hand-edited.

**Seeds.** `trainer/env.js` `evalSeeds(475).slice(325)`: held-out indices 325–474, first seed 3,182,883,642, last 3,381,787,671. They are disjoint from the demo seeds (1–60), the validation seeds and the tier-B seeds (30,003,000–30,003,999).

**This spends held-out offsets 325–474**, and this certification is the only place they get spent. **The next fresh offset afterwards is 475.** Premise check: every `evals.jsonl` under the main checkout and every worktree was scanned before registration. The highest range spent anywhere is 175–324 (offsets 0, 25, 175 in files; ES 100–174 per the decision log). Nothing at ≥ 325.

**Positive control.** The jump arm is the control: it is expected to pass 150/150, as on 175–324. It also checks the harness on these seeds.

## 3. Statistics (fixed now)
- **Pass rate:** passed/150 with the **two-sided** 95 % Clopper–Pearson interval, as `eval` writes it (`passCI95`) and as the decision-log table reports it. For 0 failures we also state the **one-sided** 95 % upper bound on the failure rate, 1 − 0.05^(1/150) = **1.98 %** (two-sided upper end 2.43 %; pass-rate interval 0.9757–1.000).
- **Paired comparison:** exact McNemar, **one-sided**, as in the decision log, with H1 "the duck clone fails more than the jump clone". Each duck arm is tested separately against the jump arm. b = seeds where only the duck arm fails; c = seeds where only the jump arm fails; p = P(X ≥ b), X ~ Binomial(b + c, ½). Seeds where both fail are concordant and do not enter the test. With c = 0, b = 5 is the smallest significant count (p = 0.031); b = 4 gives 0.0625. b = c = 0 gives p = 1.
- **A tie is not a win.** With b = c = 0 the test gives no evidence of a difference in either direction, and 150 pairs cannot show equivalence.
- **Power** (probability of seeing ≥ 1 failure / ≥ 5 failures in 150 games at true failure rate p): p = 0.5 %: 53 % / 0.1 %; 1 %: 78 % / 1.8 %; 2 %: 95 % / 18 %; 3 %: 99 % / 47 %; 5 %: 100 % / 87 %. So the 0-failure bar detects a 2 % failure rate with 95 % probability, but a significant McNemar result needs a failure rate of about 3–5 %.
- **Reliability bar** (as in step 2 and the decision log): **150/150**. One failure fails the bar even when it is not significant.

## 4. CSS gate
For each duck arm that passes 150/150, run the decision log's browser check on that model. `test/browser-check.js` has no option to point it at another model (its header and code read the root `model.js`/`model.css` via `path.join(__dirname, '..')`), so:
1. verify with `node -e` that `trainer/store.js` `ROOT` resolves to this worktree (checked before registration: `/Users/kyle/Desktop/dino-game/.claude/worktrees/agent-aceb6b8ae5aa15fef`);
2. `node trainer/cli.js export --dir runs/exp-bc-duck/d2w8-s<k> --which best` (writes `model.js` and `model.css` in this worktree only);
3. `node test/browser-check.js` only. Not the test suite: `test/net.test.js` pins the shipped model and would fail by design;
4. record the pass count N/38, the CSS agreement JSON (`frames`, `disagree`, `score`), and any non-CSS check that fails (reported separately from the gate);
5. `git checkout -- model.js model.css`, then `git status` must show them clean. Then the next arm.

**Bar: 0 CSS/JS disagreements** (the check also requires ≥ 1,000 frames), the same bar as the decision log's browser check (38/38, 0 disagreements over 1,015 frames).

**Coverage limit, stated in advance.** Section 5 of the check stops comparing once `score ≥ 100` and `frames ≥ 1,000` at 2048×. Pterodactyls need speed ≥ 8.5 (`core.js` line 230); the decision log puts the first pterodactyls at 460–740 points. So **this gate almost certainly compares no duck-class (class 2) decision**: it tests CSS/JS agreement on the opening only. A pass reads as "no CSS/JS divergence over the opening frames; class 2 not exercised by this check". The indirect evidence that covers ducks is step 2's margins over 600k frames of own play: d2w8-s1 5.61e-4, s2 3.58e-4, both above the 5.8e-5 CSS-risk flag and 3.4–3.6× their jump twins'. The check is not edited or extended here; a longer CSS comparison that reaches pterodactyls would be a separate, registered step.

## 5. Readings (fixed in advance; each duck arm is read separately)
| Result | Reading | Next |
|---|---|---|
| Every arm 150/150, and CSS 0 disagreements on both duck arms | **Equal reliability**, at the resolution of 150 seeds (each failure rate ≤ 1.98 % one-sided; McNemar b = c = 0, p = 1, no evidence of a difference). **A behaviour change only: the clone ducks.** Not a win: the duck clones are not shown to be better | **The user decides whether to ship.** If yes: export, `export-css`, browser check, `net.test.js` update and a decision-log entry, as a separate step |
| A duck arm fails ≥ 1 seed, jump 150/150 | **Reliability bar not met** for that arm. b ≥ 5: significantly worse than the jump clone (one-sided p ≤ 0.031); b = 1–4: not significant, but the bar still fails | That arm is not a ship candidate. Report the failure scores. Diagnosis (`diag.js --teacher duck`) is a separate step |
| One duck arm 150/150, the other fails | **The shape is not certified.** The passing seed alone is not a ship candidate (step 2 §7 reads an arm on both seeds) | Report per seed; the user decides whether a single-seed candidate is worth more work |
| Jump clone fails ≥ 1 seed | **The first held-out failure of the shipped clone.** It is not a duck result. The duck arms are paired against the jump clone as it stands (the general McNemar form; c > 0 possible) | Report to the user: it bears on the shipped model's own decision-log claim |
| Both duck arms and the jump clone fail the same seed(s) | Concordant failures; they don't enter McNemar | Report the seeds; check them against the solver before reading anything |
| A duck arm passes eval, but CSS ≥ 1 disagreement (or < 1,000 frames compared) | **The eval result stands; that arm is not shippable to the CSS page** as it is | Report frames and disagreements; CSS fidelity work is a separate step |
| Non-CSS browser checks fail with a duck model | Not a CSS-gate result; report it separately (e.g. a check that assumes jump-only play) | Investigate before any ship step |
| CSS passes | No CSS/JS divergence over the opening frames; class 2 not exercised by this check (§4) | Noted as a scope limit in any ship proposal |

## 6. What this does not do
- Nothing is shipped; `model.js`/`model.css` are never committed; no `session`, `finish` or `--auto-export`; the test suite is not run with a duck model in place.
- No change to `core.js`, `policy.js`, `test/solver.js` or `trainer/` between this commit and the Results commit.
- Scope: rules v2, target 1M, full games from 0, `--wasm` (bit-identical to JS), the two d2w8 training seeds, held-out offsets 325–474.
- A null (150/150 everywhere) is conditional on 150 seeds: it excludes failure rates above about 2 % per arm, not smaller ones. It reopens with more fresh offsets (from 475).
- Commits: the three `evals.jsonl` (the two under `runs/exp-*` with `git add -f`) and this file.

## Results (2026-10-10, mac subagent, branch `wt/cert-duck-d2w8`)
**Provenance.** Registered in `1a1baf2` (on main `8e98dfa`); evals committed in `d8665f0`. No change to `core.js`, `policy.js`, `test/solver.js` or `trainer/` on this branch. Core hash `0f329d4e3493…` (current `coreHash()` and all three `best.json`). Rules v2, target 1,000,000, `--wasm`, held-out offsets 325–474 (seeds 3,182,883,642 … 3,381,787,671). The three evals ran one at a time, in the registered order, with only the two dashboards otherwise running. Raw output is local (scratchpad); the records are the last line of each `evals.jsonl`.

### Certification (offsets 325–474, 150 seeds, paired)
| Arm | Run dir | Passed | 95 % CP pass rate (two-sided) | One-sided 95 % upper bound on failure | Failures | Min score | Wall |
|---|---|---|---|---|---|---|---|
| duck s1 (44-8-8-3) | `runs/exp-bc-duck/d2w8-s1` | **150/150** | 0.9757–1.000 | 1.98 % | none | 1,000,000 | 12.4 s |
| duck s2 (44-8-8-3) | `runs/exp-bc-duck/d2w8-s2` | **150/150** | 0.9757–1.000 | 1.98 % | none | 1,000,000 | 13.4 s |
| jump, shipped (44-32-32-3) | `runs/bc-v2-h32x32` | **150/150** | 0.9757–1.000 | 1.98 % | none | 1,000,000 | 26.0 s |

**McNemar (exact, one-sided, H1 "the duck clone fails more"):**
| Pair | b (only duck fails) | c (only jump fails) | Concordant failures | p |
|---|---|---|---|---|
| duck s1 vs jump | 0 | 0 | 0 | 1 |
| duck s2 vs jump | 0 | 0 | 0 | 1 |

The positive control (the jump arm) passed 150/150, as on 175–324. The shipped clone's held-out record is now 325/325 on offsets 0–24, 175–324 and 325–474.

### CSS gate (`test/browser-check.js`, each duck model exported into this worktree only)
`trainer/store.js` `ROOT` was re-checked before each export: `/Users/kyle/Desktop/dino-game/.claude/worktrees/agent-aceb6b8ae5aa15fef`. After each run, `git checkout -- model.js model.css` and `git status` was clean; `git diff main -- model.js model.css` is empty.

| Arm | Browser checks | CSS frames compared | Disagreements | Score reached | `model.css` id |
|---|---|---|---|---|---|
| duck s1 | **38/38** | 1,124 | **0** | 190 | `na-28a0ff7c2bb9f637` |
| duck s2 | **38/38** | 1,076 | **0** | 182 | `na-f5a2122fad734b6c` |

No non-CSS check failed with a duck model (sections 1–4 all pass). As registered (§4), the comparisons stopped at scores 190 and 182, below the speed at which pterodactyls spawn, so **no duck-class (class 2) decision was compared**. The indirect evidence for ducks remains step 2's margins (5.61e-4 and 3.58e-4, above the 5.8e-5 flag).

### §5 readings applied
- **Row 1 applies: every arm 150/150, CSS 0 disagreements on both duck arms.** Reading: **equal reliability** at the resolution of 150 seeds (each arm's failure rate ≤ 1.98 %, one-sided 95 %); McNemar b = c = 0, p = 1 for both duck arms, **no evidence of a difference in either direction**. This is a tie, not a win: the duck clones are not shown to be more reliable than the shipped jump clone. **The difference is a behaviour change only: the clone ducks** (step 2: duck share 1.000 on both seeds).
- **"CSS passes" row:** no CSS/JS divergence over the opening frames; class 2 not exercised by this check. This is a scope limit for any ship proposal.
- No failure row applies (no duck failure, no jump failure, no concordant failure, no CSS failure, no non-CSS failure).
- **Next: the user decides whether to ship a d2w8 duck clone.** If yes, a separate step: export, `export-css`, browser check, update `test/net.test.js`'s pinned model, and an AGENTS.md decision-log entry. A CSS comparison that reaches pterodactyls would close the class-2 gap and is worth registering before or with that step.
- **Held-out offsets 325–474 are now spent. The next fresh offset is 475.**
