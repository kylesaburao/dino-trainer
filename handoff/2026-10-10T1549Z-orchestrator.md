# Handoff: multi-machine training-efficiency work (2026-10-10, 15:49 UTC)

Snapshot of progress and findings across all sessions. Read `handoff/archive/2026-10-10T0742Z-HANDOFF.md` first (the archived root HANDOFF.md, sessions of 2026-10-09 → 10 morning). This file covers what happened after it, from about 14:50Z to 15:49Z. Where the two disagree, this file is newer.

Some of the work in it was still running at the snapshot time. Those items are marked **IN FLIGHT**: check their state before relying on them.

---

## 0. Status in brief

- The **shipped model is unchanged and still the best**: `runs/v2-es-d2w32-s1-t1m`, generation 970, 71/75 fresh held-out seeds.
- **The tie-rate σ controller (HANDOFF §9 fix 1) failed** in 14 replay runs: σ runs away. It is being reverted.
- **The stall rule is harmful** at wd 0.01: it skips 65–70% of steps and escaped 0 of 3 runs.
- **Plateau escape is a seed lottery.** On 32,32 about a third of seeds escape by generation 900. This holds for the shipped defaults, for the hygiene flags (mid ranks + decoupled wd 0.01), and for wd 0.1. Tie handling doesn't change it.
- **Decoupled wd 0.01 climbs as fast as the shipped run once escaped.** This is the first clean evidence that Run A's slow climb came from wd 0.1 (n = 1).
- **The x86 desktop reproduces the Mac bit-for-bit**, so either machine can run any experiment.
- The **gated curriculum (HANDOFF §9 fixes 2–4) is implemented** on `wt/desktop-2` and verified. It is not merged.
- **The user has opened `policy.js` input features to change, but only if tests show a clear gain.** Feature-sufficiency probes are **IN FLIGHT**.
- **Main has not moved** from `2c1e660`. Nothing has been merged or pushed to main yet.

## 1. User decisions and instructions in this period (authoritative)

1. **(a)** Proceed with HANDOFF §11 steps 1–3. Use subagents on the Mac and ask the desktop sessions to use them too, only as needed (parallel reasoning, prototyping, statistics).
2. **(b)** Work on branches. The orchestrator session on the Mac integrates into main and **may push main** (granted explicitly). The desktop sessions may push their own `wt/*` branches. The user authorised that in each desktop session; neither may push main.
3. The orchestrator should **delegate in-depth code editing to subagents**, to keep its own context small.
4. While idle, **keep Sonnet subagents researching NN training techniques**. Favour authoritative sources such as Andrej Karpathy. Write the results into the repo (`research/`).
5. "Consider that the input features as is may not be sufficient." Follow-up: **"Changes are open if tests show clear gain"** (applies to `policy.js` `observe()`).
6. Browser use is **not available on the desktop host**. `test/browser-check.js` runs only on the Mac.
7. This document was requested: "a timestamped handoff in a handoff folder".

Unchanged ground rules (AGENTS.md, HANDOFF §0–1):
- Experimental runs use `train` only, in `runs/exp-*`. Never `--auto-export`, `session` or `finish`.
- One trainer per host.
- Every change must stay generic over `--hidden`.
- New behaviour goes behind flags whose defaults are bit-identical.
- Certify only on fresh held-out offsets. The next one is **100**.
- `rm` only on literal paths.

## 2. Sessions, machines and worktrees

| Session | Machine | Worktree | Branch | Role |
|---|---|---|---|---|
| orchestrator (`dino-game-3e`) | Mac (M5 Pro, 18 cores) | `/Users/kyle/Desktop/dino-game` | `main` | coordination, integration, Mac experiments |
| Mac subagents | Mac | `/Users/kyle/Desktop/dino-game-mac` | `wt/mac` | harness and research notes |
| (reference) | Mac | `/Users/kyle/Desktop/dino-game-base` | detached `2c1e660` | old code for bit-identity checks; never commit |
| (experiments) | Mac | `/Users/kyle/Desktop/dino-game-d2` | detached `ee6995c` | milestone-1 code; holds the Mac's replay runs under `runs/exp-*` |
| desktop-1 | WSL2, Ryzen 7 9700X, 8C/16T | `/home/kyle-desktop/dino-game-desktop-1` | `wt/desktop-1` | experiments and analysis tools |
| desktop-2 | same host | `/home/kyle-desktop/dino-game-desktop-2` | `wt/desktop-2` | trainer code changes |

Desktop details:
- The shared clone is `/home/kyle-desktop/dino-trainer`. desktop-2 also has a base worktree at `/home/kyle-desktop/dino-game-base` (detached `2c1e660`).
- Node is only on PATH after `source ~/.nvm/nvm.sh`.
- The two desktop sessions share one CPU, so only one heavy job runs at a time across both.
- Pushes go over SSH: `git@github.com:kylesaburao/dino-trainer.git`. HTTPS has no credentials.
- Messaging: peers reach the orchestrator via its bridge session ID, not the name `dino-game-3e`.
- Desktop commits carry their harness's `Co-Authored-By`/`Claude-Session` trailers. They can be stripped at merge, e.g. by squash-merging.

Branch heads at the snapshot (`git fetch` first):
- `origin/main` = `2c1e660`.
- `origin/wt/desktop-2` = `7d71c32` (milestone 2), on top of `ee6995c` (milestone 1).
- `origin/wt/desktop-1` = `6f24fb4` (analysis tools). This includes force-added files under the gitignored `runs/exp-tools/`. Move them to a non-ignored path such as `tools/` at integration rather than keeping the force-adds.
- `wt/mac` (local only, not pushed) = `743abfe`:
  - `348cfa3` `test/bit-identity.js`
  - `c2b7695` `research/karpathy-training-techniques.md`
  - `6e1315f` `research/observation-design.md`
  - `743abfe` `research/es-practice.md`

## 3. Measurements and results (this period)

### 3.1 Throughput (`node trainer/cli.js bench --seconds 10`, default 16-unit net)
| | Mac, 18 workers | Desktop, 16 workers |
|---|---|---|
| 1 core, sim only | 7.98M fps | 6.21M |
| 1 core, sim + policy | 1.98M | 1.66M |
| all workers | **25.2M** | **10.25M** |

The Mac is about 2.5× faster. Put long runs there.

### 3.2 Cross-architecture determinism: PASS
desktop-1 replayed the shipped config (`--hidden 32,32 --target 1000000 --seed 1`, defaults, 800 generations) on x86. It matches the committed `runs/v2-es-d2w32-s1-t1m/metrics.jsonl` to 1e-9 in every field for all 800 generations. It does so again at `ee6995c`. The Mac's own control run also matches with 0 mismatches. **Results from either machine are interchangeable.**

### 3.3 Decay check (HANDOFF §11 step 1)
- Run A was resumed at generation 430 with `--weight-decay 0.01` for 6 minutes, in `runs/exp-A-wd001` on the Mac.
- The slope of ln fitMean per generation was 0.016 ± 0.018 over generations 431–441, against 0.028 ± 0.018 over Run A's generations 419–429. **Inconclusive**, as predicted, because the run started mid-climb.
- Validation over 36 seeds swung between 12 and 32 out of 36 from one generation to the next. Selection is noisy late in training, which supports two-stage validation (HANDOFF §9 fix 6).

### 3.4 Milestone 1: tie-rate σ controller → FAILED, being reverted
- Implemented by desktop-2 at `ee6995c` as flags `--tie-target`, `--tie-gain` and `--sigma-max`. It updates ln σ += K(q − q*), where q counts only ties below the fitness cap.
- The code is correct and bit-identical at defaults.
- Replays on the Mac: 32,32 and 32,32,32, seeds 1–3, 400 generations, base `--rank-ties mid --decay decoupled --weight-decay 0.01`. Runs live in `/Users/kyle/Desktop/dino-game-d2/runs/exp-rep-*`.

| Setting | Outcome |
|---|---|
| q* 0.2, σmax 1.0 | σ pinned at 1.0 for 38–71% of generations; fitMean ≈ 41 (every candidate dies at the first obstacle) |
| q* 0.2, σmax 0.3 | σ pinned at 0.3; q rises to 0.47–0.89 by generations 300–400; tie streaks of 23–217 generations |
| q* 0.35 (seed 1) | σ drifts down to 0.02 on 32,32 and 0.4 on 32,32,32 |

- 0 of 14 runs escaped.
- **Mechanism:** q(σ) is U-shaped, not monotone. With a large σ every candidate dies at the first obstacle, so pairs tie, and mid ranks then give a zero gradient. The Robbins–Monro argument in HANDOFF §9 fix 1 doesn't hold. HANDOFF §6.1's table already hinted at this, with ties rising again at ×3.375.
- **Don't iterate on this design.** The `--stall-rule inflate` idea was also dropped, because its trigger condition was never met.

### 3.5 Plateau escape (32,32, `--generations 900`, `--eval-every 50`)
Escape means the first generation with fitMean > 200.

| Arm | s1 | s2 | s3 | Where it ran |
|---|---|---|---|---|
| control: shipped defaults (index ties, coupled wd 0.005) | **792** | >900 | >900 | Mac |
| hygiene: mid + decoupled 0.01 | >900 | >900 | **533** | Mac |
| wd01: mid + decoupled 0.1 | **776** (matches HANDOFF §6.1) | >900 | >900 | desktop-1 |
| stall001: hygiene + `--stall-rule` | >900 | >900 | >900 | desktop-1 |

- At 400 generations, all 12 none/stall runs on 32,32 and 32,32,32 were censored. desktop-1's table is `runs/exp-tools/part2-table.md` on `wt/desktop-1`.
- **Reading:** escape looks like a lottery of about 1/3 per seed by 900, independent of the arm. In effect n ≈ 3, because arms share init and early noise.
- The stall rule skips 589–628 of 900 steps and freezes θ. Its σ capping at 0.4 is `STALL_MAX = 4`, not a bug.
- Censoring at 900 is tight: several runs were peaking around 100 near the end.

### 3.6 Climb after escape (generations / seconds from escape)
| Run | Escape | → fitMean 2k | → 20k | → 200k | ‖θ‖ |
|---|---|---|---|---|---|
| shipped (index, coupled 0.005) | 792 | 153 g / 29 s | 161 g / 54 s | 175 g / 280 s | 38.3 |
| hygiene s3 (mid, decoupled 0.01) | 533 | 160 g / 15 s | 172 g / 39 s | 186 g / 300 s | 34 → 38.5 |

- These match. Run A (wd 0.1, ‖θ‖ pinned at about 17) took 3–6× more generations.
- **Supports:** use decoupled 0.01, not 0.1 (HANDOFF §9 fix 7). This is n = 1 on a different seed.

### 3.7 Milestone 2: gated curriculum (desktop-2, `7d71c32`): VERIFIED, not merged
- **`--window-adapt` now has new semantics** (its default is off, so defaults are unchanged):
  - new key `windowAdaptEvery`, set with `--window-adapt-every N`, default 5;
  - pooled start-0 and late pass counts in checkpointed `st.adaptAcc`;
  - the window shrinks if the start-0 pass rate is < 0.3;
  - the late rule (adaptWindow) applies only if the start-0 pass rate is > 0.6, or there are no start-0 episodes;
  - otherwise the window holds;
  - a shrink never grows the window.
  - Resuming Run B backfills `windowAdaptEvery = 5`.
- **`--fitness-agg mean|geomean`** (curriculum only): F = exp(mean_j ln max(f_j, 1/span_j)).
  - One failing stage gives (1/span)^(1/E), for example 0.21 at E = 4 and span 500. It is not ≈ 0, which HANDOFF §9 overstated.
- **Verification** (desktop-2):
  - 62/62 tests; smoke 17/17;
  - bit-identity across 6 scenarios: IDENTICAL;
  - determinism across a split at generation 13 or 15 followed by resume: IDENTICAL;
  - adaptAcc restore tested with a mutant.
- **Not yet trained with.** Corrected Run B′ (HANDOFF §11 step 4) has not been run.

### 3.8 Bit-identity harness (Mac subagent, `wt/mac` `348cfa3`)
- Usage: `node test/bit-identity.js <oldTree> <newTree> [--workers N] [--keep] [--only a,b] [--allow-busy]`.
- It runs 6 scenarios (plain, plain-2x 16,16,16, curric-log, curric-points, heldout, hygiene), 20 generations followed by a resume to 40. Each scenario runs in both trees.
- It diffs metrics, checkpoint and best.json. A full run takes about 9 s. It refuses to start while a trainer is running.
- Base vs `ee6995c`: IDENTICAL, 42,949 values.
- **Gap:** the `allpass` held-out stop path is never triggered. Add a scenario that reaches it before trusting the harness on that path.

## 4. IN FLIGHT at snapshot time

1. **desktop-2: revert milestone 1 on `wt/desktop-2`.**
   - Run `git revert ee6995c`, keeping milestone 2 and the decay comment fix (~0.01).
   - Update the `--stall-rule` header to "not recommended; kept to reproduce Run A".
   - Re-run tests, smoke, bit-identity and determinism, then push and report.
2. **desktop-1: seed survey** for seed racing (HANDOFF §9 fix 10).
   - Hygiene arm, 32,32, seeds 4–12, `--generations 1500 --time-budget 3 --eval-every 50`, dirs `runs/exp-surv-32x32-sS`.
   - It reports P(escape by 900 and by 1500), the Kaplan–Meier median, sd(ln T_escape), and the K for which 1−(1−p)^K ≥ 0.95.
3. **Mac subagent: feature-sufficiency probes**, writing `research/feature-sufficiency.md` on `wt/mac`. Scripts are in the orchestrator scratchpad `.../scratchpad/features/`.
   - Probe A: crash vs survive separability before the shipped model's early (464–633) and late (~990k) crashes.
   - Probe B: behaviour-clone `test/solver.js` on the 44 features, with and without candidate features (time-to-reach = relx/speed, speed/22, spacing to the second obstacle), and judge by rollout scores.
4. **Mac subagents (Sonnet): research notes.**
   - `research/es-practice.md` is committed (`743abfe`); its report has not been reviewed yet.
   - `research/width-depth-transfer.md` (µP, per-layer σ, deep tanh init, sweep methodology) is running.

## 5. Research notes so far (`research/` on `wt/mac`)

1. **`karpathy-training-techniques.md`** (18 verified / 4 unverified references):
   - Karpathy has essentially no ES content.
   - Salimans et al. 2017 gives the rationale for weight decay in ES (keep θ from outgrowing σ) and a width argument (divide σ and lr when width doubles).
   - It reports policies that ignore their input (fixed by virtual batch norm).
   - Other takeaways: initialise the output layer to avoid a degenerate policy; use saturation diagnostics; screen each architecture first with an "overfit a small batch" check on 1–3 seeds; use random log-scale hyperparameter search.
   - CS231n's update/weight ratio target of 1e-3 does **not** carry over to Adam-ES.
2. **`observation-design.md`** (17 / 10):
   - Small dino and Flappy agents use 4–6 inputs, and speed is decisive. No source tests time-to-collision as an input.
   - Running-stat normalisation matters for locomotion but is not worth its state here; prefer fixed scaling (speed/22).
   - Late-game lookahead shrinks: about 27 frames at top speed against about 100 at the start. A frames-to-impact feature is the candidate.
   - No frame stacking is needed.
   - Bundle any input change into one fresh run.
3. **`es-practice.md`** (31 references: 19 read as source or code, 7 via a summarising tool, 5 unverified):
   - **Ties in reference implementations.** The reference ES implementations (es-starter, estool, EvoJAX) break ties in sort order, like the shipped trainer. evosax uses midranks. ARS uses pair differences divided by the reward std, with no ranks.
   - **No published fix for "most pairs tie".** The closest rule is CMA-ES's flat-fitness response: if the best value equals the 70th-percentile value, σ ×exp(0.2 + cs/damps). Hansen also calls flat fitness "rather a termination criterion".
   - **Larger σ can cross zero-fitness gaps.** Lehman et al. 2018 show this on 2-D toy landscapes, with σ ≈ 0.16–0.18.
   - **An adaptive exploration weight from NSRA-ES.** After t_w = 50 generations without a new best, the novelty weight shifts by δ = 0.05. t_w = 10 worked better on short budgets.
   - **Observation statistics are frozen at evaluation** in every reference implementation. The only effect size found is qualitative (ARS Humanoid).
   - **Typical σ and lr settings.** σ 0.05–0.1 decaying ×0.999 to a floor of 0.01, and lr 0.01–0.02 decaying to 0.001 (evosax, estool). Salimans 2017 saw no benefit from adapting σ.
   - **Restarts.** IPOP doubles the population on each restart. BIPOP interleaves large and small restarts. No ES-for-RL paper applies restarts to OpenAI-ES.
   - **Late-phase selection noise.** Such et al. re-evaluate the top 10 candidates on 30 extra episodes before naming the elite. This supports better validation over a better gradient.
   - **Output layer near zero at init.** es-starter initialises the output layer at column norm 0.01, against 1.0 for hidden layers. That is a cheap experiment, given the 52% none / 47% duck init policy here.
4. **`width-depth-transfer.md`**: IN FLIGHT.

## 6. Decisions taken by the orchestrator (with advisor review)

- Don't merge milestone 1. Revert it (§3.4).
- Deprecate `--stall-rule` in docs (keep the code for reproduction).
- Drop the plan to judge the controller against 400-generation replays. That test was mis-sized: the shipped config escapes at about 776–792.
- Seed racing is the chosen exploration lever; its K comes from the survey.
- The research subagents commit only to `wt/mac`; the orchestrator merges.

## 7. Next steps (in order)

1. When desktop-2 reports the revert:
   - fetch `wt/desktop-2`;
   - run on the Mac `node test/bit-identity.js /Users/kyle/Desktop/dino-game-base <worktree at the new head>`, plus `node --test test/`, `node test/smoke-trainer.js` and `node test/browser-check.js` (Mac only);
   - merge into main, squashing or stripping trailers if wanted.
2. Merge `wt/mac` (harness + research).
3. Relocate desktop-1's durable tools (`diff-metrics.js`, `replay-table.js`) out of `runs/exp-tools/` into `tools/`.
4. Add an `allpass` scenario to the harness.
5. Delegate to a subagent: update AGENTS.md (new flags, stop-rule wording, the stall rule not being recommended, the multi-machine notes) and fix root HANDOFF.md's stale "nothing committed" status. Then push main (authorised).
6. From the seed survey, choose K and the generation cap for seed racing. Then size Run B′ (HANDOFF §11 step 4) with the hygiene flags, wd 0.01 and the gated curriculum. Consult the advisor before launching it.
7. Feature-sufficiency verdict:
   - if a clear gain is shown, plan one bundled `observe()` change: fresh runs, the `model.css` / `css-nn.js` 44-input wiring, the anti-cheat and `net.test.js` updates, and the browser check;
   - certify any candidate on held-out offset 100 or later, against the shipped model on the same seeds.
8. Layer-size sweep (the user's plan) after that, using `research/width-depth-transfer.md` and seed racing.

## 8. Run data locations (gitignored; local disks only)

- **Mac, `/Users/kyle/Desktop/dino-game/runs/`:**
  - `exp-A-wd001` (decay check);
  - the earlier `exp-A-hygiene-h32x32`, `exp-B-curric-h32x32` and `exp-window-h32x32`.
- **Mac, `/Users/kyle/Desktop/dino-game-d2/runs/`:**
  - `exp-rep-*`: controller replays at targets 0.2 and 0.35 (`tie2`, `tie35`) and with σmax 0.3 (`tie2max3`);
  - `exp-esc-32x32-s{1,2,3}-{control,hyg}`.
- **Desktop-1 worktree `runs/`:**
  - `exp-xarch` (cross-architecture check);
  - `exp-rep-*-{none,stall}` (400 generations);
  - `exp-esc-32x32-s{1,2,3}-{wd01,stall001}`;
  - `exp-surv-*` (survey, in flight).
