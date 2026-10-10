# Handoff: orchestration transfer from `dino-game-3e` to `mac-2` (2026-10-10, 16:26 UTC)

**You (mac-2) are now the orchestrator.** The user asked for orchestration to move to mac-2 with full context, because this session's context window was filling up.

The user is **away and running the work unattended**. They review through Claude Remote Control.

Read in this order:
1. `AGENTS.md`, which is up to date: flags, verification tools and findings.
2. This file.
3. `handoff/2026-10-10T1549Z-orchestrator.md`, the previous snapshot, with full results tables for 15:49Z and earlier.
4. As a reference only, `handoff/archive/2026-10-10T0742Z-HANDOFF.md`. This is the archived, stale root HANDOFF; research notes cite its sections as "HANDOFF §N".

Also read the auto-memory index (`~/.claude/projects/-Users-kyle-Desktop-dino-game/memory/MEMORY.md`). It records:
- the user's workflow preferences;
- the multi-machine setup;
- that layer-agnostic work is required;
- that `policy.js` is open to change only on clear gains.

---

## 1. Standing user instructions (authoritative)

1. **Role.** Be the **orchestrator**.
   - Delegate in-depth code editing to subagents on this Mac, or to the desktop sessions.
   - Use subagents only as needed: parallel reasoning, prototyping or statistics.
2. **Idle time.** While idle, **keep Sonnet subagents researching NN training techniques**, favouring authoritative sources such as Karpathy. Write the results to `research/` on `wt/mac`, then merge them to main.
3. **Branches and pushing.**
   - Work goes on branches.
   - **The orchestrator integrates into main and may push main.**
   - Desktop sessions push only their own `wt/*` branches. The user authorised that in each desktop session.
4. **Docs.** The never-push and commit restrictions were **removed from the agent docs at the user's request**. The stale root HANDOFF.md was **archived** to `handoff/archive/`.
5. **Input features.** `policy.js` input features (`observe()`) **may change if tests show a clear gain**. Bundle all feature changes into one fresh run. The shipped model stays until a new one beats it on fresh held-out seeds.
6. **Browser.** No browser is available on the desktop host. `node test/browser-check.js` runs only on the Mac.
7. **Unchanged ground rules:**
   - Experimental runs use `train` in `runs/exp-*` only. Never use `--auto-export`, `session` or `finish`: they overwrite the shipped `model.js`.
   - One trainer per host. The two desktop sessions share one CPU.
   - Keep everything generic over `--hidden`.
   - New behaviour goes behind flags whose defaults are bit-identical. Verify with `test/bit-identity.js`.
   - `rm` only on literal paths, enforced by a hook. The hook also blocks `rm` in any command that contains `$`, globs, `~` or braces.
   - Don't edit `core.js` or `model.*`.
   - **Seed offsets.** Held-out offsets 0–174 are used. **The next fresh offset is 175.**

## 2. Peers and how to reach them

| Peer | Where | Role now | Address |
|---|---|---|---|
| desktop-1 | WSL2 Ryzen 9700X, `/home/kyle-desktop/dino-game-desktop-1`, branch `wt/desktop-1` | experiments and analysis | SendMessage `to: "desktop-1"` works; replies come from `bridge:session_017dSzHB17BR88LBomfMfTzk` |
| desktop-2 | same host, `/home/kyle-desktop/dino-game-desktop-2`, branch `wt/desktop-2` | trainer code owner and verifier | `desktop-2`, or `bridge:session_01LGggTtejaCN5qNEutis31D` |

Notes on the desktop host:
- Node is only on PATH after `source ~/.nvm/nvm.sh`.
- The shared clone is `/home/kyle-desktop/dino-trainer`. desktop-2 keeps a base worktree at `/home/kyle-desktop/dino-game-base`, detached at `2c1e660`.
- Pushes go over SSH.
- Desktop commits carry `Co-Authored-By`/`Claude-Session` trailers. Squash-merging strips them.
- **The first thing to do as orchestrator:** message both desktops that mac-2 is now the orchestrator and that they should report to you. Use the bridge from-address they send from.

## 3. State of main (pushed, `a5c02cc`)

- **Milestone 2:** gated `--window-adapt` with `--window-adapt-every`, and `--fitness-agg mean|geomean`.
- **Docs:** decoupled weight decay advice is ~0.01, and `--stall-rule` is deprecated.
- **The tie-rate σ controller was removed** because it failed (see the previous handoff, §3.4).
- `test/bit-identity.js` (6 scenarios, about 9 s). Its gap: the `allpass` held-out stop path is never triggered.
- `research/` holds six notes:
  - `karpathy-training-techniques.md`
  - `es-practice.md`
  - `observation-design.md`
  - `width-depth-transfer.md`
  - `wasm-simd-mlp.md`
  - `rare-failure-robustness.md`
- AGENTS.md: flag map, verification tools, and findings that constrain defaults.
- `handoff/` holds this file, the 15:49Z snapshot, and `archive/`.

## 4. Pending integration: `wt/wasm` (`origin/wt/wasm` = `0af8fed`, rebased on main `50e6f97`)

- **What it is:** an opt-in `--wasm` flag. It runs the MLP as a WASM SIMD f64x2 forward pass in the trainer only.
  - `policy.js` and the core hash are untouched. `Math.tanh` stays in JS.
  - It adds `trainer/wasm-net.js` and `test/wasm-net.test.js`.
  - It changes `trainer/env.js` (`runEpisode` takes a `makeAgent`), `trainer/worker.js`, `trainer/cli.js` (adds the flag; `eval --wasm`; `bench --wasm --hidden`) and `trainer/bench.js`.
- **Mac verification:**
  - 66/66 tests; smoke 17/17.
  - Defaults are IDENTICAL against main.
  - `--wasm` vs JS is IDENTICAL for 2-layer, 3-layer, curriculum (plus milestone-2 flags), held-out stop and resumed-checkpoint runs.
  - **1.68× faster** training on 44-32-32-3 (bench: 1.66–1.84× on one core, about 1.75× on 12 workers).
- **Waiting on:** desktop-2's x86 verification (tests, smoke, harness, with vs without `--wasm` on 3 configs, x86 bench). It was briefed with all of that.
- **Merge procedure:**
  1. When desktop-2 reports IDENTICAL, run `node test/browser-check.js` in `/Users/kyle/Desktop/dino-game-wasm`. It should be unaffected, but it is required before main.
  2. Merge `wt/wasm` into main (`--no-ff`, or fast-forward if main hasn't moved) and push.
  3. Add `--wasm` to AGENTS.md's flag map, and recommend it for all experiments.
  4. If x86 differs at all, don't merge. Ask desktop-2 for the first difference.

## 5. Results since the 15:49Z handoff

### 5.1 Seed survey (desktop-1, hygiene arm = `--rank-ties mid --decay decoupled --weight-decay 0.01`, 32,32)
Seeds 4–12, run to 1500 generations, plus seeds 1–3 from earlier.

| | Value |
|---|---|
| P(escape by gen 900) | 3/12 = 0.25 (95% CI 0.055–0.57) |
| P(escape by gen 1500), Kaplan–Meier | 0.89 (sustained escape: 0.79) |
| Median escape generation, Kaplan–Meier | 1202 |
| sd(ln T) over escaped seeds | ≈ 0.4 (biased low) |
| K seeds for 95% (1−(1−p)^K) | about 11 at a 900-gen horizon; **2–3 at a 1500-gen horizon** |

- A seed that doesn't escape costs about 0.06 s per generation on 16 desktop workers. That is about 1–1.5 minutes per 1500 generations, and less on the Mac.
- **Seed-racing plan:** run 2–3 seeds to about 1500 generations and keep the first to escape.
- **The second wall.** Of the 8 escaped seeds, only 2 reached fitMean 2000 within budget. The others crept along at 200–600 points, around the first pterodactyls; seed 9 sat at about 500 for 450 generations. The second wall may cost more than the plateau.

### 5.2 Certification on fresh held-out offsets 100–174 (75 seeds)
| Model | Passed | 95% CP | Failure scores |
|---|---|---|---|
| shipped `model.js` (`runs/v2-h32x32-1m`, gen 970) | **72/75** | 0.888–0.992 | 655, 531 (early); 991,514 (late) |
| hygiene s3 (`/Users/kyle/Desktop/dino-game-d2/runs/exp-hyg-s3-full`, best = gen 733, 12/12 validation) | **71/75** | 0.869–0.985 | 727, 585, 536, 603 (all early) |

- These are statistically tied.
- The hygiene model was trained from scratch with mid ranks and decoupled wd 0.01 (seed 3, escape at 533). It then continued with `--select count --stop-rule allpass:59 --val-seeds 12 --fold-validation` and stopped on the time budget at gen 734.
- **6 of 7 failures are early, at 530–730 points (first pterodactyls), in both models.** That is the next thing to fix.
- Offsets 100–174 are now used.

### 5.3 Other findings
- **Climb after escape.** With decoupled wd 0.01 it matches the shipped run: about 160, 172 and 186 generations from escape to fitMean 2k, 20k and 200k. Run A at wd 0.1 took 3–6× more. That is evidence (n = 1) that wd 0.1 caused Run A's slow climb.
- **The decay check (Run A resumed with wd 0.01 for 6 minutes) was inconclusive.**

### 5.4 Research note summaries (decision-relevant only)
- **`wasm-simd-mlp.md`:** led to the `--wasm` implementation.
  - A bit-exact tanh port is not feasible: this arm64 Node's tanh behaves like fdlibm with FMA. So tanh stays in JS.
  - Batching with lanes = episodes would be the next speed-up.
- **`rare-failure-robustness.md`:**
  - Min or CVaR aggregation over 4 episodes changes little, because ranks are invariant to monotone transforms of the aggregate. The bottleneck is how often failures show up in training episodes, so mine hard seeds from a *dedicated* seed range.
  - The first cheap check is to reconcile the two late-game population hazard estimates in the archived HANDOFF §4.3, which differ by about 80×. If the late failures are intrinsic to the policy, shrinking σ won't fix them.
  - Averaging the last k checkpoints is the only fix that needs no retraining, but there's no ES evidence for it.
  - Watch the winner's curse in validation selection. Certify on fresh offsets.
- **`width-depth-transfer.md`:**
  - µP gives no depth rule for plain tanh MLPs. Sweep depth; don't extrapolate it.
  - `--sigma-scale fanin` is the σ rule with a source behind it.
  - Use plateau escape (time-to-event) rather than score for rung-1 promotion in the sweep.
  - Separating shapes needs many seeds: about 35 per arm to tell 1/3 from 2/3.
- **`es-practice.md`:**
  - The reference implementations break ties in sort order; evosax uses midranks.
  - CMA's flat-fitness rule is the closest precedent for tie handling.
  - es-starter initialises the output layer near zero, a cheap init experiment.
- **`observation-design.md`:**
  - A candidate frames-to-impact feature (relx/speed), and speed scaled by 22.
  - Bundle feature changes into one run.

## 6. IN FLIGHT at transfer time

1. **Mac subagent: feature-sufficiency probes**, started about 15:42Z.
   - Output: `research/feature-sufficiency.md`, committed on `wt/mac`. Scripts are in the old orchestrator's scratchpad, `/private/tmp/claude-501/-Users-kyle-Desktop-dino-game/a1a47f55-01e9-48b9-afbe-71ea588921bc/scratchpad/features/`.
   - Probe A: separability of crash and survive situations before the shipped model's early and late crashes.
   - Probe B: behaviour-clone the exact solver on the 44 features, with and without candidate features (time-to-reach, speed/22, spacing to the second obstacle), judged by rollout scores.
   - **This decides the policy.js feature question.**
   - The orchestrator will wait for it and merge it before transfer. If it isn't merged when you read this, check `git log wt/mac`.
2. **Mac subagent (Sonnet): research note 7, `research/demo-bootstrapped-es.md`.** Topic: behaviour-clone the exact solver, then fine-tune with ES, to skip the plateau and the pterodactyl wall. Same merge procedure as item 1.
3. **desktop-1: the second wall.**
   - It is resuming stuck survey seeds s4, s5, s7, s9 and s12 (plus s8) up to 4000 generations to measure the time from 200 to 2000.
   - It is also diagnosing the deaths of two stuck θs over 50 games each (obstacle type, ptero height, speed, actions in the last 30 frames, whether the policy ever ducks).
   - It was asked to move its survey script (KM/CP/K) to `tools/` on `wt/desktop-1`. Its older tools (`diff-metrics.js`, `replay-table.js`) sit under the gitignored `runs/exp-tools/` on that branch; relocate them to `tools/` before merging.
4. **desktop-2: x86 verification of `--wasm`** (§4).

## 7. Recommended next steps (orchestrator's plan; consult the native advisor at decision points)

1. **Merge `wt/wasm`** after the x86 check (§4). From then on, run all experiments with `--wasm`.
2. **Act on the two pterodactyl-wall analyses** (desktop-1's diagnosis plus the feature-sufficiency verdict). Decide among:
   - (a) feature changes in `policy.js` (one bundled change, then fresh runs);
   - (b) a training-only fix: a curriculum, or hard-seed mining that targets the 400–800-point zone, for example `--start-zero-cap` around 2000 with gated `--window-adapt`;
   - (c) BC-from-solver initialisation (research note 7).
   - Pick by evidence. Each needs paired runs (at least 3 seeds per arm) and certification at offset 175 or later, compared with the shipped model on the same seeds.
3. **Corrected Run B′** (archived HANDOFF §11 step 4, now feasible). Use:
   - hygiene flags, `--weight-decay 0.01`, `--wasm`;
   - the gated curriculum: `--window 25000 --episodes 8 --start-dist points --start-zero-frac 0.5 --start-zero-cap 50000 --window-adapt --fitness-agg geomean`;
   - `--select count --stop-rule allpass:59 --val-seeds 12 --fold-validation`;
   - seed racing with 2–3 seeds up to about 1500 generations.
   Compare it with a hygiene arm without the curriculum.
4. **Layer-size sweep** (the user's plan; `research/width-depth-transfer.md` §4). Rung 1 is plateau escape as time-to-event, with `--sigma-scale fanin`. Seeds go mostly to the surviving shapes.
5. **Housekeeping:**
   - add an `allpass` scenario that actually triggers to `test/bit-identity.js`;
   - remove the stale worktrees `/Users/kyle/Desktop/dino-game-d2m` and, once its runs are no longer needed, `/Users/kyle/Desktop/dino-game-d2`, with `git worktree remove`;
   - write a new timestamped `handoff/` file at each milestone.
6. **Keep research subagents going** while idle. Topics not yet covered:
   - quality-diversity or novelty for the plateau, with a behaviour descriptor;
   - validation and selection design with two-stage validation;
   - batching episodes in lockstep (lanes = episodes) for WASM;
   - reconciling the population hazard estimates (that one is analysis, not research).

## 8. Run data (gitignored; local disks)

- **Mac `/Users/kyle/Desktop/dino-game-d2/runs/`:**
  - `exp-rep-*`: controller and replay runs, 400 generations;
  - `exp-esc-32x32-s{1,2,3}-{control,hyg}`: 900 generations;
  - `exp-hyg-s3-full`: the certified hygiene model, best gen 733;
  - `exp-esc-32x32-s3-hyg`: its source run.
- **Mac `/Users/kyle/Desktop/dino-game/runs/`:** `exp-A-wd001` and the older `exp-A/B/window` runs.
- **desktop-1 worktree `runs/`:** `exp-xarch`, `exp-rep-*`, `exp-esc-*-{wd01,stall001}` and `exp-surv-32x32-s4..s12`.

---

## 9. Addendum at 16:45Z: results that finished before transfer

### 9.1 Feature sufficiency: the 44 features are sufficient (`research/feature-sufficiency.md`, main `1e8213b`)
**Probe A: the shipped model's crashes are behavioural, not a lack of information.**
- **Early crashes** (2.65% of 2,000 starts, at 461–688 points). All are a small cactus with a pterodactyl visible behind it. There are two mechanisms of about equal size:
  1. **Premature jump.** The dino takes off about 31 frames before contact instead of about 12. In 24 of 27 cases it jumps on the frame a low pterodactyl first appears.
  2. **Stuck jump key.** The dino presses DOWN then JUMP in mid-air and lands with the jump key held, so later JUMP outputs don't fire. "Jump key held" is the only unobserved state found; it is the player's own input, so it is allowed information.
- **Late crashes** (0.36% of 10k windows from 960k, starting exactly at the derived 962.6k speed threshold). All are the same event: landing from a long jump into a fast, high pterodactyl. Holding DOWN to fast-drop would save the run, but the model never presses DOWN after level 0.
- **Separability:** crash and pass situations are separable on the raw 44 features. AUC is 0.85–0.93 for early crashes and 1.0 for late ones.

**Probe B: a clone of the exact solver plays to 1M on 25 of 25 full games** on the unchanged 44 features, with 0 crashes across 352 games and 30,000 windows.
- This holds even for a 44-8-8-3 network, and for one trained on only 50k frames.
- The candidate features (speed/22, time-to-reach, gap, jump key held) had nothing left to improve.
- **The features can represent a near-perfect policy.** ES is not finding it.

**Recommendation:** keep the 44 features. If `observe()` is ever reopened, add "jump key held". The fixes are behavioural and training-side.

### 9.2 Pruned-branch register update (retrospective §6.4)
- **Feature change in `policy.js`:** **pruned as a fix for the current failures.**
  - Scope: the shipped model's early and late crash modes, and solver clones on 44-32-32-3 and 44-8-8-3.
  - Validity: a positive control exists (clones reach 25/25).
  - Reopen if a new failure mode appears that the features can't separate, or if the jump-key-held state proves causal and ES can't work around it.
- **Behaviour-cloning initialisation from the solver:** **strongly promoted.**
  - The necessary condition is met: the clones survive to 1M.
  - **An owner decision is needed first.** The clone acts only on allowed features, but its labels come from the solver's privileged information (the hidden gap and lookahead). Ask the user whether a model trained this way, or one derived from it by ES fine-tuning, is acceptable to ship.

### 9.3 Saved clone models (local, gitignored)
- `/Users/kyle/Desktop/dino-game/runs/exp-bc-clones/m_base_s1.json` and `m_base_s2.json`: 44-32-32-3 clones of the solver on the unchanged features. The other `m_*` files are variants and 8,8 nets.
- Format: `{variant, layers, params}`, which may need wrapping into the `model.js` format before use (`trainer/cli.js eval --model`).
- **Suggested first experiment, pending the user's decision:**
  1. Certify `m_base_s1` against the shipped model on fresh offsets **175–324 (150 seeds)**. 150 seeds is the minimum that can show an improvement over 72/75.
  2. If it clearly wins and the user approves solver-derived models, it becomes a candidate to ship, or a starting θ for ES polishing with small σ.

### 9.4 Revised priority (replaces §7 step 2)
1. **Ask the user about solver-derived models.** That decides whether the BC route can be shipped.
2. **Run the certification in 9.3 regardless.** It is cheap, about 1–2 minutes with `--wasm` once merged. Whatever the shipping decision, it measures how far ES is from what the features allow.
3. For pure ES, the failure modes now point to specific training-side fixes:
   - more training mass in the 400–800-point opening, for example a curriculum or hard-seed mining;
   - exposure to fast, high pterodactyls past 962k, for example points-uniform late windows that reach 960k–1M.
   - Each needs the pre-registration in AGENTS.md "Research method".
