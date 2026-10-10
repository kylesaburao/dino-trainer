# HANDOFF: training-efficiency investigation (sessions of 2026-10-09 → 10)

Written for a new session with no context. It records everything found while investigating how to train the dino-game AI faster and more reliably: the measurements, every agent's findings, the code changes now in the working tree, the experiment results, and what to do next. Read `AGENTS.md` first for the architecture; this file assumes it.

> **Status in one paragraph.**
> - The shipped model (`model.js`, from `runs/v2-h32x32-1m`, generation 970) is still the best model. It passes **71/75 fresh held-out seeds** (95% interval 0.869–0.985), and about **0.92 over 125 independent seeds**.
> - A large set of opt-in trainer improvements (W1–W6) is implemented, uncommitted, and **bit-identical to the old trainer when every new flag is off** (tests 55/55, smoke 17/17).
> - Two runs using them did worse:
>   - **Run A** (stop-rule and cleanup fixes): 46/75. It hit its 20-minute budget mid-climb.
>   - **Run B** (curriculum): 0/75. It never learned the opening, because of a design flaw in `--window-adapt`.
> - The fixes for both are known and are a few lines each; they are listed in section 9 and not implemented.
> - **Nothing has been committed.** The user has not yet decided the next step or whether to commit.

---

## 0. Things that changed or broke between sessions (read first)

1. **The repo moved** from `/Users/kyle/Downloads/dino-game` to `/Users/kyle/Desktop/dino-game`.
2. **The Python venv is broken by the move.** `.venv/` was created at the old path, so its scripts' shebangs point to `/Users/kyle/Downloads/dino-game/.venv/bin/python`. Recreate it:
   ```sh
   cd /Users/kyle/Desktop/dino-game
   rm -r .venv    # the rm hook only allows literal paths; this is one
   python3 -m venv .venv
   .venv/bin/python -m pip install --upgrade pip
   .venv/bin/python -m pip install numpy scipy pandas matplotlib statsmodels sympy
   ```
   - System Python is 3.9.6 at `/usr/bin/python3`. The versions installed last time were numpy 2.0.2, scipy 1.13.1, pandas 2.3.3, statsmodels 0.14.6, sympy 1.14.0.
   - `.venv/` is already in `.gitignore` (an uncommitted edit).
3. **All analysis scripts from the research agents are gone.** They lived in the old session's scratchpad (`/private/tmp/claude-501/-Users-kyle-Downloads-dino-game/.../scratchpad/{proofs,research,fable,stats}/`), which no longer exists. Section 7 describes what each one did, so any of them can be rebuilt. The most valuable to rebuild is the **exact replay** script (7.1).
4. **The run data survives**, under `runs/` in the repo. Most of it is gitignored, so it exists only on this disk:
   - `runs/exp-window-h32x32/`, `runs/exp-A-hygiene-h32x32/`, `runs/exp-B-curric-h32x32/`: each has `metrics.jsonl`, `metrics.csv`, `best.json`, `evals.jsonl`, `checkpoint.json` and `train.log`. They are gitignored by `runs/exp-*/`.
   - `runs/exp-A-hygiene-h32x32/REPORT.md`: the implementer's full report, also gitignored. Its content is reproduced in sections 5–6 and 9.
   - `runs/v2-h32x32-1m/` (shipped model) and `runs/v2-h32x32/` (the 500k model): their `best.json`, `evals.jsonl` and `metrics.*` are committed; checkpoints and logs are not.
   - `runs/smoke`, `runs/smoke-v2`, `runs/v2-h16`: older, gitignored.
5. **The v1-era run archive** is `/Users/kyle/Downloads/dino-game-v1-runs-archive-2026-10-09.zip`, outside the repo. It still exists. It contains `runs/exp-h3x32`, a 44-32-32-32-3 network on game v1, seed 1. That run sat on a 55–88-point plateau until generation ~1000–1100 (validation 1655 at gen 1100, 331 s), with ‖θ‖ reaching 46. v2 equals v1 below 1k points, so this is a usable control for plateau experiments.
6. **The auto-memory folder moved with the project path.** The old session wrote one memory: "Don't micro-optimize training for one layer configuration; the user will sweep hidden sizes and layer counts; 32x32 is just the starting point." It was at `~/.claude/projects/-Users-kyle-Downloads-dino-game/memory/layer-agnostic-training.md` and has been copied, with its `MEMORY.md` index, to `~/.claude/projects/-Users-kyle-Desktop-dino-game/memory/`. The fact is also restated in section 1.
7. **The other sessions are gone.** `implementor` was a separate interactive Claude session that did the coding and the runs; it is not available. No trainer is running (`pgrep -fl trainer/cli.js` was empty at handoff).

---

## 1. What the user wants, and their stated preferences

- **The goal:** make training more efficient and more robust. In particular, models where *all* held-out seeds reach the target (1,000,000 points on game v2), in less wall time.
- **Generic over network shape.** No micro-optimising for one layer configuration. The user plans a **sweep of hidden layer sizes and layer counts**; 32,32 is only the starting point. Every trainer change must be generic over `--hidden`.
- **`AGENTS.md` is self-contained.** The user asked that it not reference `PROGRESS.md` or the training prompt; keep it that way. The user had `/init` write it as `AGENTS.md`. It still starts with the `# CLAUDE.md` header that `/init` requires. `AGENTS.md` is untracked.
- **Commits:** none without the user's explicit request. The user has been asked, and hasn't answered, whether to commit the trainer changes, and whether on a branch or main.
- **Working style:**
  - The user likes delegating to subagents, including Opus and Fable models in parallel.
  - Subagents should report findings to the main session as they go, so the main session can relay them to the other agents.
  - The user asked for in-depth research, first-principles proofs and statistical analysis.
- **Repo rule:** experimental runs never use `--auto-export` (it overwrites the shipped `model.js`/`model.css`), and never `session` or `finish` (they git-commit).
- **A hook restricts `rm`:** only literal, fully written paths, inspected first. No globs, `~`, `$`, `xargs`, `find -exec` or loops.
- **Pre-existing state to leave alone:** `TrainingPrompt.md` is deleted in the working tree (`D`). It was already like that when this work began, and it is not ours to commit or restore without asking. The spec it contained is still in git history (commit `c58f203`).

---

## 2. Background measurements (all measured in the first session, M5 Pro: 18 cores = 6 performance + 12 efficiency; Node 26.9.0)

### 2.1 Where training time goes (baseline `runs/v2-h32x32-1m`)
- **Config:** `{"target":1000000,"hidden":[32,32],"actionRepeat":1,"popSize":128,"episodes":4,"sigma":0.1,"lr":0.03,"weightDecay":0.005,"checkpointEvery":5,"evalEvery":5,"seed":1,"gameVersion":"v2"}`, 18 workers.
- **Result:** held-out median 1M reached at **gen 970 after 472 s** of cumulative training (gen-to-gen wall about 489 s including validation). Held-out 24/25, with one seed crashing at 499.
- **Time by generation range:**

  | Generations | Time | Share | Frames | Throughput |
  |---|---|---|---|---|
  | 1–500 | 13.3 s | 3% | | |
  | 501–800 | 11.8 s | 3% | | |
  | 801–900 | 16.2 s | 4% | | |
  | 901–950 | 23.6 s | 5% | | |
  | **951–970** | **398.6 s** | **86%** | 2.30e9 | 5.8M fps |

  Gen 970 alone: 65 s, 390M frames.
- **The ~54-point plateau** spanned gens ~20–780, at about 25–30 ms per generation, so it cost only about 25–40 s.
- An episode to 1M is **1,943,236 frames** (integrated from the speed schedule). The 500k→1M half of the points is about 48% of the frames.
  - Frames to reach each score: 1k ≈ 4.8k, 10k ≈ 29k, 100k ≈ 229k, 500k ≈ 1.02M, 1M ≈ 1.94M.
  - Frames per point = 40/speed. That is 4.8 below 1k, then falls from 3.08 to 1.82 above 1.7k (speed 13→22).

### 2.2 Per-frame cost (single thread, shipped 44-32-32-3 model)
- **Full agent:** 0.65M frames/s, about 1,540 ns per frame:
  - MLP ≈ 80% (1,164 ns per decision, of which `Math.tanh` ≈ 175 ns);
  - core simulation ≈ 16% (core alone 4.06M fps);
  - observation ≈ 4%.
- **A C version of the forward pass** (clang -O3 -ffp-contract=off, input-major weights, same per-unit summation order): **354 ns** per decision; 174 ns with tanh removed. JS reaches about 0.6 multiply-adds per cycle (scalar f64); C vectorises.
- **Straight-line generated JS** (weights as constants) gave no gain: 1,145 vs 1,181 ns. Plain JS is at its ceiling.
- **Scaling** (6-second bursts): 1 worker 0.65M, 6 → 3.54M, 12 → 6.27M, 18 → 8.15M fps.
- **Sustained 60-second run on 18 workers: 7.40M fps.** The 8.15M figure is a short-burst artifact of thermal and power limits; an earlier claim of "27% idle" was wrong for that reason.
- **Training throughput is 6.0–6.8M fps.**
- **An A/B of 128 jobs×4 episodes against 512 single-episode jobs** (378M frames, a late-generation workload): whichever ran first was about 8% faster, in both orders. So the job layout makes no measurable difference, and run order (thermal state) dominates. Run-to-run thermal noise is about 8%.

### 2.3 `startScore` faithfulness (core.js `createGame({ startScore })`)
- A `startScore` game sets the distance, v2 rules and speed to that level's cap.
- **Natural play matches it exactly from about 2k points upward:**
  - 2,000: 13.902 vs 13.903
  - 10k: 16.000
  - 100k: 19.000
  - 500k: 21.097
  - 999k: 21.999
- **Below that, natural speed is still ramping:** at 1,500 points, 12.49 natural vs 13.53 from `startScore`. So late starts must be 0 or at least 2000.
- **The remaining differences are transient:**
  - 3 s with no obstacles (`CFG.clearTime`);
  - empty spawn history (only the "no three of a kind in a row" rule uses it);
  - an empty v2 `committed` list (airborne intervals still owed).
- Nothing else that affects gameplay accumulates over a run; night and milestones are cosmetic.

### 2.4 Hardware and toolchain
No Rust, no PyTorch, no NVIDIA GPU (Apple GPU only). `clang` is at `/usr/bin/clang`.

---

## 3. History of the work (chronological)

1. **`/init` → `AGENTS.md`**, written, then made self-contained at the user's request.
2. **First efficiency analysis** (Rust vs GPU vs algorithmic). The conclusion was that algorithmic changes beat porting:
   - **Rust:** an estimated 3× per frame, but it breaks the one-source rule and parity, and needs ports of V8's fdlibm `tanh` and `log10` to stay bit-identical.
   - **GPU:** a bad fit. The episodes are up to 1.94M sequential frames, only 512 run concurrently, the game logic is branchy, and it would need a full port.
   - **WASM SIMD for the MLP only** stays as a possible compute lever.
3. **Windowed random-start curriculum implemented** (subagent): `--window/--start-zero-frac/--start-min`, log-uniform late starts.
   - **Run `runs/exp-window-h32x32`:** `--window 50000 --start-zero-frac 0.5 --start-min 2000`, plus the baseline config.
   - Stopped at **gen 255, 466 s**: a tie with the baseline in wall time.
   - Held-out 16/25, min 614,432, max 1M; failures at 614k, 700k, 840k, 877k, 896k, 908k, 979k, 981k, 983k.
   - Validation min at the stop was 583,019. Training frames: 2.574e9 (baseline 2.706e9). About 65 s were spent outside generations (validation), against 9 s for the baseline.
4. **Three parallel research agents:** literature (Opus), first-principles derivations (Opus) and independent analysis (Fable). The main session relayed findings between them. Results are in section 4.
5. **Implementation of W1–W6 by the `implementor` session**, from a detailed brief, then Runs A and B. Results are in sections 5–6. Run C was cancelled.
6. **A statistical analysis agent (Fable)** with a Python venv. Its results are in section 8.

---

## 4. Research findings (synthesised; labels: [M] measured, [D] derived, [P] proven, [C] conjecture, [L] literature)

### 4.1 Bugs found in the original trainer (all fixed behind flags or by default; see section 5)
1. **[P] Frozen-best deadlock.**
   - `validate()` in `trainer/cli.js` replaced best only if `s.median > st.best.valMedian`. Scores are capped at the target, so once the best median equals the target, best is frozen forever.
   - The stop check then re-scored the same `st.best.params` on the same 25 `EVAL_SEEDS`, with a deterministic, identical result.
   - So if the first held-out check failed, the run could never stop or improve its selection; it burned about 8 s per check until the time budget ran out. Neither run hit it, because both passed on the first try.
   - Consequence: there was effectively one held-out test per run, after a winner's-curse selection on a saturated statistic.
2. **[P] Index tie-breaking.**
   - `es.js centredRanks` broke ties by index (`xs[a] - xs[b] || a - b`), and antithetic pairs sit at (2k, 2k+1). So a pair whose two members score the same gets w = −1/(n−1) instead of 0.
   - In an all-tied generation the update is the pure-noise vector −Σε_k/(nσ(n−1)), which Adam normalises into full-size steps.
   - **[M] Tied pairs during the baseline plateau**, out of 64 (exact replay):

     | Generations | Mean tied pairs | Mean distinct fitness values (of 128) |
     |---|---|---|
     | 1–25 | 14.7 | 45.6 |
     | 26–100 | 43.7 | 13.3 |
     | 101–200 | 60.0 (all 128 tied in 18 generations) | 4.5 |
     | 201–400 | 33.7 | |
     | 401–600 | 4.4 | |
     | 601–780 | about 0 | |

   - ‖θ‖ grew from 8.3 to 38, consistent with an Adam random walk of per-coordinate RMS ≈ lr·√T (0.68 measured at T=600 against 0.73 predicted; 0.03·√(800·2595) ≈ 43).
3. **[P] Inert weight decay.** `es.js adamStep` used `g = grad − wd·θ` inside the Adam-normalised gradient (coupled L2), so it is effectively inert (Loshchilov & Hutter 2019).
4. **[L] Both 2 and 3 are inherited** from OpenAI's evolution-strategies-starter (`es_distributed/es.py`: argsort ranks, `l2coeff·θ` added before Adam). That repo also has `episode_cutoff_mode: adaptive:<limit>,<threshold>,<ratio>`, which multiplies the episode cap by `ratio` once the fraction of rollouts hitting it reaches `threshold`; evaluation ignores the cap. This is precedent for adaptive episode caps.
5. **[M] Validation idle time.** 25 validation jobs on 18 workers run in two waves, so the machine is about 25–31% idle during each pass. The held-out eval ran 46.6M frames in 8.47 s (≈ 5.5M fps against 7.4M sustained).
6. **[M] Validation cadence.**
   - In the baseline, fitMax first reached 1M at gen 967, but with eval-every 5 no validation ran until gen 970. Gens 966–970 cost 236 s (genMs 28, 42, 50, 51, 65 s), and the validation at gen 965 had median 484k, min 250k.
   - If the θ at gen 967 or 968 already passed, up to about 50–165 s was wasted. This is **unverified**: only the gen-970 checkpoint survives, so it needs the exact replay to check.
7. **[M] Validation cost:**
   - curriculum run: 81 s of 483 s (17%), measured from metrics.jsonl timestamp gaps (row time − previous row time − genMs);
   - baseline: about 25 s of 489 s (5%);
   - the final validation plus held-out pass cost about 17 s in both.
8. **[M] Feature ranges.** The speed feature (out[4]) reaches 2.29 in v2, and obstacle velocity 1.82, because both are normalised by v1's `CFG.maxSpeed` of 13. These are the only features with |x| > 2. Fixing it means editing `policy.js`, i.e. a core-hash change, so it is only possible for fresh runs. Low priority.

### 4.2 The 54-point plateau
- **[M] Mechanism: an obstacle-blind "bunny-hop" limit cycle.**
  - θ at gens 50–300 presses jump whenever grounded (y=98) and releases near the apex, with a period of about 35 frames, regardless of obstacles. It clears obstacles by phase luck and dies when one arrives during landing.
  - Seeds score 54, 61, 65 and 48. Actions are 0.6–0.77 none, 0.23–0.39 jump, about 0 duck.
  - Perturbations change some actions in 64 of 64 cases, mostly the release frame, but often not the crash frame. That is the source of the ties.
  - Hidden saturation is low at this stage (|pre|>2 in 2–13% of units). The argmax margin is large (median 3–4.6).
- 40.475 is the score of dying at the first obstacle (every constant policy scores that); 54 is dying at the second. A second, minor wall sits near 580 points (gens 835–935, the first pterodactyls at speed 8.5).
- **[M] It is a search problem, not a skill wall.** A one-parameter hand policy ("hold jump while the nearest obstacle's relx < 0.15") scores a median of 248 on the validation seeds.
- **[M] Initialisation:** the init policy is 52% none and 47% duck, and ducking blocks `startJump`.
- **[M] Counterfactual exact replays**, same seed 1, 900 generations each; the baseline escaped at about 775–800:
  - midranks only, coupled L2: **no escape by 900**. 56–61 of 64 pairs tied at gens 300–600; ‖θ‖ 21.9 at gen 400.
  - midranks plus decoupled decay (θ *= 1 − lr·0.1): ‖θ‖ bounded at 17–19 and ties about 0 from gen 200, but escape at about the **same time** (fitMean 252 at gen 800).
  - Conclusion: the index-tie random walk was the exploration that escaped the plateau. Fixing ties without an explicit exploration mechanism can make the plateau longer.
- **[M] Adam state:** mean |m|/√v in the saved checkpoints is 0.22 (baseline) and 0.31 (curriculum). Steps were about lr per coordinate almost regardless of signal.

### 4.3 Hazard (crash rate per point) measurements
- **Centre models**, fixed-start 50k windows (10k windows from 960k up), 40–320 windows per level:
  - **Baseline best:**
    - from 0: **11/240 crash (4.6%)**, all between 475 and 633 points. This is the same failure mode as the held-out seed at 499 (the first pterodactyl at speed ~8.5).
    - 0 crashes from 2k to 900k;
    - 3/120 from 950k, 1/80 from 960k.
    - An earlier "5/40 from 0" was high-side noise.
  - **Curriculum best:**
    - 0/240 from 0, so the 50% share of zero starts fixed the early game;
    - 0 from 2k to 550k;
    - 1–4 per 80 windows in 600k–950k;
    - **0.5–0.9 crashes per 100k points from 960k up** (23 crashes in 320 10k-windows started at 960k, 970k, 980k, 990k): 6–20× any other band.
    - core.js has no discrete rule change near 960k.
  - **Calibration:** integrating the windowed hazards predicted P(reach 1M) ≈ 0.68 (curriculum) and 0.93 (baseline), against observed 16/25 = 0.64 and 24/25.
- **[M] Population hazard**, i.e. the σ = 0.1 perturbed copies that training actually scores (64 antithetic pairs, 20k-point windows, 2 seeds per level, exponential MLE, about 17 s per model):

  | Start level | Curriculum θ, per point | Baseline θ, per point |
  |---|---|---|
  | 0 | 71% crash within 20k, mostly at about 70 points | 26% crash |
  | 2k | 1.3e-5 | 1.4e-6 |
  | 2k–100k | 1.0–1.3e-5 | up to 5.5e-6 at 50k |
  | 300k | 1.5e-5 | 1.3e-5 |
  | 500k | 1.9e-5 | 2.0e-5 |
  | 700k | 2.2e-5 | |
  | 900k | 2.75e-5 | 3.1e-5 |
  | 980k | 3.0e-5 | 3.9e-5 |

  - The centres crash essentially only at 950k–1M (curriculum) or at the start (baseline): about 100× lower than the population.
- **[M] A refined population hazard**, from full episodes of 512 σ-perturbations of the baseline's gen-970 θ (statistics agent):

  | Bin | At risk | Deaths | Hazard per 100k points [95%] |
  |---|---|---|---|
  | 0–1k | 512 | 175 | 47.2 [40.5, 54.4] (an atom, P(survive) 0.62) |
  | 1k–30k | 337 | 5 | 0.05–0.11 |
  | 30k–100k | 332 | 52 | 0.255 [0.19, 0.33] |
  | 100k–200k | 280 | 39 | 0.152 [0.11, 0.20] |
  | 200k–300k | 241 | 61 | 0.286 [0.22, 0.36] |
  | 300k–500k | 180 | 33 | 0.103 [0.07, 0.14] |
  | 500k–700k | 147 | 18 | 0.069 [0.04, 0.11] |
  | 700k–1M | 129 | 14 | 0.039 [0.02, 0.06] |

  - So the population is a mixture: 34% die in the opening, and survivors have a hazard of about 1–3e-6 per point, peaking at 30k–300k (4-cactus groups from 31.6k, the speed ramp) and *lower* late.
  - Candidates differ wildly: frailty by candidate ρ = 0.85 (beta-binomial LRT χ² = 291). No frailty by seed for the population (χ² = 0.93, p = 0.82).
- **[M] One full generation around the final baseline θ** (128 candidates × 4 seeds to 1M, 348M frames, 67 s):
  - episode scores are bimodal: q25 = 216, median 154k, 115/512 reach 1M;
  - frames by candidate quartile, worst to best: 0.2 / 8.1 / 24.8 / 66.9%;
  - successive halving that keeps the top 50% past 100k still costs 91% of the frames (cos 0.90 with the full gradient); keeping the top 12.5% costs 40% (cos 0.71);
  - at equal frames, a cap of 300k on 2 seeds aligned with full-game gradients at cos 0.607, against 0.563 for full games on 1 seed (noise floor 0.66; weak evidence).
- **[M] The curriculum run's generation 256, exact replay** (fitMedian 0.5009 vs the logged 0.501):
  - starts were [0, 0, 100190, 6187];
  - pass counts of 128 per episode: 35, 34, **93, 99**;
  - joint patterns: ffPP 53, PPPP 27, ffff 25, fffP 7, PfPP 5, fPPP 4, ffPf 4, PPfP 3;
  - the median candidate is ffPP: dies at about 70 points on both start-0 episodes and saturates both late windows;
  - 80/128 candidates (62%) had both late windows at 1.0;
  - the centre θ passes all four episodes, so this is σ damage.
  - (One agent's summary stated "23–27% of late windows saturated"; that is inverted. The correct figure is 73–77% saturated.)
- **[M] σ sweep:** perturbed copies (64 candidates × 6 seeds) surviving from 0 to 2k:

  | σ | Curriculum θ | Baseline θ |
  |---|---|---|
  | 0.1 | 39% | 85% (crashes mostly below 100 points) |
  | 0.05 | 86% | 99.7% |
  | 0.025 | 96% | 100% |
  | 0.0125 | 98% | |

  For the curriculum θ at 0.0125, the residual crashes are at 464 points, the first pterodactyl on one seed. So J_σ(θ), the objective ES actually optimises, sits far below F(θ) late in training.
- **[M] Weights:**
  - final weight RMS is about 0.69–0.77 in every layer regardless of fan-in in the baseline, and 0.45–0.50 in the curriculum model, against an init of 0.15–0.18;
  - |pre| > 2 for 48% and 58% of hidden units in the baseline, 18% and 26% in the curriculum model;
  - so Adam plus one absolute σ ignores fan-in: σ and lr don't transfer across widths.

### 4.4 Derived results (first-principles agent; verified numerically)
- **[P] Survival:** P(reach S) = exp(−∫₀^S h) for Poisson crashes in score. With per-seed heterogeneity, P = E[e^{−Λ_seed}] ≥ e^{−EΛ}.
- **[D] Optimal start distribution for estimating Λ:** for frame density b(s), Var = ∫h·f/b, so b* ∝ √(h·f) and the start density is q* ∝ √(h/f). This is not ∝ h and not the evaluation's visitation density.
  - For the ES gradient the right weight is √(discordant-crash rate × f), where a discordant crash is one where exactly one member of the pair crashes on the same (seed, start). It reduces to √h only if the pair outcomes are independent.
- **[P] Log-uniform coverage:**
  - coverage of point x is 0.5·1[x<50k] + 0.5·(F(x)−F(x−50k)), which is about 4060/x above 50k;
  - 79% of training exposure landed in 0–100k (10% of the evaluation's points), 0.53% in 900–960k and 0.14% in 960k–1M;
  - decoded from the RNG (verified against checkpoint.rngState), the run's 510 late windows had 83 starting ≥ 500k, 24 ≥ 800k and **6 ≥ 910k**, the only ones that reach ≥ 960k.
- **[D] Variance × budget:**
  - Under the centre hazard: natural full games 1.04e6; log-uniform curriculum 7.9e7 (76× worse); optimal 3.4e5 (3.1× better than natural). With a hazard floor of 1e-8 or 1e-7 the optimal gain shrinks to 2.4× or 1.6×.
  - **Under the population hazard:** the optimum is 17–22× better than the run's log-uniform scheme, but **uniform-in-points starts are within 5–10% of optimal**. The optimal exposure puts 0.4–0.6% in 0–2k, 5–7% in 2k–100k, 34–35% in 100k–500k and 57–60% in 500k–1M.
  - Conclusion: use starts uniform in points up to the target, with the last window truncated at the target; the √h tilt is second order.
- **[D] Saturation signal** (latent Gaussian quality, top fraction p tied, midranks):
  S(p)/S(0) = 2√π[∫₀^{1−p}(u−½)Φ⁻¹(u)du + (1−p)/2·φ(Φ⁻¹(1−p))]

  | p | Analytic | Simulated | SNR, midranks | SNR, index ties |
  |---|---|---|---|---|
  | 0 | | | 4.51 | |
  | 0.5 | 0.854 | 0.856 | 4.18 | |
  | 0.9 | 0.315 | 0.317 | 2.90 | 2.57 |
  | 0.98 | 0.086 | 0.097 | 1.80 | |

  At p = 1, index ties give a pure-noise update (0.063 RMS) and midranks give exactly 0.
- **[D] Window rule:**
  - one window has P(crash) = ½ at w = ln2/h, which is about 35k points at h ≈ 2e-5; all 4 windows saturate with P = ½ at w ≈ 8.7k;
  - progress per frame ∝ √x/(x+x₀), maximised when rollout frames per generation equal the fixed per-generation overhead;
  - Fisher information per point about ln h is h, independent of w, so window length is not a per-frame information lever;
  - what matters is rank saturation (p = pass^E for E late windows) and generations per second.

  Saturation at a given pass rate for E = 1/2/4/8 late windows:

  | Pass rate | E = 1 | E = 2 | E = 4 | E = 8 |
  |---|---|---|---|---|
  | 0.8 | 0.51 | 0.73 | 0.91 | 0.99 |
  | 0.9 | 0.31 | 0.50 | 0.71 | 0.90 |

  - Recommended band: pass^E ≈ 0.3–0.5, i.e. a per-window pass rate of about 0.6–0.8 at E = 4 and 0.75–0.9 at E = 8.
  - With h = 2.5e-6 per point and a 40k-frame overhead per episode: the best windows are about 270k/100k/46k/22k for E = 1/2/4/8, giving 1.1–4.2× the progress per second of full games, flat within 20% over a 3× window range.
- **[D] Pair-difference ranking:** with per-pair (seed, start) sets, global ranks weight pair k by about (F₊−F₋)·density_F(level), so pairs at unusual levels get almost no weight. Ranking F₊−F₋ (ARS-style) cancels the level term, so it is *required* if levels vary per pair. With shared levels, as by default, the two are equivalent to first order.
- **[P] Median stop rule:** it passes iff Bin(25,p) ≥ 13.
  - p = 0.4 → 0.154; p = 0.5 → 0.500 (both checks together 0.25); p = 0.64 → 0.926 (both 0.857).
  - So the curriculum stopping at 16/25 was the *expected* outcome.
- **[P] Certification:** zero failures in N fresh games certifies failure rate ≤ ε at level α iff N ≥ ln(1/α)/ln(1/(1−ε)) (≈ ln(1/α)/ε).

  | ε | α | Games | Frames | Time |
  |---|---|---|---|---|
  | 1% | 0.05 | 299 | 580M | 89 s |
  | 2% | 0.05 | 149 | 289M | 44 s |
  | 5% | 0.05 | 59 | 114M | 18 s |
  | 1% | 0.01 | 459 | 890M | 137 s |

  - Clopper–Pearson: 0/25 failures → fail rate ≤ 0.113 (one-sided); 24/25 → ≤ 0.176; 16/25 → p ∈ [0.425, 0.82].
  - **Contaminated stop seeds:** if K different best-θs are tested on the same 25 seeds and the passing one is reported, α_eff = K(1−ε)^25. 25/25 then rules out only fail rates above 11%/15%/17% for K = 1/3/10.
  - **Minimax:** an adversary puts hazard where exposure is thinnest, so uniform per-point exposure (full games) is optimal for certification. Windows help training, not certification.
  - **Bayesian (uniform prior):** 25/25 → P(p≥0.95)=0.74, P(p≥0.99)=0.23; 60/60 → 0.96/0.46; 150/150 → 1.00/0.78; 299/300 → P(p≥0.98)=0.98.
  - SPRT (5% vs 1%, α=β=0.05) accepts after 72 straight passes, so it saves nothing for good models; early rejection is already in `sequentialAllPass`.
- **[P] Compute ceilings:**
  - Amdahl: an infinitely fast MLP gives at most 5× (MLP ≈ 80%).
  - `actionRepeat` k ≤ 2 is forced, because v2's `minWindowFrames`/`minGroundFrames` lerp down to 2 at d = 1, so at most 1/(0.2+0.4) ≈ 1.67×. Risky, since the solver guarantee assumes 1-frame control. Not recommended.
- **[P] Early termination:**
  - Exact stopping of training candidates is possible only for the worst ones: ≤ 1% saving (they already die early). Approximate stopping saves ≤ 8%.
  - Lowering the cap is not an approximation; it is a different objective.
  - Early stopping is exact and worthwhile only in validation and the stop test: `median > best` is decided once 13 seeds fall on one side, and all-pass is decided at the first failure.
- **[D] Run B's failure, stated generally:** the full-game pass is a *product* over stages, ln P = Σ ln P_k. An arithmetic mean of window progresses can be ≈ 1 while the opening factor is 0. Gated alternatives with provable properties are in section 9.

### 4.5 Literature checked (key items; titles and authors verified by web search)
- Kakade & Langford, "Approximately Optimal Approximate RL", ICML 2002: restart distributions; the bound depends on how well the restart distribution covers the optimal policy's states. https://mlanthology.org/icml/2002/kakade2002icml-approximately/
- Sutton & Barto, *RL: An Introduction*, 2nd ed., 2018: Monte Carlo exploring starts.
- Peng et al., "DeepMimic", ACM TOG 2018: reference state initialisation and early termination are key components; without RSI a backflip policy learns only a hop. https://arxiv.org/abs/1804.02717
- Salimans & Chen, "Learning Montezuma's Revenge from a Single Demonstration", 2018: starts from demo states, moving backward. https://arxiv.org/abs/1812.03381
- Florensa et al., "Reverse Curriculum Generation for RL", CoRL 2017. https://arxiv.org/abs/1707.05300
- Resnick et al., "Backplay", 2018. https://arxiv.org/abs/1807.06919
- Ecoffet et al., Go-Explore, 2019 (https://arxiv.org/abs/1901.10995) and Nature 2021.
- Jiang et al., "Prioritized Level Replay", ICML 2021 (https://arxiv.org/abs/2010.03934); Robust PLR, NeurIPS 2021; PAIRED, NeurIPS 2020 (https://arxiv.org/abs/2012.02096); Goal-GAN, ICML 2018; ALP-GMM, CoRL 2019; Teacher-Student CL (TNNLS 2020).
- Salimans et al., "Evolution Strategies as a Scalable Alternative to RL", 2017 (https://arxiv.org/abs/1703.03864), plus the reference code https://github.com/openai/evolution-strategies-starter
- Mania, Guy & Recht, ARS, 2018: pair-difference weights, top-b directions, reward-std normalisation. https://arxiv.org/abs/1803.07055
- Wierstra et al., NES, JMLR 2014; Hansen, CMA-ES tutorial (https://arxiv.org/abs/1604.00772); Ros & Hansen, sep-CMA, PPSN 2008; Auger & Hansen, IPOP restarts, CEC 2005; Brockhoff et al., mirrored sampling, PPSN 2010.
- Heidrich-Meisner & Igel, Hoeffding/Bernstein races, ICML 2009; UH-CMA-ES, IEEE TEC 2009; Jamieson & Talwalkar, successive halving, AISTATS 2016.
- Ng & Jordan, PEGASUS, UAI 2000 (fixed random numbers / CRN).
- Loshchilov & Hutter, "Decoupled Weight Decay", ICLR 2019. https://arxiv.org/abs/1711.05101
- Lehman et al., Safe Mutations, GECCO 2018 (https://arxiv.org/abs/1712.06563); µP / Tensor Programs V (https://arxiv.org/abs/2203.03466). Applying either to ES is an inference.
- Conti et al., NS-ES / NSR-ES, NeurIPS 2018; Lehman & Stanley, novelty search, 2011; Mouret & Clune, MAP-Elites, 2015; Müller & Glasmachers, PPSN 2018.
- EvoJAX (https://arxiv.org/abs/2202.05008), evosax (https://arxiv.org/abs/2212.04180), gymnax. These would only pay off with short episodes (≤ 10⁴–10⁵ frames), ≥ 10⁴ concurrent environments, a frame-exact port of `core.js`, and a supported accelerator.
- Hanley & Lippman-Hand, the rule of three, JAMA 1983.

---

## 5. Code changes now in the working tree (uncommitted)

`git status`: modified `.gitignore`, `trainer/{cli,dashboard,env,es,ops,worker}.js`, `trainer/dashboard.html`; untracked `AGENTS.md`, `HANDOFF.md`, `test/curriculum.test.js`, `test/trainer-rules.test.js`, `trainer/rules.js`; deleted `TrainingPrompt.md` (pre-existing, leave it). About 420 insertions.

### 5.1 Files
- `trainer/es.js`: `sigmaVector`; `perturb` takes a scalar or vector σ; `centredRanks(xs, ties)`; `pairDiffWeights`; `esGradient(..., opts)`; `adamStep(..., decay)`.
- `trainer/env.js`: `validationSeeds(n)`, `evalSeeds(n)` (the first 25 are identical to `VALIDATION_SEEDS`/`EVAL_SEEDS`); `startScores` with `startDist`; `episodeCaps`; `windowFitness` with a span. Also the original curriculum helpers (`START_SALT`, a hash-based `uniform`, stratified zero starts).
- `trainer/worker.js`: jobs carry absolute `caps[j]`; `setup.sigma` is a number or a vector.
- `trainer/rules.js` (new): pure loop policies: `drawGeneration`, `valSummary`, `isBetter`, `parseStopRule`, `alreadyTested`, `wantHeldOut`, `sequentialAllPass`, `tiedPairs`, `stallUpdate`, `annealUpdate`, `adaptWindow`, `splitFolded`, `clopperPearson`.
- `trainer/cli.js`: flags, DEFAULTS and backfill, `checkConfig`/`checkCurriculum`, the train loop, `eval --seeds N --seed-offset K` (prints the pass count and a two-sided 95% Clopper–Pearson interval; records seedOffset, passed and passCI95 in evals.jsonl). With offset 0, the eval header keeps the old wording, which a smoke check depends on.
- `trainer/ops.js`, `trainer/dashboard.js`, `trainer/dashboard.html`: curriculum fitness labelled as 0–1 progress, not points. Dashboard thinning skips folded rows.
- `test/curriculum.test.js` (from the first curriculum subagent) and `test/trainer-rules.test.js` (18 tests).
- `.gitignore`: adds `.venv/`.

### 5.2 Flags (all are `train` config keys, saved in the checkpoint and backfilled from DEFAULTS on resume)

| Flag | Default | Semantics | Resume |
|---|---|---|---|
| `--window P` | 0 (off) | curriculum window in points | changeable |
| `--start-zero-frac F` | 0.5 | share of episodes starting at 0 (stratified) | changeable |
| `--start-min P` | 2000 | lowest late start; must be ≥ 2000 | changeable |
| `--start-dist log\|points` | log | log: log-uniform on [startMin, target−window]. points: uniform on [startMin, target) (clamped to target−1), window clipped to target, span = target − start | changeable |
| `--start-zero-cap P` | = window when first set | start-0 episodes end at P; fitness denominator P | stored resolved |
| `--window-adapt` | off | late-window pass rate > 60% → window ×1.5; < 30% → /1.5; clamp [5000, target]; requires `points`. **Flawed, see section 6.3** | `st.window` checkpointed; explicit `--window` overrides |
| `--select median\|count` | median | count: lexicographic (seeds ≥ target, mean ln(1+score)); at full count a tie also replaces best (newest all-pass θ), otherwise count+allpass would freeze too | median→count: the next validation replaces the old best |
| `--stop-rule median\|allpass[:N]` | median | allpass:N: held-out check when every validation seed passes; held-out indices 0..N−1 played in rounds of `--workers`, abort after the first round with a failure; stop when all N pass | changeable |
| (no flag) no-retest guard | **always on** | never re-test a best already tested at the same (best generation, target, N). Deterministic, so it can't change outcomes; it removes the deadlock. Old heldOut records without a seeds field are retested once | — |
| `--val-seeds N` | 25 | validation set = `validationSeeds(N)` | **don't change under `count`**: counts are raw |
| `--eval-when-fit-max` | off | also validate when fitMax ≥ target (curriculum: a candidate passed a window ending at the target) | ignored under fold |
| `--fold-validation` | off | validation games run inside the generation's pool batch on the pre-update θ (labelled gen g−1, stored as best with those params); validates every generation; `valFrames` recorded separately | `--eval-every` and `--eval-when-fit-max` ignored |
| `--rank-ties index\|mid` | index | mid: average ranks, so a tied pair has w = 0 | changeable |
| `--decay coupled\|decoupled` | coupled | decoupled: θ *= 1 − lr·wd after the step. `--weight-decay` means different things per mode (~0.005 vs ~0.01–0.1) | changeable |
| `--stall-rule` | off | tiedFrac > 0.5 → skip the Adam step (θ and Adam incl. t unchanged; RNG consumed normally); next σ_eff ×1.5, max 4×; otherwise reset to 1×. **Flawed, see section 6.1** | `stallMult` checkpointed |
| `--sigma-scale none\|fanin` | none | fanin: layer-l weights σ·√(layers[0]/fanIn_l); biases σ. Used identically in perturb and esGradient | Adam `v` goes stale if switched mid-run |
| `--sigma-anneal`, `--sigma-min` | off, 0.02 | > 25% of start-0 candidate episodes below 200 points AND the latest validation min ≥ 200 → σ ×0.8 (floor), cooldown 10 generations | `st.sigma` checkpointed; explicit `--sigma` overrides |
| `--pair-episodes` | off | per-pair episode seeds and starts from the trainer RNG; w_k = D_k/std(all F); normaliser stays fitness.length | changes the RNG stream from that generation |
| `--no-<flag>` | | turns a boolean off on resume | |
| `eval --seeds N --seed-offset K` | 25, 0 | held-out indices K..K+N−1 | |

- Locked on resume (unchanged): `hidden`, `actionRepeat`, `gameVersion`.
- New checkpoint state: `sigma`, `stallMult`, `annealedAt`, `lastValMin`, `window`.
- New metric columns: valCount, valMeanLog, valGeneration, valFrames, bestValCount, tiedPairs, stepSkipped, sigmaEff, window.
- `frames` stays training-only; `fps` = (frames + valFrames)/genMs. Resumed runs keep their metrics.csv header.

### 5.3 Verification already done (by the implementor)
- **Bit-identity, all flags at defaults**, old-code snapshot against new code:
  - plain 20 generations plus resume to 40; log curriculum 20 plus resume to 40; two low-target runs that stop through the held-out path;
  - compared: every old metrics field (except time, genMs, fps, cumulativeMs), plus checkpoint theta, adam, rngState, best, heldOut, lastValidated, generation, reason, config, and best.json;
  - result: **IDENTICAL, 9,726 values**. Re-run after every edit.
- **Old-code gen-20 checkpoints resumed with the new code:** identical, 4,382 values.
- **Real checkpoints** (`v2-h32x32-1m` gen 970, `exp-window-h32x32` gen 255), scratch copies resumed one generation with old and new code: identical, 20,848 values.
- **New-flag determinism:** Run B/C flag sets, 15 generations plus resume to 30, against 30 straight: identical.
- **Tests:** `node --test test/` 55/55 (37 old plus 18 new), re-checked at handoff. `node test/smoke-trainer.js` 17/17.
- **Not run:** `node test/browser-check.js` (the browser path and model.js are untouched). Re-run it before committing anyway.

---

## 6. Experiment results

All runs: `--hidden 32,32 --target 1000000 --seed 1 --game v2 --pop 128 --sigma 0.1 --lr 0.03 --checkpoint-every 5`, 18 workers, run with `train` (no auto-export).

### 6.1 Stall-rule quick check
Mid ranks plus decoupled wd 0.1, seed 1. Metric: the first generation with fitMean > 200. Stopped at escape or gen 900.

| Shape | Stall | Escape | Skipped steps | Max σ_eff | Wall |
|---|---|---|---|---|---|
| 32,32 | on | 252 | 9/303 | 0.3375 | 34 s |
| 32,32 | off | 776 | 0 | 0.1 | 30 s |
| 32,32,32 | on | none ≤ 900 | 604/900 (45–63 of 64 tied at gens 500–900) | 0.3375 | 36 s |
| 32,32,32 | off | none ≤ 900 (ties fell to 1–6; ‖θ‖ grew to 27) | 0 | 0.1 | 39 s |

- On 32,32 the rule fired only in gens 1–27, so the 776→252 gain is **probably chaos, not the treatment** (n = 1).
- The statistics agent's extra replays on 32,32:
  - with the stall rule: 252 (seed 1), > 300 (seed 2), > 300 (seed 3);
  - without it: 776, > 300, > 300.
- **On the freeze:** on plateau snapshots of *both* shapes, σ×1.5–2.25 cuts the tie fraction from 0.72–0.97 to ≤ 0.3, so inflating σ does untie both. The 3×32 freeze came from the *skipped steps* keeping θ fixed.
- **Tie fraction at σ multipliers 1 / 1.5 / 2.25 / 3.375:**

  | Snapshot | ×1 | ×1.5 | ×2.25 | ×3.375 |
  |---|---|---|---|---|
  | 2×32 baseline path, g100 | 0.86 | 0.44 | 0.05 | 0.08 |
  | 2×32 AdamW, g100 | 0.20 | 0.07 | 0.05 | 0.10 |
  | 3×32 AdamW, g100 | 0.83 | 0.24 | 0.04 | 0.15 |
  | 3×32 AdamW, g200 | 0.72 | 0.23 | 0.03 | 0.17 |
  | 3×32 stall-rule θ, g200 | 0.97 | 0.82 | 0.29 | 0.03 |

  - A floor of about 0.1 remains at any σ: both members die at the same first obstacle.
  - The argmax-margin ratio is the same for both shapes (medians 2.5–3.4), so "margin scales with depth" is not the mechanism.
  - The stall rule as specified discards 56–64% of steps on 3 of 4 trajectories, in streaks of ≤ 4. Run A's 9 skips was the outlier.

### 6.2 Run A: `runs/exp-A-hygiene-h32x32`
- **Flags:** `--select count --stop-rule allpass:25 --val-seeds 36 --fold-validation --rank-ties mid --decay decoupled --weight-decay 0.1 --stall-rule --eval-every 5 --time-budget 20` (eval-every is ignored under fold).
- **Result:** hit the 20-minute budget. 430 generations / 1200 s, 7.41e9 frames (5.88e9 training). Validation was 20.6% of frames. **0 held-out attempts** (never all-pass).
- **Best:** θ of gen 429, validation 20/36, median 1M, min 156,926.
- **Validation trajectory** (θ gen: count/36, median): 400: 0, 164k → 422: 1, 434k → 427: 9, 910k → 429: 20, 1M. It was still climbing steeply when the budget ran out.
- **Plateau escape at gen 252.** The exact replay predicted exactly that. Stall skips at gens 3, 4, 6, 7, 8, 11, 13, 23, 26; σ_eff max 0.3375 at gen 9; never after gen 27; tiedPairs < 2/64 from gen 51.
- **The climb was 3–6× slower** than the baseline. Two sets of thresholds were measured and not reconciled:

  | Phase | Baseline | Run A |
  |---|---|---|
  | fitMean 200→2k | 145 g / 28 s | 34 g / 17 s |
  | fitMean 2k→20k (statistics agent) | 8 g / 20 s | 51 g / 234 s |
  | fitMean 20k→200k (statistics agent) | 14 g / 184 s | 84 g / 1056 s |
  | fitMean 2k→20k (implementor) | 8 g / 25 s | 33 g / 59 s |
  | valMedian 20k→200k (implementor) | 10 g / 139 s* | 90 g / 713 s |
  | valMedian 200k→1M (implementor) | 5 g / 241 s* | 18 g / 405 s |

  \* The baseline validated every 5 generations, so these are coarse. Run A's fitMean never reached 200k.
  - Frames per generation are equal at matched fitMean (12M; 58–75M), so this is more generations, not slower ones.
  - Change points (binary segmentation, BIC) in ln fitMean: baseline 603, 780, 810, 935, 946, 957; Run A 99, 206, 271, 293, 312, 349.
- **Leading hypothesis for the slow climb:** `--decay decoupled --weight-decay 0.1` pinned ‖θ‖ at 16.7–17.3 for the whole run (the baseline climbed at ‖θ‖ = 38). Each Adam step is then about a 9% relative move, against 4% at 38, and the decay removes 0.3% of θ per step. **Unverified (n = 1, chaotic).**
- **valMin was 441–479 in 414 of 426 generations:** one validation seed dies at the 475–633 pterodactyl. Count selection never saw more than 9/36 until the last few generations.
- **Certification:** fresh held-out 25..99: **46/75 [0.494, 0.724]**. Stop seeds 0..24: 13/25 (information only).

### 6.3 Run B: `runs/exp-B-curric-h32x32` — failed by design flaw
- **Flags:** A's flags but `--weight-decay 0.01` (a deliberate deviation from the brief, because of A's climb), plus `--window 25000 --episodes 8 --start-dist points --start-zero-frac 0.25 --start-zero-cap 5000 --start-min 2000 --window-adapt --sigma-scale fanin --sigma-anneal --sigma-min 0.02`. Fold kept.
- **Result:** budget reached at 06:55Z. 171 generations / 1200 s, 7.17e9 frames; validation 0.1% of frames (every validation game died at 40–2000 points); 0 held-out attempts.
- **Best:** gen 92, 0/36, median 644.5. **Fresh 0/75 [0, 0.048]**; stop seeds 0/25.
- **Trajectory** (gen: fitMean, val median/min/max, window, cumulative s):
  - 129: 0.60, 48/40/66, 5000, 49 s
  - 137: 0.49, 97/40/676, 11250, 58 s
  - 145: 0.53, 50/40/146, 16875, 76 s
  - 153: 0.51, 77/40/210, 128150, 186 s
  - 163: 0.46, 196/41/692, 648761, 532 s
- σ annealed once, 0.1→0.08 at gen 90; never again, because it requires valMin ≥ 200. Tied pairs ≈ 0; ‖θ‖ 17→23.
- **Mechanism (a spec flaw, not a code bug):**
  - late starts (≥ 2000) skip the opening, so a centre that dies at 40 points still passes most 5k–25k late windows;
  - `--window-adapt` saw > 60% late passes and grew the window ×1.5 per generation, from 5k to 649k in about 25 generations; generations then cost about 100 s;
  - only 2 of 8 episodes start at 0, and their fitness (score/5000 ≈ 0.01) is swamped by the late windows at about 1.0, so the opening is never learned.
- Run C (B plus `--pair-episodes`) was **not run**: it inherits the flaw.

### 6.4 Comparison of every model on the same fresh seeds

| Model | Fresh held-out 25..99 | 95% CP | Stop seeds 0..24 | Stopped by | Gens / wall |
|---|---|---|---|---|---|
| **baseline `v2-h32x32-1m`** (shipped) | **71/75** | [0.869, 0.985] | 24/25 | held-out median | 970 / 472 s |
| exp-window | 56/75 | [0.633, 0.840] | 16/25 | held-out median | 255 / 466 s |
| Run A | 46/75 | [0.494, 0.724] | 13/25 | budget | 430 / 1200 s |
| Run B | 0/75 | [0, 0.048] | 0/25 | budget | 171 / 1200 s |

- The baseline and exp-window rows were evaluated with `--model` and `--target 1000000`, so nothing was appended to the committed `runs/v2-h32x32-1m/evals.jsonl`.
- **The statistics agent's own 50 fresh training-range seeds** (never in any RNG stream; the same seeds for all three models):
  - baseline 44/50 (crashes at 464, 486, 516 and **989k, 990k, 998k**: a second hazard in the last 1% of the game);
  - exp-window 37/50;
  - Run A 30/50.
- **Pooled over 125 independent seeds:** baseline **0.920 [0.863, 0.958]**, exp-window 0.744 [0.663, 0.814], Run A 0.608 [0.521, 0.690] (0.593 [0.514, 0.670] over 150, including its uncontaminated stop seeds).
- **Paired tests:**
  - baseline vs Run A: McNemar 19 vs 5 discordant, p = 0.007;
  - baseline vs exp-window: 11 vs 4, p = 0.12 on 50; on 125 seeds the difference is +0.176 [+0.086, +0.266] (vs Run A +0.312 [+0.214, +0.410]).
- **Hard seeds are shared across curriculum-trained models.** exp-window and Run A crash on the same 11 of 50 seeds (φ = 0.54, Fisher p < 0.001), at the *identical* score on 4: 783542, 824244, 829855, 973121, on seeds 822083584, 822432020, 822352830, 822249883. The baseline shares none. So `allpass:25` largely tests whether the stop set contains such a seed: 0.9^25 = 0.07 and 0.95^25 = 0.28.
- **Seed indices used so far:**
  - held-out 0..24: the stop seeds, contaminated;
  - held-out 25..99: used to compare three models, so **the next certification should start at offset 100** (then 175, 250, …). Never reuse an index set across a selection.

---

## 7. Lost analysis scripts (what they did, to rebuild if needed)

1. **Exact replay** (`proofs/replay.js`): re-ran the `cli.js` training loop from `createRng(config.seed)`, `initParams`, and the per-generation noise seeds and episode seeds, using the real pool and workers. It matched every fitMean in `runs/v2-h32x32-1m/metrics.jsonl` to 1e-9 over 800 generations (0 mismatches), in 26 s on 16 workers.
   - Variants: `research/replay_cf.js` (counterfactuals: mid ranks, AdamW) and `stats/replay_shape.js` (env `HIDDEN`, `SEED`, `STALL`, `WORKERS`).
   - **Easiest rebuild:** run `node trainer/cli.js train --generations N --dir runs/exp-replay-…` with the same flags. The trainer is deterministic for a given config and seed, and plateau generations cost about 30 ms, so 900 generations take about 30–40 s. Use a scratch run dir, not the scratchpad.
   - Replaying to gen 966–969 of the baseline costs about 2.3B frames (6–7 min). That is how to check the validation-cadence claim (4.1, point 6).
2. **Hazard tables** (`research/hazard.js`): play the policy from fixed `startScore` levels for fixed windows across many seeds, count crashes, and record the crash scores.
3. **Population hazard** (`proofs/hpop.js`): the same for σ-perturbed antithetic pairs around a `best.json`; exponential MLE per level.
4. **Gen-970 dataset** (`research/gen970.js`): one full generation (128 × 4 seeds to 1M) around a θ, with per-episode scores and frames. Used for the halving, cap and alignment analysis.
5. **Gen-256 replay** (`fable/replay_gen256.js`): rebuilt the curriculum run's generation 256 from `checkpoint.json` rngState; per-episode pass patterns.
6. **σ sweep** (`fable/sigma_early.js`): 64 perturbed candidates × 6 seeds, from 0 capped at 2k, at several σ.
7. **Ties vs σ** (`stats/ties_sigma.js`): tie fraction on plateau snapshots at σ multipliers.
8. **Fresh eval** (`stats/fresh_eval.js`): the 50 fresh training-range seeds for three models.
9. **Statistics** (`stats/q1..q6*.py`):
   - Q1 hazard and calibration: a Beta early-crash atom plus a Gamma–Poisson constant late hazard from 500k, P(pass) = (1−q_e)·e^{−λ(T−L0)}; PIT values 0.58, 0.82, 0.21, 0.79, 0.79, 0.74. L0 = 500k was chosen after seeing the crash locations, so the model wasn't blind, but the calibration tests were.
   - Q2 power, Q3 certification, Q4 ties, Q5 trade-offs, Q6 change points.
10. **Frames model** (`fable/frames_model.js`): frames per point from the speed schedule (reproduces 1,942,969 frames to 1M).

---

## 8. Statistics agent: experimental-design conclusions

- **Chaos scale:** the same seed and essentially the same algorithm gave plateau escape at 252 vs 776 from 9 early skipped steps; plateau drift slopes differ 7× between same-seed trajectories. sd(ln T_escape) ≥ 0.66.
- **Power** (two-sample t on ln T, 2× ratio, α 0.05, power 0.8): n per arm = 5 (sd 0.3), 10 (0.5), 18 (0.7), 34 (1.0).
  - With **n = 1 per arm**, a ≥ 6–7× wall-time ratio is needed for significance, and P(the truly 2× faster arm looks faster) = 0.76 at sd 0.7.
  - Pass-rate differences between *models* are cheap: evaluate on the same fresh seeds with McNemar; 125 seeds give ±0.09.
  - Between *training variants* you need ≥ 3 runs per arm (with between-run sd 0.1 and n = 3, the detectable difference is 0.29).
  - **One run per arm is informative only for ≥ 5–7× time effects or ≥ 0.4 pass-rate effects.**
- **Design:**
  - **Pair trainer seeds across arms.** They share init, noise and episodes until the first differing update, and decorrelate within about 50 generations (expect ρ ≈ 0.2–0.5). `--pair-episodes` changes the RNG draws and breaks pairing from gen 1.
  - **Two-rung racing:**
    - Rung 1 is a plateau proxy: `--generations 400 --time-budget 3`; outcome = escape generation (censored at 400) and fitMean slope; 3–5 seeds per arm.
    - Rung 2 is full 15–20-minute runs with ≥ 3 seeds per surviving arm. Primary outcome: wall time to validation count ≥ threshold (censored at the budget). Secondary: a fresh 75-seed certificate at a never-used `--seed-offset`.
  - **Adoption rule:** adopt a variant as default only if it wins the paired primary outcome on *every* seed (n = 3 → sign test p = 0.125; n = 5 → 0.031) **and** its pooled fresh pass rate is not worse (the 95% CI of the difference excludes −0.05). Otherwise keep the incumbent.
  - **Measure variance once:** 6 rung-1 seeds of the incumbent (about 10 min) give sd(ln T_escape); 3 full runs give a first between-run sd.

---

## 9. Proposed fixes (designed, NOT implemented)

1. **Stall rule: inflate σ on ties, but keep stepping.** In `trainer/rules.js` `stallUpdate`, change `skip: true` to `skip: false` in the tie branch. Optionally add a mode flag: `--stall-rule skip|inflate`, `stallUpdate(stallMult, tiedFrac, mode)` returning `skip: mode === 'skip'`.
   - **Better** (statistics agent, with a proof sketch): a **tie-rate σ controller** that replaces both `--stall-rule` and `--sigma-anneal`:
     - q_t = tied pairs / pairs;
     - ln σ_{t+1} = ln σ_t + κ(q_t − q*), clipped to [σ_min, σ_max];
     - q* = 0.2, κ = 1, σ_max = 1.0;
     - never skip the step.
   - Properties:
     - P(θ_{t+1} = θ_t) = 0, so the tie set is not absorbing (mid ranks already give tied pairs zero weight; decoupled decay moves θ even when all pairs tie);
     - by Robbins–Monro, if q(θ, σ) is non-increasing in σ down to its floor (measured: yes, floor ≈ 0.1 < q*), σ converges to the level where q = q* for any shape. That is about 1.3–2σ on the early plateau, falling back to about σ once ties vanish.
   - Needs a new `--tie-target` option.
   - Test by replay on 32,32 and 32,32,32 with seeds 1–3 for 400 generations (30–50 s each): controller vs `--stall-rule` vs neither. Adopt if no run has > 10 consecutive generations with q > 0.5 on either shape, and the median escape is not worse than the stall rule on 32,32.
   - Note: `--sigma-anneal` lowers σ, which raises ties, so it fights a tie controller.
2. **Gate window adaptation on the opening.**
   - In `cli.js`'s fitness map, add `let zeroPassed = 0, zeroTotal = 0;`, and in the `starts[j] === 0` branch, `zeroTotal++; if (f === 1) zeroPassed++;`.
   - Change `if (cur && config.windowAdapt && lateTotal > 0) st.window = adaptWindow(...)` to add `&& (zeroTotal === 0 || zeroPassed / zeroTotal >= 0.5)`.
   - Make it the default for `--window-adapt`.
   - Statistics-agent version: adapt only when the start-0 pass rate is > 0.6, shrink ×1/1.5 when it is < 0.3, and **rate-limit** to once per 5 generations using a pooled 5-generation pass rate. Proven property: the window is non-increasing while stage 0 fails, and w_t ≤ w₀·1.5^{t/5}.
3. **Rate-limit adaptation:** `&& st.generation % config.windowAdaptEvery === 0`, with a new key `windowAdaptEvery` (default 1; 5 once fix 2 is in).
4. **Gated curriculum fitness**, so the opening can't be traded for the late game. Choose one:
   - **Lexicographic ranks** (sort key (f₀, f_late) in `centredRanks`/`esGradient`). Whenever a pair differs in stage-0 fitness, its weight has the sign of the stage-0 difference, so E[ĝ·∇f₀] ≥ 0. Once stage 0 saturates, those ties get zero weight and the late fitness takes over. [proven for the weights' sign]
   - **Geometric-mean fitness:** F = (Π_j max(f_j, 1/span_j))^{1/E}. It is ≈ 0 whenever any stage is ≈ 0, and ∂F/∂f_j ∝ F/f_j, so the failing stage dominates. A one-line change. [proven]
5. **Flag values for the corrected curriculum:** `--start-zero-frac 0.5` and `--start-zero-cap 50000`, so stage 0 covers the opening *and* the 30k–300k hazard hump. With `--episodes 8` that is 4 start-0 plus 4 late episodes.
6. **Validation cost:** `--val-seeds 12 --fold-validation` with `--stop-rule allpass:59` (builds in an ε = 5% certificate on held-out 0..58).
   - Or two-stage validation, a small code change: 12 seeds every generation, and the other 24 only when ≥ 11/12 pass.
   - Note that under allpass:59, held-out 0..58 become stop seeds, so certify from offset ≥ 100 anyway.
7. **Weight decay:** keep `--decay decoupled` but at `--weight-decay 0.01` (or anneal it to 0 once fitMean > 2000). wd 0.1 is the suspected cause of Run A's slow climb.
8. **σ late in training:** smaller σ (0.05) keeps the opening for 86–99.7% of candidates. But smaller σ raises plateau ties, so σ should be *controlled* (fix 1), not fixed. `--sigma-scale fanin` is the right normalisation across shapes. The implementor's suggested trigger was σ ×0.8 when > 25% of candidates crash before 200 points in a from-0 episode the centre survives; it's already implemented as `--sigma-anneal`, but untested in a working run.
9. **Compute, later:**
   - WASM f64x2 SIMD MLP with lanes = episodes (shape-generic), keeping exact summation order (parity-safe); estimated 1.7–2× overall;
   - no f32 (it changes decisions);
   - WASM has no tanh: import `Math.tanh`, or ship the identical WASM to the browser and CSS paths;
   - ship the .wasm as base64 in a JS file to stay dependency-free;
   - gate on `test/net.test.js`;
   - batching episodes in lockstep in JS: about 1.25–1.4×.

   Not worth it: GPU, `actionRepeat 2`, early termination of training candidates.
10. **Exploration insurance for the shape sweep:** seed racing. Plateau generations cost 25 ms, so run K = 3–4 seeds concurrently with `--time-budget 1.5 --workers 6` each, then continue the leader with `--resume --workers 18`. Zero code.
11. **Don't put the four hard seeds into training.** The statistics agent suggested it, but they come from its 50-seed fresh test set, so that set would stop being fresh. If they are used, retire that set as a test set.

---

## 10. Implementer's verdict on defaults (agreed by the main session)

- **No-retest guard:** already the default. It can't change outcomes.
- **`--rank-ties mid`:** principled, but making it the default breaks reproduction of existing runs. Make it the default only together with a config-version bump.
- **`--decay decoupled` with wd 0.1:** no. Use 0.01, pending a test.
- **`--stall-rule` as built:** no. Replace it with fix 1.
- **`--select count` plus `--stop-rule allpass`:** correct semantics. Certify only on fresh offsets.
- **`--fold-validation`:** yes for full-game training.
- **Points starts, zero cap, clipped windows:** keep as options.
- **`--window-adapt`:** not until gated (fixes 2–4).
- **`--sigma-anneal`, `--sigma-scale fanin`, `--pair-episodes`:** insufficient evidence.

---

## 11. Recommended next steps (main session's proposal; the user has NOT approved them yet)

The pending question to the user:
- direct the implementation of steps 1–3 below (about 30 minutes, no long runs), then decide on step 4;
- and whether to commit the trainer changes, on a branch or main.

1. **Decay check (6 min).** Copy `runs/exp-A-hygiene-h32x32` to a new `runs/exp-A-wd001` directory, then:
   `node trainer/cli.js train --resume --dir runs/exp-A-wd001 --weight-decay 0.01 --time-budget 6`
   Compare generations per decade of fitMean against Run A's gens 312–428. ≥ 2× fewer generations means the decay was the confound; otherwise it's chaos. The implementor judged this confounded by the mid-climb starting point.
2. **Implement fixes 1 (tie controller, `--tie-target`), 2–4 (gated window adaptation and gated fitness) and optionally 6 (two-stage validation)**, behind flags, keeping bit-identity at defaults. Rebuild a replay harness (section 7.1) and test the controller on 32,32 and 32,32,32, seeds 1–3, 400 generations each.
3. **Re-verify:** bit-identity (the same method as 5.3), `node --test test/`, `node test/smoke-trainer.js`, `node test/browser-check.js`.
4. **Corrected Run B′,** 3 seeds paired against 3 seeds of the hygiene arm (6 runs, about 1.7 h). Run them sequentially, never concurrently.
   ```sh
   node trainer/cli.js train --hidden 32,32 --target 1000000 --game v2 --pop 128 --sigma 0.1 --lr 0.03 --seed S \
     --select count --stop-rule allpass:59 --val-seeds 12 --fold-validation --rank-ties mid --decay decoupled --weight-decay 0.01 \
     --window 25000 --episodes 8 --start-dist points --start-zero-frac 0.5 --start-zero-cap 50000 --start-min 2000 \
     --window-adapt --sigma-scale fanin --tie-target 0.2 --time-budget 18 --dir runs/exp-B2-h32x32-sS
   ```
   - Until the gated adaptation and the tie controller exist, run **without** `--window-adapt` (fixed 25k window), **without** `--sigma-anneal`, and **without** `--tie-target`.
   - Primary outcome: wall time to all-pass (censored at 18 min).
   - Secondary: `eval --seeds 75 --seed-offset 100`.
   - Adoption rule: section 8.
   - Don't run `--pair-episodes` until B′ is settled.
5. **Measure the variance once** (about 10 min): 6 plateau-proxy runs, `--generations 400 --time-budget 3 --seed 1..6`, with the hygiene flags.
6. **Layer-size sweep (the user's stated plan), as two-rung racing:**
   - rung 1: shapes {16; 32; 64; 32,32; 64,64; 32,32,32; 64,64,64} × seeds {1,2,3} on the plateau proxy, with `--sigma-scale fanin` and the tie controller (fallback: `--rank-ties mid --decay decoupled --weight-decay 0.01`, no stall rule); drop shapes with median escape censored at 400 or a bottom-half fitMean slope;
   - rung 2: surviving shapes × 3 seeds, full runs, `--stop-rule allpass:59`;
   - certify each at a new offset (100, then 175, 250, …);
   - report a Beta posterior on the fresh pass rate plus the time-to-count distribution.
7. **Certify whatever ships:** `node trainer/cli.js eval --dir runs/<winner> --which best --seeds 150 --seed-offset <fresh> --target 1000000` (about 45 s). Zero failures certify ε ≤ 2% at α = 0.05.
8. **Keep the shipped model meanwhile.** At 71/75 fresh (about 0.92 over 125 seeds), nothing has beaten it.

---

## 12. Commands cheat-sheet for this work

```sh
pgrep -fl trainer/cli.js                       # is a trainer running? (one at a time; 18 cores)
node --test test/                              # 55 tests
node test/smoke-trainer.js                     # 17 checks
node test/browser-check.js                     # headless Chrome; run before committing
nohup node trainer/cli.js train <flags> --dir runs/exp-<name> >> runs/exp-<name>/train.log 2>&1 &
kill -INT <pid>                                # PID in runs/<dir>/train.lock; saves the checkpoint, releases the lock
node trainer/cli.js eval --dir runs/<dir> --which best --seeds 75 --seed-offset 100 --target 1000000
node trainer/cli.js eval --model <file> --seeds 75 --seed-offset 100 --target 1000000   # doesn't append to a run's evals.jsonl
git diff trainer/                              # review the uncommitted trainer changes
```

- **Experimental runs:** use `runs/exp-*` directories (gitignored). Never use `--auto-export`, `session` or `finish`.
- **Seed offsets used so far:** held-out 0..24 (the stop seeds) and 25..99 (the three-model comparison). Next is 100.

---

# APPENDICES

## A. What the user asked for, turn by turn (2026-10-09 → 10)

1. `/model` was set to Opus 5.5. Then `/init` with args "to AGENTS.md": write the repo guide as `AGENTS.md`.
2. Mid-task: "Remove any reference to PROGRESS.md or training prompt" and "Document should be self contained." `AGENTS.md` was rewritten accordingly. The `# CLAUDE.md` header required by `/init` was kept.
3. "Investigate how training could be more efficient, such as a trainer written in Rust or using GPU acceleration libraries in Python." This was answered with profiling and benchmarks (section 2).
4. "Elaborate more on options 1 and 2. Is option 2 safe, and is skipping ahead in the difficulty curve an existing technique?"
   - Option 1 (finer job granularity) was withdrawn after measurement: thermal ordering dominated.
   - Option 2 (`startScore` windows) was checked as faithful from 2k up, with literature precedent.
5. "Dispatch a subagent to perform the suggested next step. Don't micro-optimise for a specific layer configuration; I'd like to try different layer sizes and layer counts; 32x32 is a good start." This produced the windowed-curriculum implementation and `runs/exp-window-h32x32`.
6. "Use a subagent to perform in-depth research" (literature agent, Opus).
7. "Also spawn a subagent for first-principles reasoning and logical proofs" (derivation agent, Opus).
8. `/advisor` was toggled, ending on Fable 5.1. "Use a Fable subagent to also do analysis" (independent agent, Fable).
9. "If a subagent finds something useful, it should report back up so you can notify the other subagents." The main session relayed findings between all agents from then on.
10. "Direct session 'implementor' on the implementation. Be as detailed as possible." A long brief (W1–W6, verification, Runs A/B/C) was sent to the peer session `implementor`.
11. "Use a Fable subagent to mathematically perform derivation and statistical analysis … set up a local Python venv in this repo and install numpy, etc." This created `.venv/` and the statistics agent.
12. "Dump all findings for a zero-context session to HANDOFF.md." Then "I believe you can add even more detail", which produced these appendices.

**Open questions to the user, unanswered:**
- (a) Proceed with the step 1–3 plan in section 11?
- (b) Commit the trainer changes, and on a branch or main?

## B. Project history before this work (from `PROGRESS.md`, which is committed)

- The baseline game is commit `2da34de`. The scaffolding and training sessions built the core, shared policy, browser AI mode, trainer, dashboard, CSS-network page, and `report`/`finish`/`session`.
- **v1 game (first training sessions, 2026-10-09):**
  - default 44-16-3 plateaued at about 1200;
  - 44-32-3 reached held-out median 10000 at gen 550 (1.6 min), then 99999 and 500000 on resume;
  - 24/25 at 500k, with one seed crashing at 759;
  - v1 difficulty stops rising at about 1.7k points.
- **Game v2** (commit `57125a5`): identical below 1k. Above that, level d = log10(score/1000)/3, clamped to [0, 1], interpolates:
  - speed cap 13→22;
  - gap coefficients 0.6/1.5 → 0.35/1.2;
  - minWindow/minGround 4/4 → 2/2;
  - pterodactyl share 1/3 → 1/2, with speed spread 0.8→1.6;
  - cactus groups of up to 4 from about 31.6k.

  An exact solver never crashes from 0 to 1M (`test/solver.js`). v1 runs were purged to the zip archive.
- **v2 runs:**
  - `runs/v2-h16` (44-16-3) stalled at about 1210 (gitignored);
  - `runs/v2-h32x32` (44-32-32-3, target 500k) reached held-out **25/25 at 500k**, gen 970, 15.0 min (commit `a657d1a`);
  - `runs/v2-h32x32-1m` (a fresh run, target 1M, same seed): gen 970, 7.9 min, held-out median 1M with 24/25 (commit `3177445`). **This is the shipped model.**
- The 500k and 1M runs share seed 1, so **their trajectories are identical until episodes exceed 500k**. Both show the same per-generation fitMean and validation through gen 915; see Appendix C.
- **Core hash change:** `48697bb` ("policy: compute MLP rows four at a time", bit-identical decisions, about 1.34× faster) changed `policy.js`. So `runs/v2-h32x32/best.json` has coreHash `f384c716f8d9…`, while every later run has `0f329d4e3493…`; the current full hash is `0f329d4e3493f794daf9cf2d013bd7927b51ff848efed768922a10b582c7410e`. Resuming the 500k run needs `--force`, which is safe because decisions are identical.
- The last commits before this work:
  - `6ab5041` CSS page: drop the in-page JavaScript cross-check
  - `65b86fe` CSS-network page
  - `0c93159` PROGRESS v2 results
  - `3177445` the 1M run
  - `28db13c` dashboard speeds

## C. Run trajectories (sampled validation rows from each `metrics.jsonl`)

Columns: gen | cumulative s | fitMean | fitMax | validation median/min/max | valCount | tiedPairs | σ_eff | window | ‖θ‖. Curriculum runs have fitness in 0–1.

**`runs/v2-h32x32`** (500k target): 970 rows; cumulative 897 s; training frames 2.335e9; ran 01:07:39Z–01:22:53Z (wall clock, 2026-10-10). It started on 9 workers while `v2-h16` shared the machine, and moved to 18 once that run stopped, so its timing isn't comparable with the 1M run.
```
5 | 0 | 41.44 | 51.76 | 40/40/40 | 9.1
70 | 6 | 64.55 | 64.72 | 54/48/88 | 15.3
200 | 18 | 50.9 | 60.46 | 54/48/88 | 16.6
330 | 30 | 57.49 | 60.71 | 54/48/88 | 24.1
460 | 42 | 58.46 | 63.17 | 54/48/95 | 31.2
590 | 55 | 67.44 | 135.5 | 74/49/144 | 35.4
720 | 72 | 109.4 | 252.4 | 74/49/162 | 37.3
785 | 81 | 157.2 | 419.7 | 211/84/426 | 38.3
850 | 109 | 410.4 | 552.3 | 580/469/794 | 38.7
915 | 150 | 573.1 | 884.4 | 580/469/794 | 38.2
970 | 897 | 238700 | 500000 | 500000/500000/500000 | 38.3
```
- Two held-out evals at gen 970: median 500000, min 500000.

**`runs/v2-h32x32-1m`** (shipped): 970 rows; cumulative 472 s; training frames 2.706e9; ran 01:24:26Z–01:32:36Z. Identical to the 500k run through gen 915 (cumulative s differs: 46 s at gen 915). Gen 970: cum 472 s, fitMean 384300, val 1M/1M/1M, ‖θ‖ 38.3.
- Held-out eval: median 1M, min 499.
- best.json: gen 970, 44-32-32-3, validation 1M/1M/1M.

**`runs/exp-window-h32x32`:** 255 rows; cumulative 466 s; training frames 2.574e9; ran 04:59:19Z–05:07:22Z.
```
5 | 0 | 0.00138 | 0.00186 | 40/40/60 | 9.0
50 | 4 | 0.0069 | 0.160 | 40/40/58 | 13.9
110 | 17 | 0.0465 | 0.259 | 40/40/75 | 18.1
140 | 43 | 0.101 | 0.752 | 49/40/126 | 19.7
155 | 71 | 0.280 | 1 | 96/41/273 | 20.3
185 | 142 | 0.2 | 1 | 186/175/211 | 21.5
200 | 181 | 0.310 | 1 | 265/173/1000000 | 21.8
215 | 248 | 0.393 | 1 | 700607/470/1000000 | 22.1
230 | 324 | 0.433 | 1 | 755844/490/1000000 | 22.5
245 | 406 | 0.578 | 1 | 743465/458465/1000000 | 22.9
255 | 466 | 0.521 | 1 | 1000000/583019/1000000 | 23.2
```
- Held-out at gen 255: median 1M, min 614432. The log says "done: held-out median 1000000 >= target 1000000; generation 255, cumulative wall 7.8 min".

**`runs/exp-A-hygiene-h32x32`:** 430 rows; cumulative 1193 s; training frames 5.883e9, validation frames 1.523e9; ran 06:12:12Z–06:32:05Z. Stopped with "time budget reached (mid-generation)", checkpoint at gen 430.
```
1 | 0 | 40.63 | 46.94 | 40/40/40 | 0 | 5 | 0.1 | 8.3
31 | 1 | 41.59 | 63.61 | 40/40/61 | 0 | 18 | 0.1 | 11.3
61 | 1 | 50.42 | 88.24 | 94/40/205 | 0 | 0 | 0.1 | 13.4
121 | 3 | 70.75 | 293.1 | 94/40/205 | 0 | 0 | 0.1 | 15.7
151 | 5 | 99.59 | 539.4 | 382/178/642 | 0 | 0 | 0.1 | 15.8
211 | 8 | 110.4 | 517 | 474.5/250/567 | 0 | 0 | 0.1 | 16.9
241 | 10 | 176.2 | 528.5 | 481/461/567 | 0 | 0 | 0.1 | 16.8
271 | 13 | 438.7 | 15240 | 560.5/461/1536 | 0 | 0 | 0.1 | 16.7
301 | 32 | 5065 | 88170 | 642/461/105158 | 0 | 0 | 0.1 | 16.8
331 | 110 | 15990 | 113100 | 805.5/461/133680 | 0 | 0 | 0.1 | 17.0
361 | 241 | 31890 | 184600 | 143894.5/479/196964 | 0 | 0 | 0.1 | 17.0
391 | 518 | 42420 | 205600 | 161276/479/242081 | 0 | 0 | 0.1 | 17.2
421 | 954 | 67220 | 705100 | 364147.5/145259/983622 | 0 | 0 | 0.1 | 17.3
430 | 1193 | 129400 | 1000000 | 1000000/156926/1000000 | 20 | 0 | 0.1 | 17.4
```
- Evals at gen 429:
  - offset 25, 75 seeds: median 1M, min 142961, passed 46, CP [0.4938, 0.7236];
  - offset 0, 25 seeds: median 1M, min 343749, passed 13, CP [0.3131, 0.7220].

**`runs/exp-B-curric-h32x32`:** 171 rows; cumulative 1189 s; training frames 7.168e9, validation frames 0.004e9; ran 06:35:17Z–06:55:06Z. Stopped with "time budget reached (mid-generation)", checkpoint at gen 171.
```
1 | 0 | 0.0053 | 0.0092 | 40/40/40 | 0 | 1 | 0.1 | 25000 | 8.3
13 | 1 | 0.0247 | 0.0998 | 40/40/53 | 0 | 1 | 0.1 | 5000 | 10.4
49 | 7 | 0.331 | 0.606 | 40/40/88 | 0 | 0 | 0.1 | 5000 | 14.4
97 | 30 | 0.298 | 0.5005 | 79.5/41/965 | 0 | 0 | 0.08 | 5000 | 17.9
121 | 44 | 0.286 | 0.629 | 65/40/547 | 0 | 0 | 0.08 | 5000 | 19.9
133 | 53 | 0.452 | 0.769 | 41/40/66 | 0 | 0 | 0.08 | 7500 | 21.0
145 | 76 | 0.533 | 0.767 | 50.5/40/146 | 0 | 0 | 0.08 | 16875 | 21.9
157 | 277 | 0.494 | 0.760 | 70/40/468 | 0 | 0 | 0.08 | 192225 | 22.5
169 | 1034 | 0.523 | 0.787 | 198/40/647 | 0 | 0 | 0.08 | 648761 | 23.0
171 | 1189 | 0.570 | 0.881 | 198/40/647 | 0 | 1 | 0.08 | 648761 | 23.1
```
- Note: the window **shrank** from 25000 to the 5000 floor in the first generations (late pass rate < 30% early), stayed there until about gen 130, then ran away.
- Evals at best gen 92:
  - offset 25, 75 seeds: median 635, min 459, passed 0, CP [0, 0.0480];
  - offset 0, 25 seeds: median 592, min 473, passed 0, CP [0, 0.1372].
- best.json validation: median 644.5, min 471, max 2041, count 0.

**Stale paths:** run logs and checkpoint log lines contain the old absolute path `/Users/kyle/Downloads/dino-game/...`. Resume uses `--dir`, so this is cosmetic, but don't be confused by it.

## D. Seed sets and identifiers

- **Ranges:** training [0, 2^30), drawn from the trainer RNG; validation [2^30, 2^31); held-out [2^31, 2^32).
  - `hashSeed(i, salt) = (Math.imul((i + 1) ^ salt, 0x9E3779B1) ^ 0x85EBCA6B) >>> 0`
  - validation = 0x40000000 + (hashSeed(i, 0x51) & 0x3FFFFFFF)
  - held-out = (0x80000000 | hashSeed(i, 0xE7)) >>> 0
  - `validationSeeds(n)` and `evalSeeds(n)` extend these sets; the first 25 never change.
- **VALIDATION_SEEDS** (25): 1958595899, 1310178824, 1747923161, 1300278702, 1876436095, 1093767500, 1665739549, 1151501538, 1727667891, 2018741120, 1516962385, 2144141094, 1577349623, 1937629892, 1572168853, 1778266010, 1211475499, 1835995960, 1336317385, 1628964574, 1129814383, 1686702716, 1187023885, 1480227090, 2054288291
- **EVAL_SEEDS** (held-out 0..24, the stop seeds): 2688327533, 2188644670, 3968052175, 3470470552, 2836571177, 2334801658, 4116820107, 2954974804, 2457397477, 4239418038, 3738166599, 3104276240, 2606690721, 2241202290, 3886907907, 2725094316, 2359638653, 4009540622, 3375642335, 2873313640, 2509947705, 4157752778, 3521765275, 2628386084, 4276189173
- **Held-out index usage:** 0–24 are the stop seeds, used by every run's stop check, so contaminated for certification. 25–99 were used to compare four models. **Next fresh offset: 100.**
- **Hard seeds** found by the statistics agent (training range): 822083584, 822432020, 822352830, 822249883. The exp-window and Run A models both crash on them at the identical scores 783542, 824244, 829855 and 973121 respectively.
  - Its other 46 fresh seeds were not recorded and are lost with the scratchpad. If a fresh training-range test set is needed again, generate a new one, e.g. a new salt, checked for no overlap.
- **Curriculum start-level hashing:** `START_SALT = 0x5EED5A17`; `uniform(seed, k) = createRng((hashSeed(seed, START_SALT) + Math.imul(k, 0x632BE5AB)) >>> 0)()`. Zero starts are stratified: the first floor(frac·n) episodes start at 0, plus one more with probability equal to the fractional part, decided by `uniform(seeds[0], 0)`.
- **Trainer RNG order per generation** (`rules.js drawGeneration`): popSize/2 noise seeds `Math.floor(rng()*2^32)`, then `episodes` training seeds, `Math.floor(rng()*2^30)`. With `--pair-episodes`, one episode-seed set per pair, drawn in pair order.

## E. Code map of the new trainer pieces (signatures as in the working tree)

- **`trainer/es.js`**
  - `gaussian(rng)`: Box–Muller on mulberry32.
  - `noise(seed, n, out)`
  - `sigmaVector(layers, sigma, mode)`
  - `perturb(theta, seed, sign, sigma)`: sigma may be a number or a vector.
  - `centredRanks(xs, ties = 'index')`
  - `pairDiffWeights(fitness)`
  - `esGradient(noiseSeeds, fitness, n, sigma, opts = {})`
  - `adamStep(theta, grad, state, lr, weightDecay, decay = 'coupled', b1 = 0.9, b2 = 0.999, eps = 1e-8)`
- **`trainer/env.js`**
  - `hashSeed`, `validationSeeds(n)`, `evalSeeds(n)`, `VALIDATION_SEEDS`, `EVAL_SEEDS`
  - `trainSeed(rng)`
  - `startScores(seeds, c)`
  - `episodeCaps(starts, c)`: `s === 0 ? c.startZeroCap : Math.min(s + c.window, c.target)`
  - `windowFitness(distance, startScore, span)`: `min(1, max(0, distance·0.025 − start)/span)`
  - `runEpisode(model, seed, capScore, startScore)`: startScore is passed to `createGame` only when non-zero, so the default path is byte-identical.
  - `median`
- **`trainer/rules.js`**
  - `drawGeneration(rng, c, cur)`
  - `valSummary(scores, target)` → {median, min, max, count, meanLog}
  - `isBetter(s, best, mode, nSeeds)`
  - `parseStopRule(text)` → {kind, seeds}
  - `alreadyTested(best, heldOut, target, seeds)`
  - `wantHeldOut(rule, val, best, heldOut, target, nVal)`
  - `sequentialAllPass(score, seeds, roundSize, target)`: async; rounds of `roundSize`, abort after a failing round.
  - `tiedPairs(fitness)`
  - `stallUpdate(stallMult, tiedFrac)`, with `STALL_GROW = 1.5`, `STALL_MAX = 4`
  - `annealUpdate({sigma, annealedAt}, {zeroStartScores, valMin, generation, sigmaMin})`, with `ANNEAL = { below: 200, frac: 0.25, factor: 0.8, cooldown: 10 }`
  - `adaptWindow(window, passRate, target)`, with `WINDOW_MIN = 5000`
  - `splitFolded(results, popSize)`
  - `clopperPearson(k, n, alpha = 0.05)`
- **`trainer/worker.js`:** a job is `{ id, noiseSeed, sign, seeds[, starts, caps] }`; setup is `{ theta, layers, actionRepeat, gameVersion, sigma (number|vector), target }`. Sign 0 means θ unperturbed, used for validation jobs.
- **`trainer/cli.js` key tables:** `DEFAULTS` (all config keys), the backfill list for resume, `STRUCTURAL = ['hidden', 'actionRepeat', 'gameVersion']`, `FLAG_TO_KEY`, `BOOL_TO_KEY` (`--window-adapt`, `--eval-when-fit-max`, `--fold-validation`, `--stall-rule`, `--sigma-anneal`, `--pair-episodes`), and `checkConfig`/`checkCurriculum`. The header comment documents every flag.
- **Small known doc issues** (not fixed):
  1. The `cli.js` header suggests `--decay decoupled` "use a larger value (~0.1)"; Run A suggests 0.1 is harmful, and 0.01 is the current recommendation.
  2. The `rules.js` header comment refers to `test/rules.test.js`; the file is `test/trainer-rules.test.js`.
  3. `AGENTS.md` does not yet document any of the new flags. It also says "the metric is the median score over the 25 held-out seeds", which is still the default but no longer the only rule. Update `AGENTS.md` if the changes are committed.
  4. The `report` command line in the `cli.js` header mentions `PROGRESS.md` (code fact; fine).

## F. Implementer's design decisions (accepted by the main session)

1. The no-retest guard is the default, with no flag. It compares best.generation, target and the held-out seed count. Older heldOut records lack a seeds field, so they are retested once.
2. Count selection plus allpass would also freeze (a saturated key can't strictly increase), so at saturation a tie replaces best (newest all-pass θ). A held-out failure deliberately does **not** unfreeze selection, because that would leak held-out into selection. The consequence: every all-pass validation after a failed held-out check becomes a new best and a new held-out attempt, so the stop seeds are contaminated and certification must use fresh offsets.
3. σ_eff = σ_base × stallMult. Anneal changes σ_base (the metrics `sigma` column); `sigmaEff` is a new column. A skipped step leaves θ, Adam (including t) and the RNG stream as normal.
4. With `fanin`, biases keep σ₀: the √fan_in correction compensates for summing over inputs, and a bias isn't summed.
5. The gradient normaliser stays `fitness.length` (= 2·pairs) for pair-difference weights too. Under Adam the constant hardly matters.
6. `--window-adapt` requires `--start-dist points` (with log starts, hi = target − window can fall below startMin).
7. `--start-zero-cap` defaults to the window when the curriculum is first set, and is stored resolved.
8. Points starts are clamped to target−1, so the span is never 0.
9. `--eval-when-fit-max` under the curriculum means "a candidate passed a window ending at the target".
10. `fps` = (training + validation frames)/genMs; `frames` stays training-only.
11. A resumed run keeps its metrics.csv header columns, so rows are never misaligned.
12. Dashboard thinning skips folded rows.
13. Booleans accept `--no-<flag>` on resume.
14. Certification: `eval --seed-offset K`. With offset 0 the eval header keeps the original wording (a smoke check matches it). evals.jsonl records gain seedOffset, passed and passCI95.
15. Run B: fold kept, because unfolded validation (two waves on an idle pool) costs more wall time per validation, and `--eval-when-fit-max` would fire on most generations with points starts. The optional 5-minute wd-0.01 resume check of Run A was skipped as low-information: it would start mid-climb and be confounded.

## G. The three research agents' ranked recommendations, verbatim in substance

**Literature agent (Opus):**
- R1 — starts: defensive mixture q = 0.5·uniform-in-points on [2000, target) + 0.5·adaptive ∝ √ĥ per level bin (pooled crash counts, an EMA plus a staleness bonus as in PLR). Windows end at min(start+W, target), normalised by W. About 25% zero starts with a short 2–5k cap.
- R2 — more distinct episodes: `--episodes 8–16` with W = 25k–12.5k (fitness is currently quantised in 0.25 steps when windows saturate), or per-pair (seed, start) sets with ARS-style ranking of F₊ − F₋ (PEGASUS-style common random numbers within the pair).
- R3 — selection and stop: rank by (count reaching the target, then mean log score); sequential validation (abort at the first failure against a 25/25 incumbent); a 50–100-seed confirmation set gated on 25/25; held-out once. Also overlap validation with the next generation's jobs.
- R4 — shape-generic optimiser: fan-in-scaled σ and lr (ES on V with W = s·V), decoupled decay, midranks.
- R5 — anneal σ late, or shrink it when the median window survival is high.
- R6 — episode-length control, e.g. the OpenAI starter's adaptive cutoff. It is subsumed by windows.
- R7 — plateau escape: portfolio or restarts (IPOP-style), or novelty search / quality diversity with a behaviour descriptor such as the crash obstacle or jump timing, as a training-only objective.
- R8 — normalise speed and velocity by 22, fresh runs only (core-hash change).
- R9 — compute:
  - WASM SIMD with lanes = episodes, f64x2 only (parity);
  - batching only as the vehicle for SIMD;
  - actionRepeat risky;
  - GPU not worth it now (about 10 µs per jitted step × 1.94M sequential frames ≥ 19 s per generation regardless of batch);
  - full CMA-ES impractical at 2.6k parameters; sep-CMA is about OpenAI-ES with per-coordinate σ.

**First-principles agent (Opus):**
1. Fix selection and stop (`--stop-rule allpass:N`, α-spending over attempts, seed sets extended to ≥ 59).
2. Uniform-in-points starts with the final window truncated.
3. Window and episode sizing (`--window 12500 --episodes 8` vs 50000×4).
4. Validation cadence and packing (36 seeds cost the same two waves as 25; validate when fitMax ≥ target; abort early once the median is decided).
5. Tie hygiene plus a stall rule (conjecture).
6. σ annealing or fan-in σ.
7. Per-frame work (≤ 5×).
8. Don't pursue early termination of training candidates.

**Independent agent (Fable):**
- R1 — count selection and the deadlock fix.
- R2 — validate every generation folded into the batch (or `--eval-every 1` once fitMax ≥ target).
- R3 — anneal σ with a measurable trigger, per-layer σ_l = σ₀/√fan_in.
- R4 — windows with uniform-in-points starts and *adaptive* length (keep the late pass rate near 50%; clamp [5k, target−start]); 25% zero starts capped at 5k. Estimated about 15× cheaper end-game generations.
- R5 — per-pair seeds, ARS-style.
- R6 — midranks, decoupled decay, plus an explicit stall mechanism.
- Plateau insurance: seed racing (K = 3–4 seeds, `--time-budget 1.5 --workers 6`, continue the leader).
- WASM MLP: 1.7–2.1× overall; batching in lockstep: 1.25–1.4×.
- It disagreed with the framing in two places: the plateau is a robustness problem, not a wall-time one; and log-uniform under-sampling is one of three compounding causes.

**What was actually adopted** in the W1–W6 brief: R3/R1 (selection and stop), R2 (folding), R6/R4 (ties, decay, stall, fanin), R3/R5 (anneal), R4/R1 (points starts, zero cap, adaptive window), R5/R2 (per-pair). The tie-rate controller and gated curriculum came later, from the statistics agent, and are not implemented.

## H. Scripts written by the main session (preserved verbatim; adjust paths)

These were in the lost scratchpad. Put them in a scratchpad or `runs/exp-*` area, not in the tracked tree.

**H.1 Per-frame cost split and worker scaling (`prof.js`).** Run `node prof.js` for the split, or `node prof.js <workers>` for scaling.
```js
const R='/Users/kyle/Desktop/dino-game/';
const { createGame, createRng } = require(R+'core.js');
const P = require(R+'policy.js');
const model = require(R+'model.js');
const { isMainThread, Worker, parentPort, workerData } = require('worker_threads');
const FR = 3e6;
function record(seed) { const g = createGame({ rng: createRng(seed), version: 'v2' }); const a = P.createAgent(model); const keys=[];
  while (keys.length < FR && (g.state !== 'over')) { const k = a.act(g); keys.push(k); g.step(k); } return keys; }
function time(fn){const t=process.hrtime.bigint(); const r=fn(); return [Number(process.hrtime.bigint()-t)/1e9, r];}
if (isMainThread && !process.argv[2]) {
  const keys = record(12345); const n = keys.length;
  const [tFull] = time(()=>{const g=createGame({rng:createRng(12345),version:'v2'});const a=P.createAgent(model);for(let i=0;i<n;i++)g.step(a.act(g));});
  const [tCore] = time(()=>{const g=createGame({rng:createRng(12345),version:'v2'});for(let i=0;i<n;i++)g.step(keys[i]);});
  const x=new Float64Array(P.OBS_DIM);
  const [tObs] = time(()=>{const g=createGame({rng:createRng(12345),version:'v2'});let s=0;for(let i=0;i<n;i++){P.observe(g,x);s+=x[0];g.step(keys[i]);}return s;});
  console.log(`full ${(n/tFull/1e6).toFixed(2)}M fps; core ${(n/tCore/1e6).toFixed(2)}M (${(100*tCore/tFull).toFixed(0)}%); obs ${(100*(tObs-tCore)/tFull).toFixed(0)}%, net+agent ${(100*(tFull-tObs)/tFull).toFixed(0)}%`);
} else if (isMainThread) {
  const W=+process.argv[2]; let done=0, tot=0; const t0=Date.now();
  for(let i=0;i<W;i++){const w=new Worker(__filename,{workerData:{seed:777+i}}); w.on('message',m=>{tot+=m; if(++done===W){const s=(Date.now()-t0)/1000; console.log(`${W} workers: ${(tot/s/1e6).toFixed(2)}M fps, ${(tot/s/W/1e6).toFixed(2)}M per worker`);}});}
} else {
  const g=createGame({rng:createRng(workerData.seed),version:'v2'});const a=P.createAgent(model);let f=0;const t=Date.now();
  while(Date.now()-t<6000){for(let i=0;i<10000;i++){g.step(a.act(g));f++; if(g.state==='over'){g.step({jump:true});}}}   // 60000 for the sustained figure
  parentPort.postMessage(f);
}
```
Measured: full 0.65M fps; core 4.06M (16%); obs 4%; net+agent 80%.

**H.2 Native ceiling for the MLP (`net.c`).** Build with `clang -O3 -mcpu=native -ffp-contract=off net.c -o net`. Random weights; same per-unit summation order as `policy.js`; weights input-major so it vectorises across output units. Measured 354 ns per decision; 174 ns with tanh replaced by a clamp.
```c
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
static double W1[44][32], B1[32], W2[32][32], B2[32], W3[32][3], B3[3];
static inline int decide(const double *x) {
  double h1[32], h2[32], o[3];
  for (int i = 0; i < 32; i++) h1[i] = B1[i];
  for (int j = 0; j < 44; j++) for (int i = 0; i < 32; i++) h1[i] += W1[j][i] * x[j];
  for (int i = 0; i < 32; i++) h1[i] = tanh(h1[i]);
  for (int i = 0; i < 32; i++) h2[i] = B2[i];
  for (int j = 0; j < 32; j++) for (int i = 0; i < 32; i++) h2[i] += W2[j][i] * h1[j];
  for (int i = 0; i < 32; i++) h2[i] = tanh(h2[i]);
  for (int i = 0; i < 3; i++) o[i] = B3[i];
  for (int j = 0; j < 32; j++) for (int i = 0; i < 3; i++) o[i] += W3[j][i] * h2[j];
  int b = 0; for (int i = 1; i < 3; i++) if (o[i] > o[b]) b = i; return b;
}
int main(void) {
  srand(1); double *all[] = {&W1[0][0], B1, &W2[0][0], B2, &W3[0][0], B3}; int n[] = {44*32,32,32*32,32,96,3};
  for (int a = 0; a < 6; a++) for (int i = 0; i < n[a]; i++) all[a][i] = (rand() / (double)RAND_MAX - 0.5) * 0.4;
  double x[44]; for (int i = 0; i < 44; i++) x[i] = sin(i) * 0.7;
  const int N = 20000000; long s = 0; struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
  for (int k = 0; k < N; k++) { x[0] = k * 1e-9; s += decide(x); }
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double d = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
  printf("native f64 net: %.0f ns/decision [%ld]\n", d / N * 1e9, s);
}
```

**H.3 Straight-line generated JS forward pass (`gen.js`).** It matched `createNet` decisions on 200k random inputs (0 mismatches), at 1,145 vs 1,181 ns: no gain. The key function:
```js
function genNet(layers, p) {
  let k = 0, src = '', inName = (j) => `x[${j}]`;
  for (let l = 1; l < layers.length; l++) {
    const nIn = layers[l-1], nOut = layers[l], b = k + nOut*nIn, last = l === layers.length-1;
    for (let i = 0; i < nOut; i++) {
      let e = `${p[b+i]}`; for (let j = 0; j < nIn; j++) e = `(${e}) + ${p[k+i*nIn+j]} * ${inName(j)}`;
      src += `const a${l}_${i} = ${last ? e : `tanh(${e})`};\n`;
    }
    const L = l; inName = (j) => `a${L}_${j}`; k = b + nOut;
  }
  const o = layers.at(-1), L = layers.length-1;
  src += `let best = 0, v = a${L}_0;\n`; for (let i = 1; i < o; i++) src += `if (a${L}_${i} > v) { best = ${i}; v = a${L}_${i}; }\n`;
  return new Function('tanh', `return function decide(x) {\n${src}return best;\n}`)(Math.tanh);
}
```

**H.4 Job-granularity A/B through the real pool (`sched.js`).** Run `node sched.js 42 [b-first]`. It plays one late-stage generation around `runs/v2-h32x32-1m/best.json`: (a) 128 jobs of 4 seeds, against (b) 512 single-seed jobs.
```js
const R='/Users/kyle/Desktop/dino-game/';
const { createPool } = require(R+'trainer/pool.js'); const { createRng } = require(R+'core.js');
const { trainSeed } = require(R+'trainer/env.js'); const best = require(R+'runs/v2-h32x32-1m/best.json');
const W = 18, POP = 128, EP = 4, TARGET = 1e6;
(async () => {
  const pool = createPool(W); const rng = createRng(+process.argv[2] || 42);
  const seeds = Array.from({ length: EP }, () => trainSeed(rng));
  const noise = Array.from({ length: POP / 2 }, () => Math.floor(rng() * 2 ** 31));
  const setup = { theta: Float64Array.from(best.params), layers: best.layers, actionRepeat: 1, gameVersion: 'v2', sigma: 0.1, target: TARGET };
  const cands = noise.flatMap(s => [{ noiseSeed: s, sign: 1 }, { noiseSeed: s, sign: -1 }]);
  for (const mode of process.argv[3] === 'b-first' ? ['b', 'a'] : ['a', 'b']) {
    const jobs = mode === 'a' ? cands.map(c => ({ ...c, seeds })) : cands.flatMap(c => seeds.map(s => ({ ...c, seeds: [s] })));
    const t0 = Date.now(); const res = await pool.run(setup, jobs);
    const secs = (Date.now() - t0) / 1000, frames = res.reduce((s, r) => s + r.frames, 0);
    console.log(`${mode}: ${jobs.length} jobs, ${secs.toFixed(1)} s, ${(frames / 1e6).toFixed(0)}M frames, ${(frames / secs / 1e6).toFixed(2)}M fps`);
  }
  await pool.terminate();
})();
```
- **Caveat:** the worker job format has changed since (it now carries `caps`). Episodes without `starts`/`caps` still play full games to `setup.target`; check `trainer/worker.js` before reusing this.
- Results: a-first: a 56.5 s, b 60.9 s. b-first: b 55.6 s, a 60.2 s. 378M frames; 141/512 episodes reached 1M; median 135663.

**H.5 `startScore` faithfulness and windowed crash rates (`start.js`).** For the shipped model, it compares natural speed and level with a `startScore` game at marks 1500…999000, then counts crashes in 50k windows from starts 0/10k/100k/300k/900k over 40 seeds. Results: section 2.3, plus 5/40 crashes from 0 (later shown to be high-side noise around a 4.6% rate) and 0/40 from every other start; about 91–114k frames per window.
```js
// core loop of the window test:
const g = createGame({ rng: createRng(1000 + i), version: 'v2', startScore: start }); const a = P.createAgent(m);
const cap = (start + 50000) / CFG.scoreCoef;
do { g.step(a.act(g)); frames++; } while (g.state !== 'over' && (g.state !== 'running' || g.distance < cap));
```

**H.6 Run summaries (`summ.js`).** Reads every `runs/*/metrics.jsonl`, `evals.jsonl` and `best.json`; this produced Appendix C.

## I. Communication setup used (for reference if it is recreated)

- **Subagents:** spawned with the Agent tool (`general-purpose`, model `fable` or the default Opus), run in the background, told to SendMessage findings to `main`. The main session relayed between them and marked relayed findings as claims, so the Fable agent's analysis stayed independent.
- **Peer session:** `implementor`, a separate interactive Claude Code session found via ListAgents, messaged by name with SendMessage. Its messages arrived as cross-session messages, and it was in "prompting" permission mode.
- **Corrections made through relaying:**
  1. One agent's "23–27% of late windows saturated" was inverted; the true figure is 73–77% (caught by the Fable gen-256 replay).
  2. "Ties are minor" (measured at gen 0 only) was overturned by the run-wide tie counts.
  3. The 776→252 stall-rule "gain" was reclassified as chaos.
  4. The 100-seed certification originally overlapped the stop seeds; it was fixed with `--seed-offset 25`.
