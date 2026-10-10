# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Chrome-dino clone (plain JS, opened over `file://`) with an "AI mode" driven by a small MLP. The game and the ES trainer run in plain Node with **no npm dependencies**.

**The shipped model is a behaviour clone of the solver, not an ES model** (since 2026-10-10; see "Decision log" at the end). It sees only the 44 allowed features. It was trained by supervised learning to imitate `test/solver.js`, which uses privileged information. The neuroevolution (OpenAI-ES) trainer in `trainer/` is still maintained, but it now serves as the research pipeline, not as the source of the shipped model.

## Commands

```sh
node --test test/                         # unit/parity tests (npm test)
node --test test/parity.test.js           # a single test file
node --test --test-name-pattern 'v2' test/   # tests matching a name
node test/browser-check.js                # headless Chrome over CDP (CHROME env var overrides the macOS Chrome path)
node test/smoke-trainer.js                # end-to-end trainer smoke test (lock, SIGINT, resume, hash mismatch, eval, export)
node trainer/cli.js bench --seconds 5     # simulated frames/s
node trainer/dashboard.js                 # live training dashboard at http://127.0.0.1:8080
```

There is no build step and no linter. To play, open `index.html` (or `index-css.html`) directly in a browser.

Trainer CLI (`trainer/cli.js`; the header comment lists every option):
- `train` / `eval` / `export` / `export-css` / `report`
- `finish`: eval, export, browser check, report and **git commit**.
- `session`: `train` followed by `finish`.

Training outlasts foreground limits, so run it detached and log to the run directory. An experimental ES run looks like this:

```sh
nohup node trainer/cli.js train --wasm --hidden 32,32 --target 1000000 --time-budget 70 --dir runs/exp-<name> >> runs/exp-<name>/train.log 2>&1 &
kill -INT <pid>    # PID is in runs/<dir>/train.lock; saves a checkpoint and releases the lock
```

Behaviour cloning (see "Model provenance"):

```sh
node trainer/cli.js bc-gen --dir runs/bc-<name>                                   # solver demos: seeds 1-40, about 2.1M rows, seconds
node trainer/cli.js bc-train --dir runs/bc-<name> --hidden 32,32                  # node backend, about 10 s/epoch, then 25-seed validation to 1M
node trainer/cli.js bc-train --dir runs/bc-<name> --hidden 32,32 --backend python # numpy backend, about 1 s/epoch
```

- `--backend python` needs `.venv` with numpy (`.venv/bin/python`); the `PYTHON` env var overrides the interpreter. A worktree has no `.venv` of its own, so set `PYTHON` there.
- `bc-gen` writes `demos.bin` (gitignored) and `demos.json`; `bc-train` writes `best.json` and `bc-metrics.jsonl` (per-epoch loss and recall, not `metrics.jsonl`). `eval` and `export` then work as for an ES run.
- `bc-train` refuses demos from another core hash unless `--force`. The flags are in the `trainer/cli.js` header.
- `bc-train --backend python --snapshots` also writes each epoch's params to `<dir>/snapshots/epoch-NN.json` (gitignored research output); it leaves training unchanged. The node backend rejects it.

The CLI's `--dir` default (`runs/main`) and `--hidden` default (`16`) are stale. `runs/main` no longer exists. Pass both explicitly.

- **Shipped model:** `runs/bc-v2-h32x32` (44-32-32-3).
- **Previous ES model:** `runs/v2-h32x32-1m` (generation 970), kept for comparison.

**`--auto-export`, `session` and `finish` overwrite the shipped `model.js` and `model.css`**, and from an ES run that would replace the shipped clone with a weaker model. Don't use them unless an ES model has beaten the shipped one on fresh held-out seeds and the user has approved replacing it. `browser-check.js` checks that `model.css` matches `model.js`; after you replace `model.js`, run `export-css`.

## Architecture

**Model provenance: two training pipelines, one model format.** Every model is a `policy.js` MLP (`dino-mlp-v1`: layers, flat params, actionRepeat, gameVersion), so the browser, CSS page, dashboard and `eval` run any model the same way. The models differ only in how the params were obtained.

| | Behaviour cloning (**shipped**) | Neuroevolution (research) |
|---|---|---|
| Training signal | the solver's action on each frame (supervised, cross-entropy) | the capped game score (OpenAI-ES, no shaping) |
| Teacher | `test/solver.js` (reads the hidden gap and the fairness lookahead) | none |
| Inputs at play time | the 44 `observe()` features | the 44 `observe()` features |
| Code | `trainer/bc.js` (demos, node backend, best.json), `trainer/bc_train.py` (numpy backend); `cli.js bc-gen` / `bc-train`. Prototype: `research/scripts/feature-sufficiency/` | `trainer/` |
| Run dir | `runs/bc-v2-h32x32` (`best.json` records the method in `config`) | `runs/v2-h32x32-1m`, `runs/exp-*` |

- The BC pipeline is `bc-gen` then `bc-train` (Commands above). The teacher is `test/solver.js`; demonstration features come only from `policy.observe`. Both backends run the same algorithm (a port of the prototype's `trainB.py`): the python backend reproduces the shipped clone's params exactly from the prototype's demos (`--seed 1`), and the node backend needs no dependencies but does not match it bit for bit.
- `bc-gen` reproduces the prototype's rows (`cloneB.js gen`), except that the boring-frame counter now resets every episode, so a few down-weighted rows differ.
- To ship a new BC model: `bc-train` into a `runs/bc-*` directory, certify it with `eval`, then `export --dir runs/bc-... --which best`.

**One source of game logic.** `core.js` is the DOM-free simulation, and the trainer, tests, browser and dashboard all run it.
- `createGame({ rng, version, hooks })` creates independent games, many per thread.
- `step(keys)` = `applyKeys` + `update` with dt = 1, on a simulated clock `now = frame * FRAME_MS`.
- The core must contain no DOM, clock, storage, audio or `Math.random`; `test/parity.test.js` enforces this.
- Gameplay sizes are computed from the sprite pixel-map rows, not from canvases.

**Dual-format files.** Every shared file (`core.js`, `policy.js`, `render.js`, `model.js`) works both as a classic `<script>` that sets a global (`DinoCore`, `DinoPolicy`, `DinoRender`, `window.DINO_MODEL`) and as a Node `require()`. That is because `file://` blocks ES modules and `fetch()`. Keep that pattern: no `import`/`export`, no JSON loading in the browser.

**Shared policy.** `policy.js` holds the observation extraction (44 features), MLP inference and the agent loop (action repeat, idle→start). The trainer workers, `eval`, the browser AI mode and the dashboard all use it, so the browser AI sees exactly what training saw.
- Anti-cheat rules (tested in `test/policy.test.js`): observations may only use obstacles with x < 600.
- Never observe `o.gap`, `requirement()`/`placeFairly()` (the fairness solver, which the core doesn't export), RNG state, spawn history or difficulty internals.
- `placeFairly` parks future obstacles off-screen, so the obstacles array leaks the future.
- The anti-cheat rules govern what the model **observes** at play time, and they apply to every model, however it was trained.
- The ES reward is the capped score, with no shaping.
- Learning from a privileged teacher (the solver) is allowed. The user approved it on 2026-10-10, and the shipped model is trained that way. A teacher may use hidden state; the student model may not observe it.

**Browser shell.** `game.js` covers input, sound, localStorage HI and the fixed-timestep loop.
- AI mode calls the policy *inside* the fixed-timestep loop, right before each `update`, never once per rAF. It drives the same `pressJump`/`releaseJump`/`pressDown`/`releaseDown` path as a human.
- With AI on, human jump/duck input is ignored and the AI auto-restarts after a crash.
- A run with AI on at any point never writes HI. HI is stored per game version (`HI_KEY`).
- `render.js` is the canvas renderer shared with the dashboard playback.

**CSS network page.** `index-css.html` + `css-nn.js` evaluate the same MLP in the browser's style engine. `model.css` is generated from `model.js` by `trainer/css-model.js` via `export` or `export-css`; never edit it by hand. JavaScript only writes `--x0..--x43` and reads `--act`.

**Game versions.** `v1` is the original rules and `createGame`'s default. `v2` is identical below 1k points, then ramps difficulty on a log scale up to 1M. All training uses v2. The version is stored in checkpoints, `best.json` and `model.js`; the page plays the model's version, and `?game=v1|v2` overrides it.

**Trainer** (`trainer/`):
- `es.js`: OpenAI-ES with antithetic sampling, centred ranks and Adam.
- `pool.js` / `worker.js`: worker_threads. Only θ and noise seeds cross threads, and workers regenerate the noise.
- `env.js`: episodes and seed sets, in three disjoint ranges:
  - training seeds;
  - validation seeds, used to select the best model;
  - held-out seeds (`EVAL_SEEDS`), used only by `eval` and the stop check.
- The stop check, by default (`--stop-rule median`), is the median score over the 25 held-out seeds, with episodes capped at `--target`. The alternative is `--stop-rule allpass:N`: all N held-out seeds must reach the target.
- `rules.js`: the per-generation policies as pure functions (selection, stop check, window adaptation, fitness aggregation), tested in `test/trainer-rules.test.js`.
- `store.js`: atomic writes (temp file, fsync, rename), the `train.lock` PID lockfile, and `coreHash`.
- `ops.js`: report, finish and session.

**Trainer flags.** Every flag is off or at its old value by default; with defaults, training is bit-identical to the original trainer. The header comment of `trainer/cli.js` documents each flag precisely; this list is only a map.
- Curriculum (training-only; validation and eval always play full games):
  - `--window`, `--start-zero-frac`, `--start-min`, `--start-dist log|points`, `--start-zero-cap`;
  - `--window-adapt`: gated, so the window shrinks while start-0 episodes mostly fail and only grows once the opening is learned. Decided every `--window-adapt-every` generations (default 5) on pooled pass counts. Needs `--start-dist points`;
  - `--fitness-agg mean|geomean`: geomean lets a failing stage (for example the opening) dominate. Needs `--window`.
  - `--ramp-frac F --ramp-min P --ramp-max P`: natural-ramp starts. The solver (`test/solver.js`) plays from 0 and hands over to the student at a level in [P_min, P_max] (for example 400–800), when grounded with keys up. These count as "late" episodes for `--window-adapt`. Needs `--window`. Costs about 2.8 ms of teacher prefix per ramp episode.
- Speed: `--wasm` runs the MLP forward pass as WASM SIMD in the trainer only (train, `eval --wasm`, `bench --wasm`). It is bit-identical to JS (verified on arm64 and x86) and about 1.65–1.7× faster. Use it for all experiments.
- Selection and stopping: `--select median|count`, `--stop-rule median|allpass[:N]`, `--val-seeds N`, `--fold-validation` (validates every generation inside the training batch), `--eval-when-fit-max`.
- ES:
  - `--rank-ties index|mid`;
  - `--decay coupled|decoupled`: with decoupled, use `--weight-decay` about 0.01;
  - `--sigma-scale none|fanin`, `--sigma-anneal` with `--sigma-min`, `--pair-episodes`;
  - `--stall-rule`: not recommended, kept only to reproduce an old run.
- `eval --seeds N --seed-offset K` scores held-out seeds K..K+N-1 and reports a 95% Clopper–Pearson interval for the pass rate. The stop check uses the first N held-out seeds, so certify a run from an offset past them.

**Verification tools.**
- `node test/bit-identity.js <oldTree> <newTree>` runs 7 small training scenarios, including one that fires the `allpass` held-out stop, in two checkouts (for example a `git worktree` at the pre-change commit, and yours) and diffs metrics, checkpoints and `best.json` (about 9 s). It refuses to start while another trainer is running unless `--allow-busy`.
- A new trainer flag must leave defaults IDENTICAL against the pre-change commit.

**Findings that constrain defaults.**
- Plateau escape is seed-dependent: on 32,32 with `--rank-ties mid --decay decoupled --weight-decay 0.01`, about 25% of seeds escape by generation 900 and 80–90% by generation 1500.
- Decoupled weight decay 0.1 slows the climb; use about 0.01.
- A tie-rate σ controller was tried and removed: σ ran away, because ties rise again at large σ.
- Research notes live in `research/`; multi-session handoffs live in `handoff/`.

## Invariants to preserve

- **Parity gate.** `test/golden/*.json.gz` were recorded from baseline commit `2da34de` by `test/record-golden.js`, and v1 must reproduce them frame-for-frame. That includes the order of RNG draws: clouds, stars and the idle blink share the game RNG, and the obstacle sequence depends on them. Any core change that alters v1 behaviour breaks `parity.test.js`. Don't re-record the traces to make it pass.
- **Core hash.** It covers `core.js` and `policy.js`. Editing either one invalidates existing checkpoints, which then `--resume` only with `--force`. Use `--force` only when decisions are provably unchanged; `test/net.test.js` checks bit-identical decisions for the shipped model.
- **Locked on resume.** `hidden`, `actionRepeat` and `gameVersion` can't change on resume; other options can be overridden.
- **Evaluation uses the unmodified game.** No changes to CFG, physics, collision or fairness for evaluation. Any curriculum is training-only.
- **Committed vs ignored.** In `runs/<name>/`, `best.json`, `evals.jsonl` and `metrics.*` are committed, and for BC runs also `demos.json` and `bc-metrics.jsonl`. Checkpoints, logs, locks and `demos.bin` are gitignored.

## Research method

Work as a methodical researcher. Don't try things to see what sticks.
- **Every run serves a decision.** Before it starts, write down:
  - the decision it informs;
  - the competing hypotheses and what each predicts;
  - the premises, each checked against existing data or a seconds-scale static probe;
  - a positive-control arm that must succeed within the horizon;
  - the horizon and seed count, sized from known variance and the effect worth detecting;
  - how each possible result will be read.

  If no outcome would change a decision, don't run it.
- **Prune the search space with the cheapest discriminating test first:** bounds, then static probes on saved θ, then oracle and necessary-condition tests, then short replays, then full runs, then certification.
- **A null is conditional.** Record its validity (could the test have detected success?), its power, its scope (shape, seeds, horizon, settings, commit) and what would reopen it. Don't declare a direction dead from a proxy or from one run.
- **The full method is in `handoff/2026-10-10T1635Z-retrospective.md`, §4 and §6.** Keep the pruned-branch register (§6.4) current in each new handoff.

## Decision log

### 2026-10-10: ship a behaviour clone of the solver instead of the ES model

**Problem.** The goal is an AI that reliably reaches 1,000,000 points in v2. The shipped ES model (`runs/v2-h32x32-1m`, generation 970) passed 72/75 fresh held-out games on offsets 100–174. Its failures were early, at 460–740 points (the first pterodactyls), plus a rare late crash near 994k. A day of ES work (σ control, tie handling, curricula, seed racing) produced no model better than it. Pure ES reaches the second wall slowly or not at all on most seeds. The open question was whether the 44 inputs were the limit, or the training method was.

**Constraints at the time.**
- `policy.js` observations and the anti-cheat rules were fixed. The model may not observe hidden state.
- The model has to run as a `dino-mlp-v1` MLP in JS and in the CSS evaluator.
- Training is CPU-only. An ES run that escapes the plateau takes a seed lottery of about 1,200 generations, and most escaped seeds then stall at the second wall.

**Evidence.**
- `research/feature-sufficiency.md`, probe B: MLPs trained by supervised learning on 2.1M frames of solver demonstrations, using only the unchanged 44 features, played 352 full games to 1M with 0 crashes. This included 44-8-8-3 networks.
- Paired certification on fresh held-out offsets 175–324 (150 seeds, target 1M, v2, core hash `0f329d4e…`), using `node trainer/cli.js eval --model <m> --target 1000000 --seed-offset 175 --seeds 150`. Raw output is in `runs/exp-bc-clones/cert-175-324-*.txt` (gitignored); the clone's record is in `runs/bc-v2-h32x32/evals.jsonl`.

  | Model | Passed | 95% CP | Failures |
  |---|---|---|---|
  | clone `m_base_s1`, now shipped | **150/150** | 0.976–1.000 | none |
  | clone `m_base_s2`, a replicate | 150/150 | 0.976–1.000 | none |
  | previous ES model | 143/150 | 0.906–0.981 | 467, 485, 664, 673, 740, 994,235, 995,049 |

  On the same seeds that is 7 games the clone passed and ES failed, and 0 the other way (exact McNemar, one-sided p = 0.008).
- The shipped clone also passed 25/25 on the validation seeds and 25/25 on held-out offsets 0–24.
- Its smallest top-2 logit margin over 600k frames of play is 2.7e-4, against 5.8e-5 for the ES model. So the CSS evaluator is less likely to diverge than before.
- Checks at export: tests 66/66; `test/browser-check.js` 38/38, with 0 CSS/JS disagreements over 1,015 frames.

**Options considered.**
- *Keep the ES model and keep improving ES* (a curriculum for the 400–800 opening, late high-pterodactyl exposure, seed racing). Rejected as the shipping path because nothing produced a better model and the second wall is seed-dependent. It continues as research.
- *Change the `policy.js` features.* Rejected: probe B shows the existing features already support near-perfect play, so the bottleneck was not information.
- *Use the clone only as an ES starting point, then fine-tune with ES.* Not needed for shipping, because the clone is already at the certification ceiling. It stays available if a future change, such as a new game version, needs a polished policy.
- *Ship the clone (chosen).* The user approved this on 2026-10-10 ("Shipping the clone is allowed"). Their earlier question was whether learning from the solver's privileged labels contaminates training. The answer: it doesn't. The evaluation seeds are disjoint from the demonstration seeds (below 1e6 against at least 2^31), and the model observes only the allowed features. It is the standard privileged-teacher, or "learning by cheating", setup.

**Decision.** `model.js` and `model.css` now hold `runs/bc-v2-h32x32/best.json`: a 44-32-32-3 behaviour clone of `test/solver.js` (variant `base`, `trainB.py` with SEED=1). The ES trainer remains the research pipeline. The project's claim changes from "neuroevolution-trained" to "distilled from an exact solver, on legal inputs".

**Reversibility.** Cheap to reverse. Run `node trainer/cli.js export --dir runs/v2-h32x32-1m --which best`, which also writes `model.css`, then run the browser check. The model format, the inputs and the game are unchanged. The clone's HI key and game version are the same as before.

**What would change this.**
- An ES model, or an ES-fine-tuned clone, beats the shipped clone on ≥ 150 fresh held-out seeds.
- A failure mode appears in the clone that the solver doesn't have, for example covariate shift in states the solver never visits.
- The user decides the AI must learn from the score alone.
- Any change to `observe()`, `core.js` or the game version invalidates the clone. Regenerate the demonstrations and retrain, then re-certify.
