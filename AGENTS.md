# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Chrome-dino clone (plain JS, opened over `file://`) with an "AI mode" driven by a small MLP. The game and the ES trainer run in plain Node with **no npm dependencies**.

**The shipped model is a behaviour clone of the solver, not an ES model** (since 2026-10-10; see "Decision log" at the end). It sees only the 44 allowed features. It was trained by supervised learning to imitate `test/solver.js`, which uses privileged information. The neuroevolution (OpenAI-ES) trainer in `trainer/` is still maintained, but it now serves as the research pipeline, not as the source of the shipped model.

## Commands

**Python setup comes first.** Before anything that needs a Python backend (`bc-train` / `bc-train-batch` with `--backend python` or `mlx`, the `bc-train-batch` default backend choice, `test/bc.test.js` and its python param pins), run `./setup.sh --check`. If it fails, run `./setup.sh` without asking: it only creates or updates the project's own `.venv` and `.venv-mlx` from `requirements.txt` / `requirements-mlx.txt` and never deletes anything. In a git worktree, don't create venvs: export `PYTHON` and `PYTHON_MLX` to the main checkout's venvs first (the `--check` output names them), then `--check` checks those. `./setup.sh --help` lists the options; README "Setup" has the platform details.

```sh
./setup.sh --check || ./setup.sh          # Python venvs for the numpy and mlx backends (read-only check first)
node --test test/                         # unit/parity tests (npm test)
node --test test/parity.test.js           # a single test file
node --test --test-name-pattern 'v2' test/   # tests matching a name
node test/browser-check.js                # headless Chrome over CDP (CHROME env var overrides the macOS Chrome path)
node test/smoke-trainer.js                # end-to-end trainer smoke test (lock, SIGINT, resume, hash mismatch, eval, export)
node trainer/cli.js bench --seconds 5     # simulated frames/s
node trainer/dashboard.js                 # live dashboard at http://127.0.0.1:8080: ES, BC and bc-gen runs, runs/<group>/<run> too; playback incl. the teacher (solver)
node test/dashboard-check.js              # headless Chrome check of the dashboard on a temp runs/ fixture
```

There is no build step and no linter. To play, open `index.html` (or `index-css.html`) directly in a browser. Both pages have a model selector and a speed selector (1×–4096×); see "Browser shell" and "CSS network page".

Trainer CLI (`trainer/cli.js`; the header comment lists every option):
- `train` / `assess` / `export` / `export-css` / `export-models` / `report`
- `export-models` regenerates the pages' model selector from `BUNDLED_MODELS` in `trainer/cli.js` (run dirs and labels): `models.js` (`DINO_MODELS`, with weights, for `index.html`), `models-css/<id>.css` (one CSS network each) and `models-css/index.js` (`DINO_MODEL_INDEX`, metadata and `--nn-model-id` only, for `index-css.html`). The id is the run-dir name. It never touches the root `model.js` / `model.css`. The outputs are committed, and `test/models.test.js` fails when they are stale. After shipping a new model, move `shipped: true` in `BUNDLED_MODELS` and rerun it.
- `assess` (held-out assessment) was called `eval` until 2026-10-10 (user order). `eval` still works as a deprecated alias and prints a one-line notice. New records go to `<run>/assess.jsonl`; older runs' `evals.jsonl` were renamed, and every reader (dashboard, `ops.js` report, `scans.js`) merges both names (`readAssessLines` in `trainer/store.js`). Historical pre-registrations still say `eval`. It runs the network on wasm by default (`--no-wasm` for JS; see "Trainer flags").
- `finish`: eval, export, browser check, report and **git commit**.
- `session`: `train` followed by `finish`.

Training outlasts foreground limits, so run it detached and log to the run directory. An experimental ES run looks like this:

```sh
nohup node trainer/cli.js train --hidden 32,32 --target 1000000 --time-budget 70 --dir runs/exp-<name> >> runs/exp-<name>/train.log 2>&1 &
kill -INT <pid>    # PID is in runs/<dir>/train.lock; saves a checkpoint and releases the lock
```

Behaviour cloning (see "Model provenance"):

```sh
node trainer/cli.js bc-gen --dir runs/bc-<name>                                   # solver demos: seeds 1-40, about 2.1M rows, seconds
node trainer/cli.js bc-train --dir runs/bc-<name> --hidden 32,32                  # default backend (below), then 25-seed validation to 1M
node trainer/cli.js bc-train --dir runs/bc-<name> --hidden 32,32 --backend python # numpy backend, about 1 s/epoch
node trainer/cli.js bc-train --dir runs/bc-<name> --hidden 32,32 --backend node   # plain Node, no dependencies, about 10 s/epoch
```

- `bc-train`'s default backend follows `bc-train-batch` (user's decision, 2026-10-12): `mlx` when an mlx interpreter is configured (`.venv-mlx` or `PYTHON_MLX`), else `python` when a numpy interpreter is configured (`.venv` or `PYTHON`), else `node`. It logs the choice (`bc-train: --backend X (default: ...)`), and `best.json` `config.backend` records it. Runs from before 2026-10-12 without `--backend` used node: pass `--backend node` to reproduce them. In a venv-less checkout, python-only flags (`--snapshots`, `--extra-demos`, `--transition-weight`) fail and say the default fell back to node.
- `--backend python` needs `.venv` (`./setup.sh`: CPython 3.9–3.12, 3.9 preferred, `numpy==2.0.2` from `requirements.txt`; on macOS arm64 that wheel links Accelerate, which the `test/bc.test.js` python pins assume). The `PYTHON` env var overrides the interpreter. A worktree has no `.venv` of its own, so set `PYTHON` there.
- `--backend mlx` (Apple GPU, `trainer/bc_train_mlx.py`): `bc-train` and `bc-train-batch` both accept it.
  - It needs `.venv-mlx` (`./setup.sh`, macOS 14+ on Apple silicon only): pyenv Python 3.14.8 (3.12–3.14 work), `mlx==0.32.3`, `mlx-metal==0.32.3`, `numpy==2.5.4` from `requirements-mlx.txt`. The `PYTHON_MLX` env var overrides the interpreter; a worktree has no `.venv-mlx`, so set `PYTHON_MLX` there.
  - It runs the python backend's algorithm with the same init and batch order (host-side RNG in `trainer/bc_common.py`). Its params match the numpy backend's to float32 noise, about 1e-7 after an epoch, and it is bitwise deterministic run to run.
  - It sets `MLX_ENABLE_TF32=0`: MLX's GPU float32 matmul defaults to TF32, which made gradients 1,000× noisier.
  - Every python-only flag works with it. `config.engine` records the version, the device and `tf32`.
- Many models at once: `node trainer/cli.js bc-train-batch --dir-pattern 'runs/<group>/v2-bc-<teacher>-d{depth}w{width}-s{seed}' --shapes 32,32/8x2 --seeds 1-10 --demos <dir> [--backend python [--jobs 6] | mlx]`. Each model gets its own ordinary run dir.
  - The default backend is `mlx` when an mlx interpreter is configured (`.venv-mlx` or `PYTHON_MLX`) and `--jobs` isn't given; otherwise `python` (there is no node backend). This was the user's choice on 2026-10-11. The command logs which one it picked. `bc.defaultBackend` in `trainer/bc.js` owns this rule for both commands.
  - `--backend python` runs `--jobs` `bc_train.py` processes at once, each with 1 BLAS thread. Each member's params are byte-identical to `bc-train --backend python` with the same shape and seed (checked on the full demos, 10/10).
  - `--backend mlx` trains each shape's seeds in one GPU process (groups of at most 40).
  - Tier A runs on one shared worker pool, packed: every model waiting when the pool frees up plays its 25 games in one pool run (`scoreSeedsMany`; `worker.js` takes `setup.models` with a per-job `model` index). Validation results are unchanged. Training continues meanwhile.
  - The dashboard lists every member while it waits, trains and validates (`config.batchRun`; `config.batch` stays the minibatch size).
  - Measured for 10 seeds of 2×32 on the M5 Pro (`research/gpu-trainer-results.md`):
    - sequential `bc-train`: 201 s;
    - `--jobs 6`: 91 s;
    - mlx: 84 s, of which training is 21 s;
    - mlx with 40 seeds: training 59 s.
  - Tier A is now the bottleneck. Plan: `research/plan-gpu-trainer.md`.
  - Every other `bc-train` option is shared by the batch. The numpy step lives in `trainer/bc_train.py`; host-side parts shared by both python engines are in `trainer/bc_common.py`, which must stay Python 3.9-compatible.
- `bc-gen` writes `demos.bin` (gitignored) and `demos.json`; `bc-train` writes `best.json` and `bc-metrics.jsonl` (per-epoch loss and recall, not `metrics.jsonl`). `eval` and `export` then work as for an ES run.
- `bc-gen --teacher jump|duck` picks the teacher (default `jump` = `createSolver`, byte-identical demos to before; `duck` = `createDuckSolver`, label 2 = down key). The manifest records `teacherName`, `bc-train` copies it into the `best.json` config, and recall gains a `duck` key only when duck rows exist.
- `bc-train` refuses demos from another core hash unless `--force`. Its validation runs on wasm by default (`--no-wasm` for JS). The flags are in the `trainer/cli.js` header.
- `bc-train --backend python --snapshots` also writes each epoch's params to `<dir>/snapshots/epoch-NN.json` (gitignored research output); it leaves training unchanged. The node backend rejects it.
- Distillation and quantization (both backends, off by default, defaults bit-identical): `bc-train --distill-from <best.json> [--distill-mode soft|hard] [--tau T] [--alpha A]` (Hinton soft targets or the teacher's argmax; `config.distillFrom` records path and params sha256), `--quant-bits N [--quant-scheme row|layer|pow2]` (QAT, symmetric int weights stored as q × scale, still `dino-mlp-v1`; `pow2` makes ALL params N-bit, weights and biases, as q × 2^-k with one integer k per layer, `config.quant.k`, verified and packed by `research/scripts/prereg-distill/pack.js`), and `bc-quantize --from <best.json> --dir <run> --quant-bits N` (post-training). Definitions in the `trainer/bc.js` header; plan in `research/prereg-distill.md`.

The CLI's `--dir` default (`runs/main`) and `--hidden` default (`16`) are stale. `runs/main` no longer exists. Pass both explicitly.

- **Shipped model:** `runs/exp-bc-duck/v2-bc-duck-d2w8-s1` (44-8-8-3, duck generation: jumps and ducks; since 2026-10-10, see the decision log).
- **Previous shipped clone:** `runs/v2-bc-jump-d2w32-s1` (44-32-32-3, jump generation), kept for comparison.
- **Previous ES model:** `runs/v2-es-d2w32-s1-t1m` (generation 970), kept for comparison.

**`--auto-export`, `session` and `finish` overwrite the shipped `model.js` and `model.css`**, and from an ES run that would replace the shipped clone with a weaker model. Don't use them unless an ES model has beaten the shipped one on fresh held-out seeds and the user has approved replacing it. `browser-check.js` checks that `model.css` matches `model.js`; after you replace `model.js`, run `export-css`.

## Architecture

**Model provenance: two training pipelines, one model format.** Every model is a `policy.js` MLP (`dino-mlp-v1`: layers, flat params, actionRepeat, gameVersion), so the browser, CSS page, dashboard and `eval` run any model the same way. The models differ only in how the params were obtained.

| | Behaviour cloning (**shipped**) | Neuroevolution (research) |
|---|---|---|
| Training signal | the solver's action on each frame (supervised, cross-entropy) | the capped game score (OpenAI-ES, no shaping) |
| Teacher | `test/solver.js` (reads the hidden gap and the fairness lookahead) | none |
| Inputs at play time | the 44 `observe()` features | the 44 `observe()` features |
| Code | `trainer/bc.js` (demos, node backend, best.json), `trainer/bc_train.py` (numpy backend); `cli.js bc-gen` / `bc-train`. Prototype: `research/scripts/feature-sufficiency/` | `trainer/` |
| Run dir | `runs/v2-bc-jump-d2w32-s1` (`best.json` records the method in `config`) | `runs/v2-es-d2w32-s1-t1m`, `runs/exp-*` |

- The BC pipeline is `bc-gen` then `bc-train` (Commands above). The teacher is `test/solver.js`; demonstration features come only from `policy.observe`. Both backends run the same algorithm (a port of the prototype's `trainB.py`): the python backend reproduces the shipped clone's params exactly from the prototype's demos (`--seed 1`), and the node backend needs no dependencies but does not match it bit for bit.
- `bc-gen` reproduces the prototype's rows (`cloneB.js gen`), except that the boring-frame counter now resets every episode, so a few down-weighted rows differ.
- To ship a new BC model: `bc-train` into a `runs/bc-*` directory, certify it with `assess`, then `export --dir runs/bc-... --which best`.

**Naming scheme (unambiguous; the user asked for it, 2026-10-10).** A bare `v<N>` means **game rules only**. Nothing else is named with a bare version number.

| Thing | Name | Where it appears |
|---|---|---|
| Game rules | **rules v1**, **rules v2**, **rules v3**; write "rules v2" in prose | `gameVersion: 'v1' \| 'v2' \| 'v3'`, `?game=`, `--game`, `HI_KEY`. The code field is unchanged |
| Model file format | **`dino-mlp-v1`** | `model.js`, `best.json` |
| BC teacher | **`teacher-jump`** (`test/solver.js` `createSolver`, jump only) and **`teacher-duck`** (planned: ducks under mid pterodactyls) | `best.json` `config.teacherName` and demos manifest `teacherName`, stored as `jump` / `duck` (= `--teacher`) |
| Project generation | named after its teacher: the **jump generation** (everything up to 2026-10-10: the shipped clone and the depth × width sweep) and the **duck generation** (`teacher-duck` and the clones trained on it) | docs, handoffs, register rows |
| Game rules | **rules v3**: rules v2 up to 100k points, then superhuman reaction budgets (a speed step at 100k; < 150 ms for most must-act obstacles from there on; `research/game-v3.md`) | `gameVersion: 'v3'`, `?game=v3`, `HI_KEY` `dino-hi-v3`, run dirs `v3-…` |
| Run dirs (every model, any method) | **`runs/[<group>/]v<R>-<method>-[<teacher>-]d<depth>w<width>-s<seed>[-<tag>]`**. `v<R>` is the game rules the model was **trained for**, and always comes first. `<method>` is `bc` or `es` (later also `distill`, `quant`). Examples: `v2-bc-duck-d2w8-s1`, `v2-es-d2w32-s1-t1m`, `v3-bc-duck-d2w32-s1`. Demo dirs: `v<R>-demos-<teacher>-s<seeds>` | run dir names, `model.js` `source`, briefs, docs |

- **A model is trained for one game-rules version.** The name prefix, `best.json` `gameVersion` and `model.js` `gameVersion` must agree. Evaluating a model on other rules (for example a v2 model on rules v3) is a cross-rules evaluation: say so explicitly ("v2 model on rules v3").
- **Renamed 2026-10-10** (user order "rename all now"): `bc-v2-h32x32` → `v2-bc-jump-d2w32-s1`, `v2-h32x32-1m` → `v2-es-d2w32-s1-t1m`, `v2-h32x32` → `v2-es-d2w32-s1`, and `exp-bc-duck/<shape>-s<n>` → `exp-bc-duck/v2-bc-duck-<shape>-s<n>`, `exp-bc-duck/jump-d2w32-s2` → `exp-bc-duck/v2-bc-jump-d2w32-s2`, `exp-bc-duck/demos` → `exp-bc-duck/v2-demos-duck-s1-40`, `exp-bc-duck/heldout-demos` → `exp-bc-duck/v2-demos-duck-s41-60`. Gitignored local research dirs (for example `runs/exp-bc-sweep/*`) keep their old names because pre-registrations cite them; all of them are rules v2.
- Avoid "T"-prefixed labels for teachers. "Type T/M" (ES stuck θs) and "T1/T2" (sweep agreement thresholds) already use them.

**One source of game logic.** `core.js` is the DOM-free simulation, and the trainer, tests, browser and dashboard all run it.
- `createGame({ rng, version, hooks })` creates independent games, many per thread.
- `step(keys)` = `applyKeys` + `update` with dt = 1, on a simulated clock `now = frame * FRAME_MS`.
- The core must contain no DOM, clock, storage, audio or `Math.random`; `test/parity.test.js` enforces this.
- Gameplay sizes are computed from the sprite pixel-map rows, not from canvases.

**Dual-format files.** Every shared file (`core.js`, `policy.js`, `render.js`, `netview.js`, `model.js`, `models.js`, `models-css/index.js`) works both as a classic `<script>` that sets a global (`DinoCore`, `DinoPolicy`, `DinoRender`, `DinoNetView`, `window.DINO_MODEL`, `DINO_MODELS`, `DINO_MODEL_INDEX`) and as a Node `require()`. That is because `file://` blocks ES modules and `fetch()`. Keep that pattern: no `import`/`export`, no JSON loading in the browser.

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
- Model selector: `?model=<id>` (an id in `models.js`), else the choice remembered in localStorage `dino-model` (shared with the CSS page), else the root `model.js`. Picking one stores it and reloads with `?model=<id>`, keeping `?game=`. The page plays the selected model's rules unless `?game=` overrides.
- Speed selector (1×–4096×) applies to AI mode only and is disabled while AI is off, so human play and HI stay at 1×. Sound is muted above 2×.
- Each display frame adds speed × real time to the fixed-timestep accumulator. Updates stop after a 12 ms step budget (`STEP_BUDGET_MS`), and the backlog is dropped. The page shows the achieved speed, measured as simulated time over real time.
- The game clock (`clock`) advances `FRAME_MS` per update and is passed to `update`, `applyKeys`, `pressJump` and `draw`, so the restart lockout, night, flash and blink scale with game time at any speed.
- `render.js` is the canvas renderer shared with the dashboard playback.
- **Network view** (`netview.js`, `DinoNetView`): the one owner of the network drawing, used under the game on `index.html` and `index-css.html` and in the dashboard playback. `createForward(model)` is a forward pass that keeps every layer's activations (`test/netview.test.js` checks its argmax equals `createNet` for every bundled model and shapes up to 16 × 32). `createView(canvas, opts)` gives `setModel`, `idle`, `capture(x, chosen, check)` (counts and shows a warning when its argmax differs from the applied action) and `draw()`; `bindToggle` is the show/hide button (localStorage `dino-netview`).
  - It never decides and stays off the simulation path: `game.js` observes a decision only after the agent made it (before the keys are applied), on the first and the expected last update of a display frame, and draws at most once per display frame. With AI off it shows the model's shape only.

**CSS network page.** `index-css.html` + `css-nn.js` evaluate the same MLP in the browser's style engine. `model.css` is generated from `model.js` by `trainer/css-model.js` via `export` or `export-css`; never edit it by hand. JavaScript only writes `--x0..--x43` and reads `--act`.
- The page plays `model.css` by default. A selected model (`?model=`, or the remembered choice) swaps the `<link>` to `models-css/<id>.css`, and the page waits until `--nn-model-id` equals that entry's `cssId` before it plays. Every decision comes from the stylesheet, and the page never loads the root `model.js`.
- The network view is the one exception to "no weights in JavaScript": the page loads `models.js` for the view only, matching the loaded stylesheet by `--nn-model-id` = `cssId`. Once per display frame it runs the JS forward pass on the `--x` inputs of the latest CSS decision and warns if its argmax differs from `--act`. A stylesheet not in `models.js` (for example a candidate in `test/css-agreement.js`) gets no view.
- Speed goes from 1× to 4096× with a 12 ms step budget. The status line shows the achieved speed, which falls far short at high settings: at 4096× in headless Chrome on the dev Mac (2026-10-10), about 65× for 2×8, 13× for 2×32 and 99× for 1×6.
- `test/css-agreement.js` copies `models-css/index.js` into its scratch page; with no `?model=` and a fresh profile, the page still plays the candidate's `model.css`.

**Game versions.** `v1` is the original rules and `createGame`'s default. `v2` is identical below 1k points, then ramps difficulty on a log scale up to 1M. All training so far uses v2. The version is stored in checkpoints, `best.json` and `model.js`; the page (and `index-css.html`) plays the model's version, and `?game=v1|v2|v3` overrides it. `assess --game <v>` plays other rules than the model's: a cross-rules assessment, so it warns, and the `assess.jsonl` record carries `gameVersion` and `modelGameVersion`.

**Rules v3** (`research/game-v3.md`): built to be effectively impossible for humans past 100k, while staying legal and solvable.
- **Definition (user ruling, 2026-10-10).** The reaction budget of a must-act obstacle is the time from its first visible frame (x < W) to the latest frame on which a valid action can still start: the most lenient of the clearing jumps and, for a mid pterodactyl, the duck. Past 100k, most must-act obstacles must have a budget below 150 ms (9 frames), tightening with score. An exact solver must never crash.
- **Engine.** Exactly rules v2 up to 100,000 points (proven by `test/v2-parity.test.js` and `test/v3.test.js`). From there a second level `e = log10(score/100k) / log10(V3.endScore/100k)` sets the speed cap. It **steps** from v2's 19 to `V3.stepSpeed` (41) at 100k (user ruling "Step at 100k"), then rises to `V3.maxSpeed` (60) at 1M on the log scale (`curve` 1). Every other rule keeps following the v2 schedule.
- **The step** never lands on obstacles placed at the old speed: their frame spacing would shrink below one airtime. Once e > 0, spawning pauses until the track is empty and the dino has landed (1–48 frames). The speed then jumps in one frame and spawning resumes (`stepPending` in `core.js`; tested in `test/v3.test.js`). See `rulesAt(d, r, e)`, `superLevel()` and `V3` in `core.js`; `game.superLevel` exposes e.
- **Why speed is the lever.** `jumpVelocity` grows with speed, but `maxJumpY`/`dropVel` cap the arc, so airtime stays about 32 frames at every speed. The budget is about (W − dino front)/v minus the frames needed to clear the obstacle's height, so only obstacle speed moves it. Above about 25 px/frame the fairness spacing (one airtime per obstacle) sets obstacle density, not the gap coefficients.
- **Fairness change (v3 only, above 100k).** Each obstacle spawned with e > 0 is frozen with `o.late = true`, and `requirement()` anchors its takeoff in the last `minWindow` frames of the window instead of the centre. With the centre, the canonical (solver) takeoff at v3 speeds would come before the obstacle appears, which a legal-input student can never imitate. With the late anchor it comes after the obstacle appears, on every obstacle (the probe's `canonicalBeforeVisible` is 0).
- **Probe.** `node research/scripts/game-v3/budget.js` (single thread, about a second) prints the budget distribution per rules version and start level. `--set maxSpeed=..,curve=..` explores other endpoints in-process.
- **Survey.** `research/scripts/prereg-teacher-duck/survey.js --game v3 --start 100000` runs a solver on rules v3; the dashboard shows the file's game.

**Trainer** (`trainer/`):
- `es.js`: OpenAI-ES with antithetic sampling, centred ranks and Adam.
- `pool.js` / `worker.js`: worker_threads. Only θ and noise seeds cross threads, and workers regenerate the noise.
- `env.js`: episodes and seed sets, in three disjoint ranges:
  - training seeds;
  - validation seeds, used to select the best model;
  - held-out seeds (`EVAL_SEEDS`), used only by `assess` and the stop check.
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
- Speed: the trainer runs the MLP forward pass as WASM SIMD (`trainer/wasm-net.js`). It is bit-identical to JS (verified on arm64 and x86) and about 1.65–1.7× faster. The browser always uses `policy.js`.
  - On by default (since 2026-10-12) in every command that plays the network: `train`, `session`, `finish`, `assess`, `bench`, and the validation in `bc-train`, `bc-train-batch` and `bc-quantize`; also in `research/scripts/prereg-bc-robustness/scan.js`. `--no-wasm` runs them on JS. `--wasm` is still accepted.
  - It is a runtime choice, not a training setting: checkpoints and `best.json` no longer record it, and on `--resume` the current command line decides (older checkpoints' `config.wasm: false` is dropped). `bit-identity.js` skips that key.
  - Without wasm SIMD these commands fail at startup and name `--no-wasm`; there is no silent fallback.
- Selection and stopping: `--select median|count`, `--stop-rule median|allpass[:N]`, `--val-seeds N`, `--fold-validation` (validates every generation inside the training batch), `--eval-when-fit-max`.
- ES:
  - `--rank-ties index|mid`;
  - `--decay coupled|decoupled`: with decoupled, use `--weight-decay` about 0.01;
  - `--sigma-scale none|fanin`, `--sigma-anneal` with `--sigma-min`, `--pair-episodes`;
  - `--stall-rule`: not recommended, kept only to reproduce an old run.
- `assess --seeds N --seed-offset K` scores held-out seeds K..K+N-1 and reports a 95% Clopper–Pearson interval for the pass rate. The stop check uses the first N held-out seeds, so certify a run from an offset past them.

**Verification tools.**
- `node test/bit-identity.js <oldTree> <newTree>` runs 7 small training scenarios, including one that fires the `allpass` held-out stop, in two checkouts (for example a `git worktree` at the pre-change commit, and yours) and diffs metrics, checkpoints and `best.json` (about 9 s). It refuses to start while another trainer is running unless `--allow-busy`.
- A new trainer flag must leave defaults IDENTICAL against the pre-change commit.
- `bit-identity.js --core-changed` is for a deliberate `core.js`/`policy.js` edit (for example rules v3): the coreHash fields are reported, not compared, and everything else must still be identical.

**Findings that constrain defaults.**
- **Shape cap (user order, 2026-10-10):** at most 16 hidden layers and at most 32 units per layer in any run, sweep or brief. The code stays generic over `--hidden`.
- Plateau escape is seed-dependent: on 32,32 with `--rank-ties mid --decay decoupled --weight-decay 0.01`, about 25% of seeds escape by generation 900 and 80–90% by generation 1500.
- Decoupled weight decay 0.1 slows the climb; use about 0.01.
- A tie-rate σ controller was tried and removed: σ ran away, because ties rise again at large σ.
- Research notes live in `research/`; multi-session handoffs live in `handoff/`.

## Invariants to preserve

- **The dashboard sees every training run on this computer.** This is the user's requirement (2026-10-10). `node trainer/dashboard.js` must list and show every training run under `runs/`, while it is running and after it finishes:
  - ES runs (`train`) and behaviour-cloning runs (`bc-gen` / `bc-train`, any teacher), including runs nested one level down (`runs/<group>/<run>`);
  - their metrics and log, with live updates while running, and playback of their models.
  - game scans and teacher surveys (`scan.js` / `survey.js` JSONL anywhere under `runs/`), running and finished, with pass/fail counts, failure scores and the failure-rate CP interval (`trainer/scans.js`).

  **Any new training mode, metrics file or run-dir layout must ship with dashboard support in the same change.** A change that leaves a kind of training run invisible on the dashboard is incomplete.

- **Parity gate (rules v2).** `test/golden-v2/*.json.gz` were recorded from commit `10cd0e0` (before rules v3) by `test/record-golden-v2.js`, which loads that commit's `core.js` and `test/solver.js`. They cover 0 → 1.6k, 30k (groups of 4), 148k → 151k and 998k → 1M, with both solvers and random keys. `test/v2-parity.test.js` replays them on rules v2, and `test/v3.test.js` replays the part below 100k on rules v3. The same rule applies: never re-record them to make a test pass.
- **Parity gate.** `test/golden/*.json.gz` were recorded from baseline commit `2da34de` by `test/record-golden.js`, and v1 must reproduce them frame-for-frame. That includes the order of RNG draws: clouds, stars and the idle blink share the game RNG, and the obstacle sequence depends on them. Any core change that alters v1 behaviour breaks `parity.test.js`. Don't re-record the traces to make it pass.
- **Core hash.** It covers `core.js` and `policy.js`. Editing either one invalidates existing checkpoints, which then `--resume` only with `--force`. Use `--force` only when decisions are provably unchanged; `test/net.test.js` checks bit-identical decisions for the shipped model.
  - The rules-v3 change moved the hash from `0f329d4e…`. For rules v1/v2 work, `--force` on pre-v3 checkpoints and `bc-train --force` on pre-v3 demos are justified: the v1 and v2 goldens, the jump and duck demos (sha256 `3a67bfd4…` and `363872f6…` regenerated byte for byte), `net.test.js` and `bit-identity.js --core-changed` all prove v1/v2 games and decisions unchanged. Never reuse pre-v3 artifacts for rules v3.
- **Locked on resume.** `hidden`, `actionRepeat` and `gameVersion` can't change on resume; other options can be overridden.
- **Evaluation uses the unmodified game.** No changes to CFG, physics, collision or fairness for evaluation. Any curriculum is training-only.
- **Committed vs ignored.** In `runs/<name>/`, `best.json`, `assess.jsonl` (legacy `evals.jsonl`) and `metrics.*` are committed, and for BC runs also `demos.json` and `bc-metrics.jsonl`. Checkpoints, logs, locks and `demos.bin` are gitignored.

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

### 2026-10-10 (later): ship the duck-generation clone `runs/exp-bc-duck/v2-bc-duck-d2w8-s1`

**Decision.** `model.js` and `model.css` now hold `runs/exp-bc-duck/v2-bc-duck-d2w8-s1/best.json`: a 44-8-8-3 clone of `teacher-duck` (`createDuckSolver`). It jumps cacti and low pterodactyls, ducks under mid ones, and stays grounded under high ones. The user approved it ("Ship d2w8-s1").

**Evidence.**
- Step 1, `research/prereg-teacher-duck.md`: the duck solver crashed 0 times in 5,000 full games, with 0 fallbacks.
- Step 2, `research/prereg-teacher-duck-step2.md`: tier A 25/25, tier B 0/1,000, duck share 1.000 vs the teacher.
- Certification, `research/prereg-teacher-duck-cert.md`: 150/150 on offsets 325–474, paired with the jump clone, which also scored 150/150. McNemar b = c = 0, so this is a tie on reliability and a behaviour change only.
- CSS: `test/css-agreement.js` found 0 disagreements over 25,446 frames, 421 of them duck frames, on the root after export. `browser-check.js` passed 38/38.

**Caveat.** At d2w32, one of two seeds got stuck ducking (204/1,000). How often that happens is being measured in `research/prereg-teacher-duck-stuck.md`. The shipped 2×8 model is certified, but retraining it should be screened.

**Reversibility.** Run `node trainer/cli.js export --dir runs/v2-bc-jump-d2w32-s1 --which best`, then the browser check.

### 2026-10-10: ship a behaviour clone of the solver instead of the ES model

**Problem.** The goal is an AI that reliably reaches 1,000,000 points in v2. The shipped ES model (`runs/v2-es-d2w32-s1-t1m`, generation 970) passed 72/75 fresh held-out games on offsets 100–174. Its failures were early, at 460–740 points (the first pterodactyls), plus a rare late crash near 994k. A day of ES work (σ control, tie handling, curricula, seed racing) produced no model better than it. Pure ES reaches the second wall slowly or not at all on most seeds. The open question was whether the 44 inputs were the limit, or the training method was.

**Constraints at the time.**
- `policy.js` observations and the anti-cheat rules were fixed. The model may not observe hidden state.
- The model has to run as a `dino-mlp-v1` MLP in JS and in the CSS evaluator.
- Training is CPU-only. An ES run that escapes the plateau takes a seed lottery of about 1,200 generations, and most escaped seeds then stall at the second wall.

**Evidence.**
- `research/feature-sufficiency.md`, probe B: MLPs trained by supervised learning on 2.1M frames of solver demonstrations, using only the unchanged 44 features, played 352 full games to 1M with 0 crashes. This included 44-8-8-3 networks.
- Paired certification on fresh held-out offsets 175–324 (150 seeds, target 1M, v2, core hash `0f329d4e…`), using `node trainer/cli.js eval --model <m> --target 1000000 --seed-offset 175 --seeds 150`. Raw output is in `runs/exp-bc-clones/cert-175-324-*.txt` (gitignored); the clone's record is in `runs/v2-bc-jump-d2w32-s1/evals.jsonl`.

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

**Decision.** `model.js` and `model.css` now hold `runs/v2-bc-jump-d2w32-s1/best.json`: a 44-32-32-3 behaviour clone of `test/solver.js` (variant `base`, `trainB.py` with SEED=1). The ES trainer remains the research pipeline. The project's claim changes from "neuroevolution-trained" to "distilled from an exact solver, on legal inputs".

**Reversibility.** Cheap to reverse. Run `node trainer/cli.js export --dir runs/v2-es-d2w32-s1-t1m --which best`, which also writes `model.css`, then run the browser check. The model format, the inputs and the game are unchanged. The clone's HI key and game version are the same as before.

**What would change this.**
- An ES model, or an ES-fine-tuned clone, beats the shipped clone on ≥ 150 fresh held-out seeds.
- A failure mode appears in the clone that the solver doesn't have, for example covariate shift in states the solver never visits.
- The user decides the AI must learn from the score alone.
- Any change to `observe()`, `core.js` or the game version invalidates the clone. Regenerate the demonstrations and retrain, then re-certify.
