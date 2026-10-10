# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Chrome-dino clone (plain JS, opened over `file://`) with a neuroevolution-trained "AI mode". Training runs in plain Node with **no npm dependencies**.

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

Training outlasts foreground limits, so run it detached and log to the run directory. To resume the shipped run:

```sh
nohup node trainer/cli.js session --resume --target 1000000 --time-budget 70 --auto-export --dir runs/v2-h32x32-1m >> runs/v2-h32x32-1m/session.log 2>&1 &
kill -INT <pid>    # PID is in runs/<dir>/train.lock; saves a checkpoint and releases the lock
```

The CLI's `--dir` default (`runs/main`) and `--hidden` default (`16`) are stale. `runs/main` no longer exists, and the shipped model is `runs/v2-h32x32-1m` with `--hidden 32,32`. Pass both explicitly.

`--auto-export` overwrites the shipped `model.js` and `model.css` on every new best validation score, so leave it off for experimental runs. `browser-check.js` checks that `model.css` matches `model.js`; after you replace `model.js`, run `export-css`.

## Architecture

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
- The reward is the capped score, with no shaping.

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
- The metric is the median score over the 25 held-out seeds, with episodes capped at `--target`.
- `store.js`: atomic writes (temp file, fsync, rename), the `train.lock` PID lockfile, and `coreHash`.
- `ops.js`: report, finish and session.

## Invariants to preserve

- **Parity gate.** `test/golden/*.json.gz` were recorded from baseline commit `2da34de` by `test/record-golden.js`, and v1 must reproduce them frame-for-frame. That includes the order of RNG draws: clouds, stars and the idle blink share the game RNG, and the obstacle sequence depends on them. Any core change that alters v1 behaviour breaks `parity.test.js`. Don't re-record the traces to make it pass.
- **Core hash.** It covers `core.js` and `policy.js`. Editing either one invalidates existing checkpoints, which then `--resume` only with `--force`. Use `--force` only when decisions are provably unchanged; `test/net.test.js` checks bit-identical decisions for the shipped model.
- **Locked on resume.** `hidden`, `actionRepeat` and `gameVersion` can't change on resume; other options can be overridden.
- **Evaluation uses the unmodified game.** No changes to CFG, physics, collision or fairness for evaluation. Any curriculum is training-only.
- **Committed vs ignored.** In `runs/<name>/`, `best.json`, `evals.jsonl` and `metrics.*` are committed. Checkpoints, logs and locks are gitignored. Never push.
