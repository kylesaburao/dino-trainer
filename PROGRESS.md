# PROGRESS

**Status: scaffolding complete; training not started.**

Spec: `TrainingPrompt.md` (unchanged). This session's scope: `ScaffoldingPrompt.md` covers Phases 1–3 in full and the Phase 4 trainer built and smoke-tested, but no real training. `model.js` currently holds an **untrained** model from the smoke run; I make no claim about its skill. It crashes at the first obstacle (held-out median 40).

| Item | Value |
|---|---|
| Target / metric | 10000 (default); **median** score over the 25 fixed held-out seeds (`EVAL_SEEDS` in `trainer/env.js`), episodes capped at the target |
| Best training score | n/a (no real training yet) |
| Held-out median / min / max | 40 / 40 / 52 for the smoke model (untrained, generation 1) |
| Generation / cumulative wall time | 0 / 0 for real training (the smoke runs are throwaway, in `runs/smoke*`, gitignored) |
| Workers detected | **18** (`os.availableParallelism()`) |
| Benchmark, 1 core, simulation only | **~7.4M frames/s** (7,379,624) |
| Benchmark, 1 core, simulation + policy | **~1.3M frames/s** (1,282,874) |
| Benchmark, 18 workers, simulation + policy | **~17.4M frames/s** total (~0.97M per worker) |

Benchmark command: `node trainer/cli.js bench --seconds 5`. It replays the inputs of the 9000-frame golden trace, which covers every speed up to the maximum, pterodactyls and night, so the numbers reflect real gameplay rather than early crashes. A 10,000-point episode is about 33k frames, so one core runs about 40 such episodes per second with the policy. The MLP forward pass dominates the cost: the core alone is about 6× faster.

## Start real training (next session)

Run these from the repo root. The trainer runs in the background with its log in the run directory, and stops itself at the time budget. The 70-minute budget fits the 1h30 limit in TrainingPrompt.md, leaving the final ~15 minutes reserved.

```sh
mkdir -p runs/main && nohup node trainer/cli.js train --target 10000 --time-budget 70 > runs/main/train.log 2>&1 &
tail -f runs/main/train.log           # or watch the dashboard:
node trainer/dashboard.js             # -> http://127.0.0.1:8080 (live charts + live playback)
```

To resume, raising `--target` if needed (this continues the generation count, Adam state and RNG stream):

```sh
nohup node trainer/cli.js train --resume --target 10000 --time-budget 70 >> runs/main/train.log 2>&1 &
```

Stop safely with `kill -INT <pid>` (the PID is in `runs/main/train.lock`). This saves the last completed generation and releases the lock. Then:

```sh
node trainer/cli.js eval --dir runs/main --which best          # 25 held-out seeds
node trainer/cli.js export --dir runs/main --which best        # writes model.js
node test/browser-check.js                                      # verify AI mode in headless Chrome
```

## What was built

| File | Role |
|---|---|
| `core.js` | DOM-free simulation core: all game rules, CFG, sprite pixel maps (gameplay sizes are computed from them), the fairness model, `createRng` (mulberry32) and `createGame({ rng, hooks })`. Input API: `pressJump(now)`, `releaseJump()`, `pressDown()`, `releaseDown()`, `update(ms, now)`; `applyKeys(keys, now)` is the per-frame key-state interface; `step(keys)` = applyKeys + update with dt = 1 on a simulated clock `now = frameIndex * FRAME_MS`. No `Math.random`, DOM, clock, storage or audio (checked by a test). Classic script (`DinoCore`) + `require`. |
| `policy.js` | The one shared observation + MLP inference + agent (action repeat, idle→start). Used by the trainer workers, `eval`, the browser AI mode and the dashboard. Classic script (`DinoPolicy`) + `require`. |
| `render.js` | Canvas renderer (sprites, font, ground, frame drawing), shared by the game and the dashboard playback. Classic script (`DinoRender`) + `require`. |
| `game.js` | Thin browser shell: sound, input events, localStorage HI, the fixed-timestep loop, the AI toggle. |
| `model.js` | Exported model: sets `window.DINO_MODEL` and `module.exports`. Loaded with a plain `<script>`; no JSON or fetch, so it works over `file://`. |
| `trainer/cli.js` | `train`, `eval`, `bench`, `export`. |
| `trainer/{env,es,pool,worker,store,bench}.js` | Episode runner and seed sets, OpenAI-ES maths, the worker_threads pool, worker, checkpoint/lock persistence, benchmark. |
| `trainer/dashboard.{js,html}` | Live training dashboard (added at the user's request), read-only. |
| `test/record-golden.js`, `test/golden/*.json.gz` | Golden traces recorded from **baseline 2da34de** itself. |
| `test/parity.test.js`, `test/policy.test.js` | `node --test test/` (22 tests). |
| `test/smoke-trainer.js` | End-to-end pipeline smoke test (16 checks). |
| `test/browser-check.js` | Headless Chrome over CDP (27 checks). |

## Test and smoke results (this session)

- **Parity, `node --test test/`: 22/22 pass.** There are 13 golden traces (56k frames) recorded from 2da34de:
  - short taps, held jumps, ducks, fast-drops, and jump+down combinations
  - idle frames, which exercise the blink RNG
  - random keys
  - restart presses at 44, 45 and 46 frames after a crash (the float boundary of the lockout)
  - a 9000-frame run driven by a lookahead script, which reaches max speed 13, pterodactyls, cactus groups and night mode (score ≥ 700)

  Every frame compares, exactly: state, dino y/vy/flags, speed, distance, RNG-call count, and each obstacle's type/x/y/size/w.

  Mutation checks on a scratch copy confirm the gate catches subtle changes: a reordered cloud RNG draw, a missing star RNG draw, `>` changed to `>=` in the restart lockout.
- Observation rules are tested over 40 seeded games, more than 1000 frames. Tampering with any of the following leaves the observation unchanged: obstacles at x ≥ 600 (including obstacles parked off-screen by placeFairly), `o.gap`, and fake off-screen obstacles. An obstacle at x < 600 does change it. Features stay within |x| ≤ 2.
- **Browser, `node test/browser-check.js`: 27/27 pass.**
  - With AI off, the idle canvas is pixel-identical to 2da34de, with `Math.random` seeded identically in both pages. The canvas has the same position and size.
  - Human play works: Space starts and jumps, scoring, duck, fast-drop, crash, HI saved, restart after the lockout, click-to-jump.
  - The AI toggle works both ways. The toggle click never presses jump, and the button never keeps focus.
  - The AI starts, scores, crashes and auto-restarts. It calls `applyKeys` exactly once per fixed-timestep update, and its presses reach the game through the real input functions.
  - Human jump/duck input is ignored while AI is on.
  - AI runs never write HI, including a run where AI was on only partway.
  - Turning AI off mid-run releases the AI's keys, and the player has control again.
  - The page logs no console errors, warnings or exceptions.
- **Smoke, `node test/smoke-trainer.js`: 16/16 pass.** Settings: population 8, 3 workers, `--time-budget 2`, `--generations 3`.
  - Checkpoints and JSONL/CSV metrics are written.
  - The lockfile holds the trainer's PID, and a second concurrent `train` is refused.
  - SIGINT mid-generation 2 exits 130, saves the last completed generation (1) and releases the lock.
  - `--resume` continues at generation 1 → 2 → 3. The metrics show generations 1, 2, 3, so the count does not restart.
  - A stale lock (dead PID) is replaced.
  - A fresh `train` refuses to overwrite an existing checkpoint.
  - On a temporary copy of the repo with an edited core, `--resume` refuses the core-hash mismatch and `--force` accepts it; the real core's hash is unchanged.
  - `eval` runs the 25 held-out seeds (median 40, min 40, max 52).
  - `export` writes `model.js`, and evaluating `model.js` gives the same result as `best.json`.

  The smoke run uses `--episodes 3000` per candidate so one generation takes about 2 s and SIGINT can land mid-generation. That is about 6.5M frames per generation, a few seconds of compute.
- Dashboard: checked in headless Chrome with a second throwaway 3-generation run (`runs/smoke-dash`). It shows live status, tiles, charts and the log tail. Playback advances in real time, with no console errors.

## Design decisions (with reasons)

- **Golden traces come from the real baseline.** `test/record-golden.js` runs `git show 2da34de:game.js` in a Node `vm` with the DOM stubbed, the clock stubbed (`now = frameIndex * FRAME_MS` for both `update` and `performance.now`), and `Math.random` replaced by the seeded PRNG after load. The ground texture is drawn at load, so swapping afterwards keeps it out of the game's random stream. The stored inputs, not the controller that generated them, are replayed on the core. Traces are gzipped JSON (about 0.5 MB in total).
- **One RNG stream, as in the baseline.** Clouds, stars and the idle blink stay in the core and draw from the game's `rng` in exactly the baseline order: the baseline's obstacle sequence depends on those cosmetic draws. Only the ground texture, drawn once in the shell, still uses `Math.random`.
- **State the baseline kept across restarts stays per game.** `clouds`, `groundX`, `moonX`, `stars`, `nightAlpha` and `downHeld` live once per `createGame`, not in `reset()`, which preserves restart behaviour.
- **Sizes come from pixel maps.** They are computed as `maxRowLength × 2` by `rows × 2`, and a test asserts the baseline values: STAND_TOP 98, duck height 24, small cactus 16×34, large cactus 24×50, ptero 46×20, cloud width 46.
- **Fixed key-transition order.** `applyKeys` applies a frame's key changes as releases before presses, and jump before down. The golden recorder uses the same order.
- **Observations, 44 features:**
  - For the dino: height above standing, vy, jumping, ducking, normalised speed.
  - For each of the 3 nearest obstacles with **x < 600** that the dino has not passed: present, relative x of the hitbox, hitbox y/w/h, type one-hot, group size, pterodactyl height class (one-hot), and on-screen velocity (speed + speedOffset, which a player sees as motion).
  - An absent obstacle is encoded as "one screen away".
  - Never observed: `o.gap`, spawn history, the solver (`requirement`/`placeFairly`; the core does not even export them), or RNG state.
  - The reward is the capped score (fractional distance × 0.025), with no shaping.
- **Actions:** {none, jump held, down held} each frame. `actionRepeat` is configurable (default 1) and stored with the model, so the browser applies the same repeat.
- **Model selection uses a separate validation set.** The best model is chosen on 25 *validation* seeds in [2^30, 2^31). The 25 *held-out* seeds in [2^31, 2^32) are used only by `eval` and by the stop check (held-out median ≥ target, run only once validation reaches the target). Training seeds are in [0, 2^30), so the three sets are disjoint by construction. This deviates slightly from "best by held-out"; the reason is that picking the best by held-out score would leak those seeds into model selection.
- **The core hash covers core.js and policy.js.** TrainingPrompt.md says "core source". A change to the observations invalidates trained weights just as a rules change does, so policy.js is included. `render.js`, `game.js` and the trainer are excluded because they cannot change what the weights mean.
- **Trainer:**
  - OpenAI-ES with antithetic sampling, centred ranks, Adam (lr 0.03) and L2 decay 0.005, σ = 0.1.
  - Population 128 × 4 episodes per candidate, all candidates sharing the same episode seeds within a generation.
  - MLP 44-16-3 with tanh (771 parameters), plain Node, no dependencies.
  - Noise vectors are regenerated from per-candidate seeds in the workers and in the main thread, so only seeds and θ cross threads.
- **Checkpoints:**
  - Written atomically: temp file, fsync, rename.
  - Contents: θ, Adam (m, v, t), generation, trainer RNG state, the full config, core hash, best model (params + validation scores), cumulative wall time.
  - Saved every `--checkpoint-every` generations (default 5), at the end, on SIGINT/SIGTERM, and when the time budget timer fires.
  - Only state committed at a generation boundary is saved. The commit is synchronous, so a signal can never capture a half-updated state.
  - On resume, `hidden` and `actionRepeat` are locked. Everything else (target, population, σ, lr, workers, budget, generations) can be overridden.
- **Lockfile:** `runs/<name>/train.lock` is created with `O_EXCL`. A live PID (`kill -0`) is refused; a dead PID is treated as stale and replaced. It is released on every exit path.
- **Browser AI mode:**
  - The policy runs *inside* the fixed-timestep loop, immediately before each `update`, and calls `game.applyKeys`, which drives the same `pressJump`/`releaseJump`/`pressDown`/`releaseDown` functions as a human. This keeps the browser identical to headless play at any refresh rate.
  - **Auto-restart chosen:** after a game over the AI taps jump; presses during the 750 ms lockout are ignored by the core, exactly as for a human. All human jump/duck input is ignored while AI is on.
  - A run counts as an AI run if AI was on at any point during it, and AI runs never write HI.
  - From idle, the AI's first action presses jump, exactly like the first frame of a training episode. A restart after a game over begins without that initial jump, as for a human restarting.
- **Toggle button:** a small `#535353` on white button below the canvas. It is wrapped together with the canvas in a column container so the canvas keeps its position (verified pixel-identical). Its pointer events never reach the jump handler, and it blurs after a click, so Space can't activate it later.
- **Shell flash handling (behaviour-preserving deviation):** the baseline's `drawScore` reset `flashStart` to -1 after the flash; the shell now leaves core state alone. The result is visually identical, since an expired flash shows the plain score either way.
- **No curriculum** is used. All evaluation runs the unmodified game (CFG, physics, collision and fairness untouched).
- **Additions requested by the user:**
  - `trainer/dashboard.js`, a read-only local server for watching runs, with charts, log tail and **live playback** of the run's newest weights through the real core, policy and renderer. The trainer prints the dashboard command when it starts.
  - Extracting `render.js`, so the playback and the game share one renderer.
  - `--mute-audio` in the browser test, because headless Chrome played the game's beeps aloud.
- `ScaffoldingPrompt.md` and `TrainingPrompt.md` are committed unchanged.

## Next steps

1. Start real training with the command above and watch `runs/main/train.log` or the dashboard.
2. If fitness plateaus early, try:
   - more episodes per candidate (noisy fitness), e.g. `--episodes 8`
   - a larger population (`--pop 256`)
   - a smaller σ (0.05)
   - `--action-repeat 2` (halves inference cost; must be chosen at the start of a run)
3. At the end: stop safely, run `eval`, `export`, `node test/browser-check.js`, update this file, then commit.
