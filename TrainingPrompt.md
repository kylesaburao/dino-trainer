GOAL: Train a neural network to play the Chrome-dino clone in
/Users/kyle/Downloads/dino-game, and ship it as an "AI mode" that can be
switched on and off in the browser game. Train in a fast headless simulation
of the real game logic. Training must be resumable so the score target can
be raised later.

## Budget and priorities
- Hard wall-clock limit: 1 hour 30 minutes for this whole goal, measured from
  when you start.
- The score target (default 10000) caps the effort; it is not a minimum.
  Within the time limit, the priorities are:
  1. A working, verified browser AI-mode toggle driven by the best model so
     far.
  2. A resumable training pipeline.
  3. The highest score you can reach.
- Plan your time accordingly:
  - Get the full pipeline working end to end early, with a weak first model
    playing in the browser.
  - Then give the remaining time to training.
  - Reserve the final ~15 minutes to:
    1. stop training safely
    2. run the final evaluation
    3. export the model
    4. verify the browser
    5. update PROGRESS.md
    6. commit

## Baseline (verify first; reconcile any drift before you start)
- Repo: /Users/kyle/Downloads/dino-game. It is a standalone git repo with
  baseline commit 2da34de on main.
- Files:
  - index.html: one 600x150 <canvas id="game"> plus <script src="game.js">.
  - style.css
  - game.js: a single classic (non-module) script with all state in top-level
    globals.
- How game.js works:
  - Fixed-timestep main loop: requestAnimationFrame accumulates elapsed time
    (clamped to 50ms) and calls update(FRAME_MS, now) once per whole 60fps
    frame, so physics always steps with dt = 1. draw(now) renders.
  - State is 'idle', 'running' or 'over'.
  - score() = floor(distance * 0.025); distance grows by speed each frame.
  - Speed ramps from 6 to 13 by 0.001 per frame. An episode reaching 10,000
    points is about 33k frames.
  - Input functions:
    - pressJump() / releaseJump(): Space/↑, click or tap. A longer hold gives
      a higher jump.
    - pressDown() / releaseDown(): ↓ ducks, or fast-drops if mid-air.
  - stepJump(j, dt) is the shared jump physics.
  - The fairness model guarantees every generated obstacle sequence is
    survivable, so 10,000 is reachable in principle. It consists of:
    - requirement(), placeFairly(), jumpArc()
    - RELEASE_FRAMES
    - CFG.minWindowFrames, CFG.minGroundFrames
  - spawnObstacle(), the clouds and the ground use Math.random.

## Phase 1: Headless simulation core (behaviour-preserving refactor)
Split game.js into a pure simulation core and a thin browser shell. The game
logic must have exactly ONE source, with no port and no second copy.
- The core must not touch the DOM. Move these into the shell or inject them:
  - the canvas
  - document.createElement
  - document.body in setNight
  - localStorage
  - AudioContext
  - the event listeners
  - performance.now() (used in pressJump and the restart lockout)
- Gameplay sizes currently come from sprite canvases: S.stand.height,
  obstacle img.width/height, STAND_TOP, the duck height and so on. The core
  must compute these from the pixel-map row data. Rendering stays in the
  shell.
- Expose a factory, for example createGame({ rng }):
  - Many independent games must be able to run in one thread.
  - It takes a seeded, fast, documented PRNG. Math.random must not appear
    inside the core.
  - It has a step API that applies the pressed/released input state for one
    frame, then updates with dt = 1.
- Loading over file://:
  - index.html is opened over file://, where Chrome blocks ES modules and
    fetch().
  - Every shared file must work both as a classic <script> that sets a
    global and as a Node require(). Have each file also assign
    module.exports when that exists.
- Parity gate (must pass before Phase 2):
  1. From baseline commit 2da34de, record golden traces. Stub the clock and
     replace Math.random with the same seeded PRNG.
  2. For each fixed seed and fixed input script, record per frame: dino.y,
     speed, distance, state, and obstacle x/type. The scripts must cover
     short and held jumps, ducks, fast-drops, and runs to a crash.
  3. The refactored core must reproduce these traces exactly.
  4. Commit the traces and a `node` test that checks them.
- The browser game must still play the same with no console errors. Check
  this in headless or real Chrome.
- Add a benchmark command that reports simulated frames per second per core,
  and record the result in PROGRESS.md.

## Phase 2: Environment and shared policy code
- Actions:
  - Applied ONLY through the real input functions, once per frame.
  - Per-frame key state, e.g. {none, jump-held, down-held}, so hold duration
    and fast-drop matter.
  - A frame-skip / action-repeat setting is allowed.
- Observations: only what a player can see:
  - dino y, vy, and the jumping/ducking flags
  - speed
  - the nearest few obstacles with x < 600 (the canvas width): relative x,
    hitbox y, width, height, type/size, and pterodactyl height class
  - normalise every feature
- FORBIDDEN as observations or as reward shaping:
  - obstacles at x ≥ 600. placeFairly pushes obstacles off-screen, so the
    obstacles array leaks the future.
  - requirement() or placeFairly() output (they are a solver)
  - RNG state, o.gap, spawn history, or any other planner internals
- Put observation extraction and MLP inference in ONE shared file, e.g.
  policy.js, in the same classic-script/require dual format. The trainer and
  the browser must both use it, so the browser AI sees exactly what training
  saw.
- Final evaluation uses the unmodified game: no changes to CFG, physics,
  collision or fairness. A training-only curriculum is allowed (e.g. a speed
  cap); document it in PROGRESS.md.

## Phase 3: Browser AI mode (required)
- Add a toggle button, e.g. "AI: OFF" / "AI: ON":
  - Place it below the canvas.
  - Make it small and minimal, in the game's grey (#535353) on white.
  - It must not change the existing look of the game.
- Model format:
  - Export the model as a JS file, e.g. model.js, that assigns a global
    (window.DINO_MODEL = {...}) and also module.exports.
  - Do not use JSON or fetch() — both fail over file://.
  - index.html loads it with a plain <script> tag.
- While AI mode is on, the policy chooses the input state INSIDE the
  fixed-timestep loop, right before each update. This keeps it identical to
  headless at any refresh rate, including 120Hz.
  - Do not decide once per requestAnimationFrame.
  - The AI drives the same pressJump/releaseJump/pressDown/releaseDown
    functions as a human.
- Keyboard and mouse:
  - Clicking the toggle must not also make the dino jump.
  - With AI on, human jump/duck input is ignored. The exception is restart
    after a game over; or auto-restart, at your choice, and document which
    you chose.
- Runs played with AI mode on must NOT overwrite the human "HI" score in
  localStorage.
- Turning AI off mid-run releases any keys the AI is holding and hands
  control back to the player.
- Verify in Chrome (headless is fine):
  - the toggle works both ways
  - the AI plays and scores
  - there are no console errors
  - normal human play is unchanged with AI off
- Get this working with an early checkpoint first. Re-export the best model
  at the end.

## Phase 4: Training
- Suggested default (switch only with a reason recorded in PROGRESS.md): a
  small MLP in plain Node.js with no npm dependencies, trained by
  neuroevolution (OpenAI-ES, CMA-ES or a GA), because evaluating candidates is
  embarrassingly parallel on CPU.
- Parallelism:
  - Use worker_threads.
  - Default worker count: os.availableParallelism(), falling back to
    os.cpus().length.
  - Accept a --workers override.
  - Use all available CPU while training.
- Training will far exceed the 10-minute foreground command limit. Therefore:
  - Run it detached or in the background, with stdout/stderr going to a log
    file.
  - Monitor that log file.
  - Use a lockfile with a PID check so a resume can never start a second
    trainer writing the same checkpoint.
  - Enforce the time limit with a --time-budget option.
- Checkpoints:
  - Write them atomically: write to a temp file, then rename.
  - Write them every N generations and on SIGINT/SIGTERM.
  - Each holds: weights, optimiser state, generation, RNG state, full config,
    and a hash of the simulation-core source.
  - On --resume, refuse a core-hash mismatch unless --force is given.
- CLI commands:
  - `train --target <score> [--resume] [--workers N] [--time-budget <min>]`
  - `eval`
  - `bench`
  - `export` (writes model.js)
  - Raising --target on resume continues from the checkpoint.

## Evaluation and stop condition
- Metric: the MEDIAN score over 25 fixed held-out seeds that are never used
  in training, with episodes capped at the target.
- Stop training at whichever comes first:
  - the held-out median ≥ target
  - the time limit is reached, keeping the final ~15 minutes reserved
- Then, in order:
  1. Save the checkpoint.
  2. Run the final eval.
  3. Export the best model to model.js.
  4. Verify AI mode in Chrome.
  5. Update PROGRESS.md.
  6. Commit.
  7. Stop.
- If the target isn't reached, that is an acceptable result: record the
  achieved score, the blockers, hypotheses, and next steps for a later
  resume.

## Progress log and persistence
- Keep PROGRESS.md at the repo root as the human-readable record. Update it
  at each milestone and at the end of the session. It contains:
  - target and metric
  - best training score
  - held-out median/min/max
  - generation and cumulative wall time
  - benchmark throughput
  - workers
  - full config and curriculum
  - decisions, with reasons
  - the exact resume command
  - next steps
- Also append per-generation metrics in machine-readable form (JSONL or CSV)
  in the training directory.
- Add bulky or intermediate checkpoints and logs to .gitignore.
- Commit, with clear messages, after each phase and at the end:
  - code, tests and golden traces
  - PROGRESS.md
  - the best model and model.js (both are small)
- Do not push.

## Out of scope
- GPU training
- Pixel-based observations
- Changes to game feel, constants or visuals, other than the AI toggle
  button
