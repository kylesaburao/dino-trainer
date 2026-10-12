# Dino game with an AI mode

A Chrome-dino clone in plain JavaScript with an "AI mode" driven by a small neural network (MLP). The game and the trainers run in plain Node with no npm dependencies. Design notes and invariants are in `AGENTS.md`.

## Setup

Playing needs only a browser, and the Node trainer needs only Node (v22+; v26 is used here). The fast behaviour-cloning backends need Python; `setup.sh` sets that up:
```sh
./setup.sh --check || ./setup.sh   # check (read-only, no network); set up only if something is missing or wrong
```
- `.venv`: the numpy backend (`--backend python`), pinned in `requirements.txt` (numpy 2.0.2). It needs CPython 3.9–3.12 (3.9 preferred; on a Mac the Command Line Tools' `/usr/bin/python3` is 3.9). Every platform.
- `.venv-mlx`: the Apple GPU backend (`--backend mlx`), pinned in `requirements-mlx.txt` (mlx 0.32.3, numpy 2.5.4). It needs macOS 14+ on Apple silicon and CPython 3.12–3.14 (pyenv 3.14.8 preferred; the script prints the `pyenv install` command if it is missing). Skipped on other platforms.
- On Linux/WSL only `.venv` is set up. numpy uses OpenBLAS there, so numpy-backend params are not expected to match the Mac's bit for bit (`test/bc.test.js` checks its param pins only on macOS arm64). If venv creation fails on Debian/Ubuntu, the script prints the `apt install python3.X-venv` command.
- A venv that passes the check is left untouched. Nothing is deleted: `--recreate` moves an old venv aside to `<dir>.bak-<UTC time>`. No sudo; system Python is only used as the base interpreter.
- `--verify` runs `node --test test/bc.test.js` against the venvs afterwards. `./setup.sh --help` lists every option (`--numpy-only`, `--mlx-only`, `--venv-dir`, `--python`, ...).
- **Git worktrees** have no venvs of their own. Point the trainer at the main checkout's (`setup.sh` prints these lines; `--check` then checks those interpreters):
  ```sh
  export PYTHON=/path/to/dino-game/.venv/bin/python
  export PYTHON_MLX=/path/to/dino-game/.venv-mlx/bin/python
  ```

## Play

Open `index.html` in a browser, directly over `file://`. There is no build step. Toggle AI mode on the page to watch the shipped model play. `index-css.html` runs the same network in the browser's CSS engine. On both pages you can pick another bundled model trained for the latest rules (the other rules-v3 clones) and fast-forward the AI from 1× to 4096×; the page shows the speed it actually achieves. Under the game, a network view (as on the dashboard) draws the AI's latest decision: activations, weights and the chosen action; a button hides it. `node trainer/cli.js export-models` regenerates the bundled models.

**Game rules.** Both pages always play the latest rules, currently rules v3; there is no option to pick other rules, and the model selector lists only models trained for the latest rules. The high score is kept per rules version. The training dashboard plays each run on the rules it was trained for. The rules versions:
- **rules v1**: the original game; difficulty stops rising at about 1,700 points.
- **rules v2**: v1 up to 1,000 points, then harder on a log scale up to 1,000,000 points (speed cap 13 → 22, more pterodactyls, tighter gaps).
- **rules v3**: v2 up to 100,000 points. At 100k the speed jumps from 19 to 41 px/frame (after a moment of empty track), then climbs to 60 by 1,000,000. From the jump on, most obstacles leave less than 150 ms between appearing on screen and the last moment to jump or duck, beyond human reaction time; every sequence stays solvable by an exact solver. See `research/game-v3.md`.

To score a model on other rules than it was trained for: `node trainer/cli.js assess --dir <run> --game v3 ...` (it warns that this is a cross-rules assessment). Train for rules v3 with `--game v3` (`train`, `bc-gen`).

## Train a model

The shipped model is a **behaviour clone**: a small MLP trained by supervised learning to copy an exact solver (`test/solver.js` `createDuckSolver`, which jumps and ducks). The model sees only the 44 allowed `observe()` features. Training takes two steps: generate demonstrations, then train on them.

**Requirements**
- Node (v22+; v26 is used here).
- For the fast backends: the Python venvs from `./setup.sh` (see [Setup](#setup)). `PYTHON` / `PYTHON_MLX` point the trainer at other interpreters (needed in a git worktree).

**1. Generate demonstrations** (seconds):
```sh
node trainer/cli.js bc-gen --dir runs/exp-bc-mine                  # teacher-jump (default): jumps only
node trainer/cli.js bc-gen --dir runs/exp-bc-mine --teacher duck   # teacher-duck: also ducks under mid pterodactyls
```
This writes `demos.bin` (gitignored) and `demos.json` (the manifest: teacher, row counts per class, sha256).

**2. Train** (always pass `--hidden`; the CLI default is stale):
```sh
node trainer/cli.js bc-train --dir runs/exp-bc-mine --hidden 32,32 --backend python --seed 1   # about 1 s/epoch
node trainer/cli.js bc-train --dir runs/exp-bc-mine --hidden 32,32                             # plain Node, about 10 s/epoch
```
- `--hidden` sets the layers, for example `8`, `8,8` or `32,32`.
- After training it plays 25 validation games to `--target` (default 1,000,000).
- It writes `best.json` (the model) and `bc-metrics.jsonl` (loss and per-class recall for each epoch: none, jump, and duck when present).
- Optional, off by default: `--distill-from <best.json> [--distill-mode soft|hard] [--tau T] [--alpha A]` learns from another model's logits (distillation), `--quant-bits N` trains with quantized weights (`--quant-scheme pow2`: every weight and bias an N-bit integer times one power-of-two scale per layer; `research/scripts/prereg-distill/pack.js` packs it), and `bc-quantize --from <best.json> --dir <run> --quant-bits N` quantizes a trained model after the fact. Details are in the `trainer/cli.js` header.

**3. Assess on held-out seeds** (`assess`; the old name `eval` still works and prints a deprecation notice):
```sh
node trainer/cli.js assess --dir runs/exp-bc-mine --which best --target 1000000 --seeds 25 --seed-offset 0
```
This prints the pass rate with a 95% confidence interval.

**4. Export, only if you mean to replace the shipped model:**
```sh
node trainer/cli.js export --dir runs/exp-bc-mine --which best   # writes model.js and model.css
node test/browser-check.js                                      # headless Chrome check, including CSS/JS agreement
```
`export`, `finish`, `session` and `--auto-export` all overwrite the shipped `model.js`/`model.css`. The shipped model is `runs/exp-bc-duck/v2-bc-duck-d2w8-s1`; the previous jump-only clone is `runs/v2-bc-jump-d2w32-s1`. `node test/css-agreement.js --dir <run>` checks CSS/JS agreement through the duck phase for a candidate without touching the root files.

**Neuroevolution (research pipeline).** The OpenAI-ES trainer learns from the game score alone. It runs for hours, so start it detached:
```sh
mkdir -p runs/exp-es-mine
nohup node trainer/cli.js train --wasm --hidden 32,32 --target 1000000 --time-budget 70 --dir runs/exp-es-mine >> runs/exp-es-mine/train.log 2>&1 &
kill -INT <pid>   # the PID is in runs/exp-es-mine/train.lock; this saves a checkpoint and releases the lock
```
Resume with the same command plus `--resume`. The header comment of `trainer/cli.js` lists every option.

## View the dashboard

```sh
node trainer/dashboard.js                 # then open http://127.0.0.1:8080
node trainer/dashboard.js --port 9000     # use another port
node trainer/dashboard.js --runs /path/to/runs   # read runs from another directory
```
- The dashboard is read-only. It only reads files in each run directory, so it is safe to leave running alongside a trainer.
- It lists the runs under `runs/`, with live training curves, held-out assessments and game playback of a run's model.
- To view a remote machine's dashboard, forward the port, for example `ssh -L 8080:127.0.0.1:8080 <host>`.
- It shows every run under `runs/`, including nested `runs/<group>/<run>` dirs: ES runs (`metrics.jsonl`, checkpoints, evals) and behaviour-cloning runs (`bc-metrics.jsonl` with recall per class, the teacher name, demo manifests, and snapshot epochs you can play). The playback menu also lists every solver in `test/solver.js` (`createSolver`, `createDuckSolver`), so you can watch either teacher play. Reload the page to pick up new solvers.
- A Scans section lists every game scan (`research/scripts/prereg-bc-robustness/scan.js`) and teacher survey (`survey.js`) JSONL under `runs/`, at any depth, with games done, passes, failure scores and a 95% Clopper–Pearson interval on the failure rate. Each one shows as running, stalled or done, using the pid that scan.js records in the file (older files fall back to the file's mtime). A run's view also lists the scans of its `best.json`.

## Tests

```sh
node --test test/                 # unit and parity tests (python-backend tests skip without numpy)
node test/browser-check.js        # headless Chrome; set CHROME to override the macOS Chrome path
node trainer/cli.js bench --seconds 5   # simulated frames per second
```
