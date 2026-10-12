# x86 determinism probe

Pre-registration: `research/prereg-x86-determinism.md` (decision, hypotheses H0/H1/H2, how each result clears job classes (a)–(d)).

| File | What it is |
|---|---|
| `probe.js` | One command, single-threaded, about 15 s on the Mac (M5 Pro) and an estimated 40 s on the desktop. Prints one JSON object (about 24 KB) on stdout, also written to `--out`, and one summary line per stage on stderr. |
| `compare.js` | `node compare.js <reference.json> <candidate.json>`: per-stage verdict, confounds, hypothesis, job classes. Exit 0 = all identical. |
| `mac-reference.json` | The probe's output on the Mac (darwin arm64, node v26.9.0, V8 14.6.202.34-node.32), run with `--python` so S7 is included. |
| `compare-selftest.js` | Mutation control for compare.js (11 cases), on `mac-reference.json` by default. |

The probe writes nothing except `--out` and a scratch dir under the system temp dir (removed on exit). It needs no npm packages, no browser and no python (S7 is optional).

## Desktop session: what to run

Replace `<REF>` with the commit or branch the orchestrator names (it must contain this directory).

```sh
cd /home/kyle-desktop/dino-trainer
source ~/.nvm/nvm.sh
git fetch
git status --short --untracked-files=no   # must print nothing; stop and report if it does
git checkout <REF>
node --version                # expect v26.9.0
node research/scripts/x86-determinism/probe.js --out /tmp/x86-probe.json > /dev/null
node research/scripts/x86-determinism/compare.js research/scripts/x86-determinism/mac-reference.json /tmp/x86-probe.json
cat /tmp/x86-probe.json
```

If the shared clone is in use by another session, run the same commands in a desktop worktree (for example `/home/kyle-desktop/dino-game-desktop-1`) after `git fetch && git checkout <REF>` there; the probe reads files only.

Optional, only if a python with numpy already exists on the desktop (do not install one for this): add `--python /path/to/python` to the probe command. Without it S7 is reported as skipped.

### What to send back
1. The probe's stderr summary lines (one per stage, plus the final "probe done" line with the runtime).
2. The full output of `compare.js` (about 25 lines).
3. The JSON: the contents of `/tmp/x86-probe.json` (one line, about 24 KB). If a message cannot carry it, send 1 and 2 first; the orchestrator can then ask for single stages with `node -e "const r=require('/tmp/x86-probe.json'); console.log(JSON.stringify(r.stages.S4))"` (S1, S3, S4, S5 are the large ones).

## Orchestrator (Mac): reading the result

Save the desktop JSON as, say, `/tmp/desktop-probe.json`, then:

```sh
node research/scripts/x86-determinism/compare.js research/scripts/x86-determinism/mac-reference.json /tmp/desktop-probe.json
```

- `CONFOUND` (node/V8 differ): an IDENTICAL stage still clears; a DIFFERS stage is unattributed. Rerun on the desktop with node v26.9.0.
- `REFUSED` (input files differ): the desktop ran another tree. Check `<REF>` and `git status` there.
- `S1` names the functions, buckets and input ranges that differ, and for the rules tables the score bands. To see the exact values, run `--dump` on both machines and `diff` the outputs:
  `node research/scripts/x86-determinism/probe.js --dump tanh:7 > tanh7.txt` (sets: every S1 function name, `rules-v2`, `rules-v3`, `adam-pow`; buckets 0..31 for functions, 0..100 for the rules tables, 0..19 for `adam-pow`).
- `S3` names the arm and the first 1,000-frame checkpoint that differs; `S4` the demo set, feature columns (names from `policy.js observe()`) and start levels; `S5` the model, arm and seed index.
- The job-class lines are the pre-registered reading (prereg §7). `ANOMALY` lines are results the premises exclude; follow them up before relying on the clearance.

## Stages (what each one exercises)

| Stage | Content | Mac runtime |
|---|---|---|
| S0 env | arch, node, V8, compiler variables, CPU, git HEAD and dirty flag, coreHash, sha256 of the input files | 0.0 s |
| S1 libm | 18 functions × 96,000 deterministic inputs (32 buckets) + edge values; the rules tables at every integer score 0..1,000,000 on rules v2 and v3 (the game's whole libm dependence); `pow(0.9, t)`, `pow(0.999, t)` for t ≤ 200,000; sqrt, fround and arithmetic as negative controls; a 1-ulp self-test | 0.3 s |
| S3 trajectories | 22 games (rules v1/v2/v3, jump and duck solvers, random keys; starts 0 to 998k; 3,000-point spans): per-frame state hashes (golden-format fields) and 1,000-frame checkpoints | 0.1 s |
| S4 demos | the three committed demo sets regenerated in memory: `v2-demos-jump-s1-40` (`runs/exp-bc-e2e`, Mac-only), `v2-demos-duck-s1-40`, `v3-demos-duck-s1-40`; sha256 vs the manifests, rows, labels, episodes, per-level and per-feature-column hashes | 6 s |
| S5 decisions | 25 held-out `assess` seeds to 20,000 points + 10 late games (150k, 998k) for `model.js` (v2 duck 2×8), `v2-bc-jump-d2w32-s1`, `v3-bc-duck-d2w32-s1`; JS net and wasm kernel; scores, action hashes, logit hashes, smallest top-2 margin | 8 s |
| S6 training | `test/bc.test.js` node pin recipe on the SMALL jump and stub-duck demos, a 2-epoch 32×32 arm, Box-Muller init, ES noise, 50 Adam steps | 0.1 s |
| S7 python | optional: `bc_train.py` on the pin recipe (reported, not cleared) | 0.5 s |
| S2 capture | last: Math.* wrapped while short games, policy play, a training epoch and ES noise run; which functions each path calls, how often, on which inputs | 0.0 s |
