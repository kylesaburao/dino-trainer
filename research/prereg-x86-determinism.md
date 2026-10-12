# Pre-registration: are x86 desktop results equivalent to the Mac's? (cross-architecture determinism probe)

Status: **registered, 2026-10-12**, before any desktop run. Written on main `2d360ca` (core hash `65f3546cea0b28ebab020a55e9772ddd37933e6d52c2a2ad08d2431df39998e9`, the rules-v3 core). The Mac reference was recorded after this file was written (§5 and §7 were refined, adding the mutation control, the per-class S1 functions and the anomaly lines, before the reference was recorded and before any desktop run); the desktop run has not happened.
Tools: `research/scripts/x86-determinism/` (`probe.js`, `compare.js`, `mac-reference.json`, `README.md`).

## 1. Decision
Which jobs may run on the x86 desktops (WSL2, Ryzen 7 9700X, node v26.9.0) with results treated as equivalent to the Mac's (darwin arm64, node v26.9.0, V8 14.6.202.34-node.32):

- **(a)** demo generation (`bc-gen`);
- **(b)** assessment, certification and scans (`assess`, `scan.js`, `survey.js`);
- **(c)** BC training on the node backend (`bc-train`, default backend);
- **(d)** ES training (`train`).

"Equivalent" has two grades, and the decision needs to know which one holds:
- **bitwise**: the same bytes as on the Mac (demos sha256, params, per-seed scores), so a desktop artifact can stand in for a Mac one, reproduce a committed manifest, and pass the repo's pins;
- **statistical**: the same algorithm on a numerically different but legal instance of the arithmetic, so results are valid replicates but not reproductions.

## 2. Evidence that prompted it
An x86-64 Linux cloud VM (node version unknown) regenerated BC demos with identical rows, label counts and episodes but a different sha256: rules-v3 duck s1-40 gave `6b1e98…` against the committed `b1bded86…` (`runs/exp-v3-clones/v3-demos-duck-s1-40/demos.json`), and rules-v2 duck s1-40 also differed from `363872f6…`. On the unmodified tree two `test/bc.test.js` tests failed there: the node-backend params pins (not gated to darwin/arm64, unlike the python pins). The v1 and v2 parity goldens passed there.

## 3. Premises, checked by reading the code (static, seconds)
- **P1. The game's only transcendental calls are `Math.log10` (rules v2/v3 `difficulty`, `superLevel`) and `Math.pow` (rules v3 `rulesAt`, `Math.pow(e, V3.curve)` with `curve` 1).** `grep Math\. core.js test/solver.js policy.js` lists otherwise only floor/ceil/round/min/max/imul (exact) and `Math.random` (not on the simulation path). `policy.observe()` has none. Both inputs are functions of the integer score only (`score() = floor(distance * scoreCoef)`), so the whole libm dependence of the game is a finite table: the rules record at every integer score 0..1,000,000. S1 hashes that table exhaustively.
- **P2. How a demo feature column can differ without a transcendental in `observe()`.** `difficulty` (log10) sets `rules.maxSpeed`, `pteroSpeedOffset`, `gapCoef`, `maxGapCoef` and the fairness tolerances. Once the run is at the speed cap (always, after a `startScore` start), `speed = rules.maxSpeed` every frame, so a last-bit change in `log10` is a last-bit change in `speed`, which feeds `observe()` features 4 (speed) and `o*.vel` (speed + speedOffset) directly, and, through `distance`, obstacle `x` and `jumpVelocity(speed)`, also `o*.dx` and the dino height/vy. The rows are Float32, so most float64 last-bit differences round away; a few land on a float32 rounding boundary and change one stored bit. That gives a different sha256 with the same rows, labels and episodes, as on the VM. A difference that flips a comparison (collision, fairness, the boring-frame rule, the solver's choice) would change rows or labels; the VM saw none.
- **P3. The goldens bound where such a difference can be.** The v2 goldens compare `speed`, `distance`, dino y/vy and every obstacle's x/y/size/w bit for bit at 0–1.6k, 30k, 148k–151k and 998k–1M, and they passed on the VM. So if the VM differed through `log10`, it was at scores the goldens do not cover (the demos' 2k–7k, 50k–55k, 300k–305k, 900k–905k windows and the v3 bands). S1's per-10k-score bucket hashes and S4's per-level hashes localise it.
- **P4. The node-backend pin failure cannot come from the game.** The pinned SMALL demos stop at 300 points, where `difficulty` is 0 and no transcendental runs. bc-gen has no MLP. So that failure comes from training: `Math.tanh` (forward), `Math.exp`/`Math.log` (softmax CE), `Math.cos` (cosine LR), `Math.pow` (Adam bias correction), `Math.sqrt` (Adam; correctly rounded), and the init's Box-Muller (`trainer/es.js` `gaussian`: `Math.log`, `Math.sin`, `Math.cos`, `Math.sqrt`), unless the VM's V8 simply differed (H0).
- **P5. Plain float64 arithmetic is not a suspect.** JS `+ - * /` are IEEE correctly rounded and V8 does not contract them into FMAs (the spec forbids it); `Math.sqrt` and `Math.fround` are correctly rounded. The wasm kernel uses separate f64x2 mul and add (no relaxed SIMD). These are the negative controls.
- **P6. Mechanism suspected.** V8 ships its own libm (`base::ieee754`, an fdlibm port) compiled per platform. The Mac's node is built with clang 17 for arm64 (`process.config.variables`), where clang's default `-ffp-contract=on` may fuse multiply-adds inside the polynomial evaluations; official Linux x64 builds use GCC without FMA. Either can change last bits. S0 records the compiler variables.
- **Task-implied additions.** `Math.sin`/`Math.cos` were not in the task's list but are on the paths of (c) and (d) (Box-Muller noise and init; cosine LR), so S1 includes them. `Math.log2` (`bc.js` pow2 quantization) is included too.

## 4. Hypotheses and what each predicts in the probe
| | H0: bit-identical | H1: libm last bits only | H2: differences reach the game or decisions |
|---|---|---|---|
| S1 libm tables | all identical | some of tanh/exp/log/log10/pow/sin/cos/… differ; sqrt, fround and arithmetic identical | as H1 |
| S1 rules tables (log10 domain) | identical | identical, or differ only in `bits` | differ |
| S3 trajectories | identical | `discrete` identical; `bits` may differ if the rules table differs | `discrete` differs (crash frame, obstacle sequence, rng calls) |
| S4 demos | sha256 = the committed manifests | rows, labels, episodes, label/weight columns identical; some feature columns and the sha256 differ | rows, labels or episodes differ |
| S5 decisions | identical | scores and action hashes identical; logit hashes and margins may differ | a score or an action hash differs |
| S6 BC node / ES primitives | identical, pins match | params hash and pins differ; init hash differs iff log/sin/cos differ | (same as H1: training has no discrete outcome) |

H0 also covers "the VM had another V8". If the desktop's node or V8 differs from the Mac's, the probe records it and compare.js reports the confound: an IDENTICAL result is still a valid clearance for those versions, but a DIFFERS result cannot be attributed to the architecture.

## 5. Controls
- **Negative controls (must be IDENTICAL on any IEEE machine):** S1 `sqrt`, `fround`, `arith` (separately rounded `a*b+c`, `a/b`). If one differs, the probe or the harness is broken; nothing else is read.
- **Positive control (sensitivity):** S1 `selfTest` flips the lowest bit of one output and checks that exactly that bucket's hash and the full hash change. `compare-selftest.js` runs compare.js on 11 mutated copies of the reference (one known change each: an S1 bucket, a rules-table bucket, a demo feature column, a demo label column, a score, a trajectory checkpoint, a node pin, the node version, the sqrt control, a file hash) and checks that each verdict names exactly that item and reads it as pre-registered.
- **Run-to-run control:** the probe run twice on the Mac must compare IDENTICAL (all hashed content excludes timings).
- **Known anchors:** S4 reports whether each demo set matches its committed manifest sha256 (`3a67bfd4…`, `363872f6…`, `b1bded86…`); S6 reports whether the SMALL demos (`cf4962aa…`, stub duck `a468d45a…`) and the node pins (`27e19582…`, `5674e0f0…`) match `test/bc.test.js`.

## 6. Horizons and their limits
Each stage is sized to run in about a minute in total on the desktop (about 2.5× slower than the Mac):
- S1: 96,000 sampled inputs per function, plus the exhaustive rules tables (2 × 1,000,001 scores) and Adam's bias-correction powers (t = 1..200,000).
- S3: 2 seeds × 11 solver/random arms, 3,000-point spans, starts 0 to 998k.
- S4: the three full demo sets (7.3M rows in total), single thread.
- S5: the 25 `assess` held-out seeds (offset 0) to a **20,000-point cap** for three models, with the JS net and the wasm kernel, plus a late arm (5 seeds, 3,000 points from 150k and from 998k), so rules-v3 speeds and `Math.pow` are exercised by a policy. A full 1M game is 1.9M frames; 25 of them per model would take minutes per model, beyond the budget.
- S6: the `test/bc.test.js` pin recipe (1 epoch, hidden [4]) on the SMALL demos, a 2-epoch 32×32 arm, ES noise for 8 seeds and 50 Adam steps.
A trajectory difference that needs more than these horizons to appear (a rare comparison flip in a long game) is **not excluded** by an IDENTICAL S3/S5; under H1 it is possible in principle. That is why the H1 readings below keep certification comparisons on one architecture.

## 7. How each outcome will be read (job classes)
compare.js applies these rules mechanically:

| Result | (a) bc-gen | (b) assess / scans | (c) BC node training | (d) ES training |
|---|---|---|---|---|
| **H0** (all identical) | cleared, bitwise | cleared, bitwise | cleared, bitwise (pins pass) | cleared, bitwise (a Mac checkpoint may resume on x86) |
| **H1** (only `bits` differ) | conditional: valid demos, not byte-identical; never reported under a Mac manifest's sha256; record the machine | conditional: per-seed scores matched within the probed horizon; every arm of one paired comparison (McNemar etc.) runs on one architecture; final certification of a ship candidate stays on the Mac | conditional: valid replicates, not reproductions; the repo's node pins fail there; record the machine | conditional: valid independent runs; no cross-architecture resume, and `bit-identity.js` compares two trees on one machine only |
| **H2** (`discrete` differs in S3, S4 or S5) | not cleared (if S4 is the stage that differs) | not cleared (if S3 or S5 differs) | conditional as H1; if S4 differs, train only on Mac-made demos | train-only: candidates are assessed and certified on the Mac |

A class is "cleared, bitwise" only if its probed stage is IDENTICAL **and** S1 is identical for the functions and tables that job runs: (a) the rules tables; (b) the rules tables and tanh; (c) tanh, exp, log, sin, cos, sqrt, pow and the Adam powers; (d) all of (b) and (c) except exp. A difference there with identical job outputs reads as the H1 row (the job's arithmetic differs on some inputs that the probed outputs happened not to hit).

compare.js also prints an **ANOMALY** line for results the premises exclude: S3 or S4 differing while the exhaustive rules tables are identical (contradicts P1), S5 differing while the rules tables, tanh and S3 are identical (points at the JIT dot product), or the candidate's JS net and wasm kernel disagreeing. An anomaly is a follow-up item, not a clearance.

An env confound (node or V8 version differs) downgrades every "cleared" for a stage that DIFFERS to "unattributed, rerun with the Mac's node"; IDENTICAL stages still clear. A `coreHash` or file-hash mismatch makes compare.js refuse the stages that depend on the mismatched files.

## 8. Not covered
- **python/numpy on x86** (`bc-train --backend python`, `bc-train-batch`). OpenBLAS (x86) and Accelerate (Mac) are expected to differ in summation order, so params are expected to differ; the repo's python pins are already gated to darwin arm64. Optional stage S7 runs only with `--python <interpreter with numpy>` and is reported, not cleared.
- **MLX** (Apple GPU only), the browser and the CSS evaluator (no browser on the desktop).
- **Worker-thread scheduling**: all trainer pools are order-independent by construction (`genSeedsParallel` emits in seed order, `scoreSeeds` per seed); the probe runs single-threaded.
- Different node versions on the two machines (reported, see §7).

## 9. What would reopen it
A node upgrade on either machine (rerun the probe on both), a `core.js`/`policy.js` change that adds a transcendental call, or any job whose desktop and Mac results disagree outside what §7 allows.
