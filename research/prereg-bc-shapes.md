# Pre-registration: what is the smallest clone that is as reliable as the shipped 44-32-32-3?

Status: **registered.** Author: mac-3, 2026-10-10, on main `0e334ff`.

The user's goal (chosen 2026-10-10): **a smaller, faster model.**

## 1. Decision
Should `model.js`/`model.css` be replaced by a smaller clone? If so, which shape?

- Shipping still needs the user's explicit approval, because it replaces the shipped model. This study produces the candidate and the evidence.
- A smaller model is worth shipping only if two things hold:
  - (a) it is as reliable on fresh held-out seeds;
  - (b) it is materially smaller or faster where it runs: the JS AI mode, and above all the CSS evaluator page.

## 2. Hypotheses
- **H-small:** a 44-8-8-3 (or smaller) clone is as reliable as 44-32-32-3.
  - Probe B supports it: 8-8 nets reached 1M on all their full games (`research/feature-sufficiency.md` lines 21, 121 and 263).
  - The evidence is thin: tens of games, and the prototype's data pipeline.
- **H-capacity:** below some width or depth, rare failures appear. They would show as scan failures, and as a smaller top-2 logit margin, which also raises the risk that the CSS evaluator diverges from JS.
- **H-no-gain:** a smaller model gives no material speed-up on the CSS page, because the frame time there is dominated by something else (layout, rendering). In that case, "faster" is not a reason to ship, and only the size gain (bytes) remains.

## 3. Premise checks (Step 0, run first, seconds to minutes)
0. **Shape-generic code paths.** Everything validated so far is 44-32-32-3. Train tiny models on small demos with `--hidden 8` and `--hidden 8,8,8`, on both backends. Export each with `export --out <scratch>/model.js` (never use the default, which writes the root `model.js`; see `cli.js:647`). Generate CSS from each with `trainer/css-model.js` to a scratch path, and check that `css-nn.js` actually evaluates it to the same decisions as `policy.js`.
   - If any path assumes two hidden layers, **report it and exclude those shapes as CSS-page candidates. Don't patch** `css-model.js`, `css-nn.js` or `trainer/` inside this experiment.
1. **Sizes:** the shipped `model.js` is 53,082 B, and `model.css` is 104,562 B.
   - Parameter counts: 44-32-32-3 = 2,595; 44-16-16-3 = 1,043; 44-8-8-3 = 459; 44-16-3 = 771; 44-8-3 = 387; 44-4-4-3 = 215.
   - Generate `model.css` for each shape with `trainer/css-model.js` into scratch paths. **Never** write the root `model.js`/`model.css`. Record the bytes.
2. **CSS page speed:** measure the **forced style-recalc time per decision**, `performance.now()` around the `--x0..--x43` writes plus the `getComputedStyle` read of `--act`. Measure it over ≥ 1,000 real played frames for 32-32 and 8-8, in headless Chrome. fps is only a floor check: under rAF it reads 60 whenever recalc fits the frame budget.
   - **H-no-gain for speed holds if 8-8's median per-decision time is not at least 1.5× lower than 32-32's.**
   - `test/browser-check.js` already reports the CSS page's `fps`.
   - Use a scratch copy of `index-css.html` pointed at a scratch `model.css`, or an equivalent harness. Don't modify the root files.
 3. **JS speed:** `node trainer/cli.js bench` policyFps per shape. Bench takes `--hidden`; check how it builds the net.

## 4. Design
- **Demos:** the validated `bc-gen` output in `runs/exp-bc-e2e/` (seeds 1–40, default settings). No regeneration.
- **Backend:** Python (about 1 s per epoch at 32-32), with `PYTHON=/Users/kyle/Desktop/dino-game/.venv/bin/python`. Use the same hyperparameters as the shipped clone: 15 epochs, batch 1024, lr 2e-3.
- **Shapes:** 44-16-16-3, 44-8-8-3, 44-16-3, 44-8-3 and 44-4-4-3, each with training seeds 1 and 2. That is 10 models. The shipped 44-32-32-3 is the reference.
- **Rung 1 (screen), per model:**
  - `bc-train` validation (25/25 required);
  - a scan of 1,000 full v2 games to 1M on **seeds 30,003,000–30,003,999** with `research/scripts/prereg-bc-robustness/scan.js --wasm`;
  - the min top-2 logit margin over **600k frames of the model's own play**, the same measure as the decision log's figures (shipped clone 2.7e-4, ES 5.8e-5). Use a separate margin script; `scan.js` stays unchanged.
- **Positive control:** the shipped clone on the same 1,000 seeds. It must be 0/1,000; it was 0/2,000 elsewhere. If it fails any game, STOP: the harness or the seed range is suspect.
- **Promotion:** a shape passes rung 1 if **both** training seeds have 0/1,000 failures and validation 25/25. The candidate is the passing shape with the fewest parameters. Break ties by the larger min margin.
- **Gate between rung 1 and rung 2:** certify only if Step 0 shows a speed gain (H-no-gain rejected) **or** the candidate has ≤ half the shipped model's `model.css` bytes. Otherwise stop after rung 1 and report: held-out offsets are not spent on a marginal win.
- **Rung 2 (certification), the candidate only:** held-out **offsets 325–474** (150 seeds) with `eval --seed-offset 325 --seeds 150 --target 1000000`. Use the candidate's training seed 1, unless only seed 2 passed. Also run the shipped clone on the same offsets, paired.
  - **After this, the next fresh offset is 475.** Record that in the handoff.
- **CSS faithfulness for the candidate:** compute the min top-2 logit margin. Then run a CSS vs JS agreement check on ≥ 1,000 frames in headless Chrome, in a scratch harness, never on the root files.
- **Cost:**
  - training: 10 × about 15 s, plus validation;
  - scans: 11 × about 3 min, about 35 min in total, with smaller nets slightly faster;
  - certification: 2 × about 45 s.

## 5. Readings (fixed)
| Result | Reading |
|---|---|
| Shipped control fails any rung-1 game | STOP; diagnose the harness or seeds and read nothing |
| A shape passes rung 1, the candidate certifies 150/150, and CSS agrees 0-disagree on ≥ 1,000 frames | **The candidate is a ship-ready alternative.** Report its size and speed gain to the user, who decides. If Step 0 shows H-no-gain for speed, say the gain is file size only |
| The candidate certifies < 150/150 | The rung-1 screen was optimistic. Don't ship. Try the next-larger passing shape only after reporting |
| No shape below 32-32 passes rung 1 | **H-capacity at the shipped hyperparameters** (15 epochs, lr 2e-3): the shipped shape stays. Record the failing shapes and their failure scores in the register. Reopen with a longer schedule as a separately registered step, since a small net may be underfit rather than too small |
| A shape passes with one training seed but not the other | Seed-dependent capacity at that size. Not a candidate; report it |
| The candidate's min margin is < 5.8e-5 (the old ES model's, which shipped without CSS divergence) | Flag it as a CSS-divergence risk, even if the agreement check passes |

**Power and scope.** 0/1,000 bounds a shape's rate below 0.3% (one-sided 95%), and 0/2,000 over both seeds below 0.15%. 150/150 matches the shipped model's certification. It can't prove "as reliable" beyond about 2% per game. The shipped model's stronger bound (0/2,000 at 0.15%) comes from the scans, which apply to the candidate as well. Scope: v2, the 44 features, the default demos and the Python backend.
