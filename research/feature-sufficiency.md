# Are the 44 input features sufficient? Two read-only probes

Compiled 2026-10-10 on branch wt/mac. No project code was changed. All scripts are in the session scratchpad
(`$S = /private/tmp/claude-501/-Users-kyle-Desktop-dino-game/a1a47f55-01e9-48b9-afbe-71ea588921bc/scratchpad/features/`).
They `require()` the repo modules from `/Users/kyle/Desktop/dino-game` by absolute path. A trainer job
(`runs/exp-hyg-s3-full`) was running for most of this session, so every step used at most 4 processes.

Labels: **[M]** measured in this session; **[D]** derived from the code or by arithmetic; **[I]** inference or opinion.

## 1. Question and verdict

The question: "the input features as is may not be sufficient". Do the 44 features in `policy.js observe()`
hold enough information, scaled well enough, for a policy to survive v2 to 1M? Two sub-questions:
- Do the shipped model's remaining crashes come from missing or badly scaled features?
- Could any policy built on these features play near-perfectly?

**Verdict: the features are sufficient. Neither probe found a case for changing `observe()`.**

1. **[M] Probe B is the decisive test.** A 44-32-32-3 network, behaviour-cloned from the exact solver on the
   unchanged 44 features, reached 1,000,000 on **25 of 25** fresh full-game seeds. Both training seeds did, and so
   did every variant with extra features. Even a **44-8-8-3** network, and a 44-32-32-3 trained on only 50k
   frames, survived every one of 1,000 test windows from 0 to 900k points, and both reached 1M on 25/25 seeds.
   The features can represent a policy at least as good as the shipped one. Engineered features had nothing to
   improve: every variant hit the ceiling.
2. **[M] Probe A: each of the shipped model's crash modes is visible in the features in time to act.** In
   every crash the obstacle that killed the dino was on screen (x < 600), and a rescue still existed:
   - early crashes: visible ≥ 54 frames before death, rescuable until 1–15 frames before;
   - late crashes: visible 22 frames before, rescuable by fast-drop until 7 frames before.

   On the raw 44 features, crash and pass situations separate well, with cross-validated AUC 0.85–0.93 at
   the decision frames and 1.0 late.
3. **[M] One crash mode involves state that `observe()` does not expose: whether the jump key is still
   held.** This mode is 26 of the 53 early crashes, about 1.3% of starts from 0. But the policy causes it
   with its own odd action sequence (DOWN then JUMP mid-air), and the policy could avoid it on the current
   features.
4. **[M] Late crashes do coincide with the most out-of-range inputs, but scaling is not the cause.**
   - At those crashes the speed feature reads 2.28 and the obstacle-velocity feature 1.81.
   - The real cause is physical. **[M]** The dino lands about 3 frames after the pterodactyl arrives,
     while the fairness model's canonical plan lands at or before it. **[I]** The model's takeoff time and
     key hold both differ from canonical, and the model never fast-drops.
   - Hidden-unit saturation is flat across levels (51–54%), and the clone trained with /22 rescaling did
     no better.

**Recommendation: keep the features (section 6).** The crashes are behaviour and search problems, not
representation problems. Changing `observe()` costs a full retrain (section 7) and the probes show no benefit.

## 2. Methods

Shared facts [D]:
- `tau` = frames until an obstacle's hitbox reaches the dino's front: `(obstacleBox(o).x − (DINO_X+6+30)) / (speed + speedOffset)`.
- The shipped model is `model.js` (44-32-32-3, `actionRepeat` 1, v2), driven through `policy.createAgent`.
- No seed used here overlaps the held-out set: held-out seeds are ≥ 2^31, and every seed here is below 10^6.

### Probe A: crash separability of the shipped model
- **`$S/probeA.js`** plays the shipped model and logs every obstacle encounter.
  - `early` mode: seeds 5000–6999 from 0, capped at 2,000 points.
  - `late` mode: startScore 950k, 960k, 970k, 980k and 990k, each in 10k-point windows.
    - Batch 1: seeds 5000–7399, 12,000 windows.
    - Batch 2 (high pterodactyls only): seeds 8000–9199, 6,000 windows.
  - For each encounter it records the 44 features and the action at the first frame with tau < 40, 30, 20,
    12, 6 and 2.
  - It also records full-state extras the features lack:
    - `o.gap` and `requirement()` from/to/release;
    - `dino.released`, `speedDrop`, `reachedMin`, `keys.jump`;
    - the next obstacle, including off-screen ones, with its gap and visibility.
  - Each crash also gets a 45-frame ring buffer of features, action, y, vy and key state.
  - Passes are sampled: 5% of early cacti; every early pterodactyl; and in batch 2 every high (y-class 2) pterodactyl.
- **`$S/rescue.js`**: counterfactual replays.
  - It replays the model's exact actions up to frame f before each crash, then tries rescue macros from f:
    - hand over to `test/solver.js`;
    - hold DOWN for n ∈ {1..20} frames, then the solver;
    - NONE for n frames, then the solver.
  - It reports the latest f from which some macro survives the crash obstacle plus 150 frames, plus the takeoff
    frame and when the crash obstacle first became visible.
- **`$S/analyzeA.py`**: crash vs pass separability at each tau snapshot (numpy):
  - L2 logistic regression with 5-fold stratified CV AUC;
  - leave-one-out 1-NN on standardised features.

  Each runs on three feature sets:
  - the raw 44;
  - 44 plus engineered features from allowed information (tau of each obstacle, the gap between obstacles 0
    and 1 in frames, speed/22);
  - 44 plus hidden or forbidden extras (`keys.jump`, `released`, `speedDrop`, `reachedMin`, `requirement.from`).
- **`$S/keyheld.js`** and **`$S/actfreq.js`**: per-level statistics for the shipped model:
  - how often it lands with the jump key held;
  - the fraction of hidden pre-activations with |pre| > 2;
  - the argmax margin;
  - action frequencies and jump-key hold lengths.

### Probe B: global sufficiency by behaviour cloning the exact solver
- **`$S/featx.js`** extends `policy.observe` with 10 extra columns. Each is a function of the 44 features, except
  `keyJump`, which is the agent's own key state:
  - `tau0..2`: frames until each obstacle reaches the dino, /30;
  - `gap01`: frames between obstacle 0's back edge and obstacle 1's front, /30;
  - `keyJump`;
  - `tLand`: frames until a free fall lands, /30;
  - speed rescaled `(speed−6)/16`, and obstacle velocities /22.
- `expert(game, k)` reproduces `test/solver.js` per state. It agreed with the solver on **100% of 2.1M
  demonstration frames [M]**.
- **`$S/cloneB.js gen`** collects solver demonstrations: seeds 1–40 × start levels {0 (3k points),
  2k, 50k, 300k, 900k (5k-point windows)}.
  - That is 2.10M frames. Boring frames (grounded, label none, nearest obstacle more than 40 frames away)
    are kept 1-in-10 with weight 10.
  - **[M]** Solver crashes: 0. Jump presses: 47,931, and the obstacle the solver took off for was on screen
    (x < 600) at takeoff in **every** case. So the solver's choice of *when to jump* never needed off-screen information.
  - The solver never presses DOWN, so class 2 has no samples.
- **`$S/trainB.py`** trains in numpy:
  - a tanh MLP with a linear output and softmax cross-entropy;
  - sqrt-balanced class weights, Adam with a cosine-decayed learning rate, batch 1024;
  - raw, unstandardised inputs, exactly as `policy.js` would feed them;
  - export in `createNet`'s flat layout.

  **[M]** A 44-feature clone run through `policy.createAgent` matched `cloneB.js`'s own loop on all 19,999
  frames checked (`$S/parity.js`).
- **`$S/cloneB.js roll`** rolls out each clone in `core.js` with the same idle→START and argmax loop as
  `createAgent`:
  - 200 fresh seeds (100000–100199) per level: 0→3,000 points, plus 20k-point windows from 2k, 50k, 300k
    and 900k;
  - 25 fresh seeds (200000–200024) from 0 to 1,000,000.
- Drivers: `$S/trainall.sh` and `$S/rollall.sh` run 7 variants × 2 training seeds at 32-32. `$S/hard.sh`
  covers the harder regimes: 8-8 hidden on all data, and 32-32 on a random 50k frames.
- DAgger was planned to correct for covariate shift. It was not needed: plain behaviour cloning already
  reached the ceiling (section 5).

## 3. Probe A results: what the shipped model's crashes are

### 3.1 Early zone, from 0 (2,000 starts, cap 2,000 points)
**[M] 53 of 2,000 starts crash (2.65%)**, at scores 461–688.
- Every one dies on a **single small cactus** with a pterodactyl visible about 250–305 px behind it.
- Speeds are 8.48–9.49, so these are the first pterodactyls.

The crashes split into two mechanisms of almost equal size:

| Mode | Count | What happens | Crash obstacle visible | Latest rescue |
|---|---|---|---|---|
| A: premature full jump | 27 | The model takes off ~31 frames before contact (canonical: ~12). In 24 of 27 it is the *exact frame* the low (y-class 0) pterodactyl first appears at x < 600. It holds a full-height jump and lands on the cactus. | ≥ 54 frames before | 1–15 frames before (DOWN 1–10 or solver) |
| B: stuck jump key | 26 | Mid-air the model outputs DOWN (fast-drop, which releases the jump key), then JUMP again. The second press does nothing in the air but leaves the key held. It lands with `keys.jump = true`, keeps outputting JUMP, which needs a fresh key-down edge to fire, and runs into the cactus on the ground. | ≥ 54 frames before | 4 frames before (solver: release, then jump) |

**[M] Separability**, on small cacti met at 400–750 points with a low pterodactyl behind them: 53 crashes vs 43 passes.

| Snapshot | Logistic AUC, raw 44 | + engineered | + hidden or forbidden | 1-NN AUC, raw 44 | Model's JUMP rate, crash / pass |
|---|---|---|---|---|---|
| tau < 40 | 0.858 | 0.904 | 0.899 | 0.866 | 0.51 / 0.19 |
| tau < 30 | 0.914 | 0.919 | 0.919 | 0.896 | 0.26 / 0.05 |
| tau < 20 | 1.000 | 1.000 | 1.000 | 1.000 | 0.96 / 0.21 |

The wider set, any pterodactyl follower (53 vs 137), gives 0.853 / 0.928 / 1.000 on the raw 44. Adding the hidden
state changes AUC by ≤ 0.04.

**[I] Reading:** at the decision frames the raw features already separate the crash situations from the passes.
The model acts on them consistently, but wrongly: it jumps on the pterodactyl's appearance. That is a policy
error, not missing information.

**[M] Mode B is the one observation-aliasing case found.**
- While grounded with the key held, the 44 features are identical to grounded with the key up, but the action
  JUMP has different effects in the two states.
- `keys.jump` is the agent's own input, so it is allowed information.
- Across 25,801 landings in 200 other early runs (`keyheld.js`), the model landed with the key held 0 times.
  So the alias only arises in this failure, after the rare mid-air DOWN→JUMP sequence.
- The model uses DOWN in 0.16% of frames at level 0 and in 0 frames at 50k, 300k and 970k (`actfreq.js`, 40 windows each).

### 3.2 Late zone (950k–1M, 10k-point windows)
**[M] 64 crashes in 18,000 windows. By start level:**

| Start level | Crashes |
|---|---|
| 950k | 0 |
| 960k | 13 |
| 970k | 14 |
| 980k | 14 |
| 990k | 23 |

- Crash scores range from 962,854 to 998,494. The rate is about 0.36% per 10k window, or roughly 1.8% of games
  dying in the last 50k points.

**[M] All 64 crashes are the same event:**
- A **high (y-class 2) pterodactyl on the fast side** (v = speed + 1.6 ≈ 23.55–23.60 px/frame) arrives while
  the dino is **landing from a jump that began 29–30 frames earlier**. The dino is at y ≈ 70.8 and falling
  (vy 9.4) into the pterodactyl's hitbox (y 71–79).
- The jump was over a small cactus. The model held the jump key about 15 frames.
  - **[M]** Typical hold at 970k is 17–21 frames; the solver's canonical release is at frame 6.
  - **[D]** `requirement()` gives release 6 and airtime 32 for one cactus at every speed, and `compatible()`
    lets the fairness model place the "stay grounded" pterodactyl right behind that canonical airborne interval
    (`p.to <= q.from`).
  - **[M]** The model's dino lands about 3 frames after the pterodactyl arrives (the margin below).
  - **[I]** That overshoot comes from the model's takeoff and its hold, both of which differ from canonical.
    I did not separate how much each contributes.
- At takeoff the pterodactyl was **not yet on screen** (x ≥ 600). It appears 22 frames before contact, about 8
  frames after takeoff.
- **[M] Rescue:** from the moment it appears, a fast-drop (DOWN for 8 frames) saves the run on every one of the 39 batch-1
  crashes tested until **7 frames before contact**. That leaves a 15-frame window, and the model never presses DOWN.
- **[M] Physical margin** at the tau < 20 snapshot is the pterodactyl's arrival minus the free-fall landing time:
  - all 25 crashes in batch 2: −3.05 to −3.00 frames;
  - 3,000 sampled airborne passes: min −2.05, median +10.4.
- The crash and pass sets are disjoint on this one quantity, and on the raw 44 features as well (logistic and
  1-NN AUC 1.000 at tau < 20).
- **[D] Why the zone starts at about 962.6k.** At the speed cap, v_fast-ptero = 13.8 + 9.8·d, where
  d = log10(S/1000)/3. The crash geometry needs v ≥ 23.546, which means d ≥ 0.9945, i.e. S ≥ 962.6k. The first
  observed crash was at 962,854, and no 950k window crashed in 3,600.

**Did the late crashes coincide with out-of-range speed features?**
- **[M] Yes, but not causally.**
  - At every late crash out[4] = 2.28 (speed 21.95–22.0, normalised by v1's 13) and the obstacle-velocity
    feature is 1.81. These are the largest values the game ever produces.
  - But the 3,000 airborne passes in the same zone read the same out[4], and their pterodactyl velocities reach
    23.60, so the range itself doesn't discriminate. The −3-frame landing margin does.
- **[M] Conditioning check (`keyheld.js`).**
  - The fraction of hidden pre-activations with |pre| > 2 is 0.509, 0.525, 0.528, 0.533 and 0.541 at levels 0,
    2k, 50k, 300k and 970k.
  - The 5th percentile of the argmax margin is 1.89, 1.72, 1.34, 3.23 and 3.57 at the same levels.
  - Nothing degrades at the top of the input range. The least confident zone is 50k, not the late game.
- **[I] What is genuinely ill-conditioned is the gap between canonical and actual play.**
  - The fairness guarantee holds for the solver's minimal jump.
  - A policy whose habitual jump is longer than the minimum can be ambushed by an obstacle that is not yet
    visible at takeoff. That is a property of the 600 px screen the anti-cheat rules impose, so no feature can
    remove it.
  - The available remedies are behavioural: shorter jumps, or a fast-drop once the obstacle appears. Both need
    only information the current features already carry.

## 4. Named missing or ill-conditioned quantities

| Quantity | In `observe()`? | Evidence | Status |
|---|---|---|---|
| Jump key currently held (`keys.jump`, or the previous action) | No | Mode B: 26/53 early crashes, aliased state | **[M] the only genuinely missing state found.** Allowed (own input). Avoidable behaviourally. |
| Time-to-reach (tau) per obstacle | Implicitly (relx and v), as a ratio | Engineered tau raised early AUC by ≤ 0.05; clones did not need it | **[M]** not needed for sufficiency |
| Gap to the second obstacle in frames | Implicitly (two relx, w and v) | Clone variants with it: no rollout difference; 50k-frame fit loss 0.018 vs 0.024 | **[M]** marginally easier to fit; no rollout effect |
| Free-fall landing time vs pterodactyl arrival | Implicitly (y, vy, relx, v) | Perfectly separates the late crashes (−3 vs ≥ −2.05 frames) | **[M]** the key late quantity, but the raw 44 already separate it (AUC 1.0) |
| Speed and velocity normalised by 13 (out[4] up to 2.28, velocity up to 1.81) | Yes, out of [0, 1] | No saturation or margin anomaly late; the /22 clone was no different | **[M]** cosmetic |
| Pterodactyl height class | Yes (one-hot and box y) | All late crashes are class 2, correctly encoded | Present |
| Obstacles at x ≥ 600 | No (anti-cheat) | Late pterodactyl invisible at takeoff; the solver never needed one at takeoff | Structural limit; must not be added |

## 5. Probe B results: clone rollouts with and without candidate features

**[M]** Survival is the fraction of 200 fresh seeds reaching the end of each window. "Full" counts fresh seeds
reaching 1,000,000 from 0. Each row covers two training seeds (s1, s2) unless noted. The training columns are
2.1M demonstration frames and 15 epochs unless noted.

| Variant (input columns) | Inputs | Hidden | 0→3k | 2k +20k | 50k +20k | 300k +20k | 900k +20k | Full 0→1M |
|---|---|---|---|---|---|---|---|---|
| `base`: the current 44 | 44 | 32,32 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **25/25, 25/25** |
| `rescale`: out[4] and velocities /22 | 44 | 32,32 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 25/25, 25/25 |
| `tau`: + tau0, tau1 | 46 | 32,32 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 25/25, 25/25 |
| `tau_gap`: + gap01 | 47 | 32,32 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 25/25, 25/25 |
| `key`: + keyJump | 45 | 32,32 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 25/25, 25/25 |
| `tau_gap_key` | 48 | 32,32 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 25/25, 25/25 |
| `all`: rescaled + tau0..2, gap, key, tLand | 50 | 32,32 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 25/25, 25/25 |
| `base`, small net | 44 | **8,8** | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 25/25 (s1) |
| `rescale`, `tau_gap`, `all`, small net | 44–50 | 8,8 | all 1.000 | all 1.000 | all 1.000 | all 1.000 | all 1.000 | not run |
| `base`, **50k frames**, 40 epochs | 44 | 32,32 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 25/25 (s1) |
| `rescale`, `tau_gap`, `all`, 50k frames | 44–50 | 32,32 | all 1.000 | all 1.000 | all 1.000 | all 1.000 | all 1.000 | not run |

Fit, as final training loss, frame recall none / jump:
- Full data, 32-32: 0.0043–0.0063 for every variant; recall ≥ 0.997 / ≥ 0.997.
- 50k frames, 32-32:
  - `base`: 0.024
  - `rescale`: 0.025
  - `tau_gap`: 0.018
  - `all`: 0.019
- 8-8: 0.0058–0.0076, with no ordering by variant beyond noise.

Reading:
- **[M]** There were 352 full games to 1M and 30,000 windows (200 seeds × 5 levels × 30 models), with **zero crashes** in any configuration,
  including the unmodified 44 features on a 44-8-8-3 net. By the rule of three, the per-game failure rate of
  the clones is below about 1% (3/352) at 95% confidence, pooled over variants.
- **[M]** No candidate feature changed rollout survival, because there was no failure left to fix. The only
  measurable effect is the slightly lower fit loss with tau and gap at 50k frames: easier imitation, no
  behavioural consequence.
- **[I] Caveat:** this shows that the features *can represent* a near-perfect policy and that supervised
  learning finds it easily.
  - It does not show that ES finds it as easily. Engineered tau or gap inputs might shorten the ES search,
    which this probe doesn't test.
  - The existing literature note (`observation-design.md`, item 2) found no source that tests tau against
    distance, so that remains unmeasured.
- **[I]** The solver plays *canonical* minimal jumps and never ducks. That is exactly the behaviour that avoids
  the shipped model's late failure: landing about 3 frames late. A 44-feature net can carry that behaviour.

## 6. Recommendation

**Keep the 44 features.** [I], based on the measurements above:
1. **No `observe()` change is justified by these probes.** Every crash of the shipped model was visible and
   rescuable within the current features. A 44-feature net of the shipped size, or much smaller, can play to
   1M. Rescaling by 22 is cosmetic.
2. **If `observe()` is ever reopened for another reason, bundle these into that one change.**
   - `keyJump` (jump key held) is the single piece of genuinely missing state, and it removes the Mode-B
     alias (about half the early crashes, ~1.3% of starts).
   - tau or gap features are optional; their only measured effect is easier fitting.
   - An alternative to `keyJump` that leaves the features alone is to change the agent's action interface so
     that JUMP while the key is held issues a release first. That is also an edit to `policy.js`, at the same
     cost below.
3. **Target behaviour instead**, which needs no feature change:
   - Early Mode A and B (the first pterodactyls at 460–690 points): more start-0 coverage. The curriculum run
     already reached 0/240 crashes from 0, per HANDOFF 4.3.
   - Late (≥ 962.6k): the policy needs shorter jumps or a fast-drop reflex for fast high pterodactyls.
     - Evaluation windows at 960k–1M expose this failure: about 0.36% per 10k window.
     - Training episodes that end before 962.6k never meet it.
   - **[I] Behaviour-cloning the solver into the 44-feature net is a strong warm start or reference for ES.**
     It reached 25/25 at 1M after about 20 s of training. Whether a policy *trained from* the privileged
     solver is acceptable under the project's anti-cheat spirit is the user's call. At runtime it reads only
     `observe()`, but its labels came from information the policy may not use.

## 7. Cost of any `observe()` change

[D] from the code, unless marked:
- **Core hash:** `trainer/store.js coreHash()` hashes `core.js` and `policy.js`. Any edit to `observe()`, or
  `OBS_DIM`, invalidates every checkpoint, so `--resume` needs `--force`. And `--force` is only valid when
  decisions are provably unchanged, which a new feature never satisfies. Every run, including the shipped
  `runs/v2-h32x32-1m`, would need fresh training.
- **Model shape:** `createAgent` rejects a model whose `layers[0] !== OBS_DIM`. The shipped `model.js` (44
  inputs) stops loading until it is retrained and re-exported.
- **CSS network:** the input names are generic (`trainer/css-model.js` loops over `layers[0]`; `css-nn.js`
  loops over `OBS_DIM`). But the documentation says `--x0..--x43` in `css-nn.js` and `css-model.js`, and
  `model.css` must be regenerated with `export-css`.
- **Anti-cheat tests** (`test/policy.test.js`):
  - `policy.js` source must not contain `_internal`, `requirement`, `placeFairly`, `minWindow`, `minGround`,
    `level`, `committed`, `.gap`, `rulesAt` or `difficulty`. A gap feature must be named and computed without
    `.gap`.
  - Only obstacles with x < 600 may be read.
  - Observations must satisfy |x| ≤ 2 in v1 (seeds 1–20, 3,000 frames) and ≤ 2.5 in v2 at 1M (solver play).
    Every candidate here is designed to stay in range: tau is capped at 80/30 = 2.67. **[I]** That cap would
    break the 2.5 bound, so it would need a cap of 75 frames or /32.
  - `keys.jump` does not touch the anti-cheat word list.
- **`test/net.test.js`:** "the shipped model decides exactly like the reference MLP over real play" runs on
  the shipped `model.js`, so it has to be rerun against the new model. Its synthetic shapes use `P.OBS_DIM`
  and follow automatically.
- **`test/browser-check.js`:** it checks the AI mode and that `model.css` matches `model.js`. Run
  `export-css` after re-export, then rerun it.
- **Golden parity** (`test/parity.test.js`): unaffected, because it covers `core.js` only. A feature change
  must not touch `core.js`.
- **Seed or selection hygiene:** a fresh run needs fresh held-out certification. Held-out offset 100+ is the next
  unused offset (HANDOFF D).

## 8. Reproduce

```sh
S=/private/tmp/claude-501/-Users-kyle-Desktop-dino-game/a1a47f55-01e9-48b9-afbe-71ea588921bc/scratchpad/features
PY=/Users/kyle/Desktop/dino-game/.venv/bin/python
cd $S
# Probe A (~2 min on 4 processes)
node probeA.js early 5000 6000 data/early1.jsonl; node probeA.js early 6000 7000 data/early2.jsonl
node probeA.js late 5000 5300 data/late1.jsonl   # ... late2..5 as in section 2
node probeA.js late 8000 8300 data/high1.jsonl 0 high   # ... high2..4
node rescue.js data/crash_early.json data/rescue_early.json   # crash lists built from the jsonl files
node rescue.js data/crash_late.json data/rescue_late.json
$PY -W ignore analyzeA.py; node keyheld.js; node actfreq.js
# Probe B (~15 min on 4 processes)
node cloneB.js gen 1 11 B/d1.bin   # ... d2..d4: seeds 11-21, 21-31, 31-41
./trainall.sh; ./rollall.sh; ./hard.sh
```

Compute used: about 25 CPU-minutes in total, at most 4 processes at a time.
