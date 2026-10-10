# Observation design and input normalisation for small RL/ES agents, read against the dino policy

Compiled 2026-10-10. Branch wt/mac. No project code was changed and no trainer was run.

Conventions:
- **STATED**: what a fetched source says (paraphrase, or a short quote where the fetched text gave one).
- **INFERENCE**: my reasoning about this project. Not from the source.
- **UNVERIFIED**: I could not fetch or read the source text this session. Several PDFs (Salimans 2017, Mania 2018 full text, DQN) came back as binary and no PDF text extractor was available, so some paper-body claims rest on abstracts, search snippets, official code, or secondary papers. Each is labelled.
- Project facts (44 features, speed feature reaching 2.29, v2 speed to ~22 px/frame, 600 px view) come from the task brief and `policy.js observe()`, which I read. I did not re-measure anything.
- Overlap with prior notes: `karpathy-training-techniques.md` covers weight decay, perturbation scale and training-recipe items. Not repeated here. `es-practice.md` did not exist when I looked.

Wording caveat: web-fetch answers were produced by a small summariser model. Quotes below are only those it returned in quotation marks; treat them as near-verbatim, not guaranteed character-exact.

---

## 1. Summary: the most decision-relevant takeaways

1. **Every dino/Flappy project I found uses very few inputs (4-6), and the one with a controlled comparison shows speed was the decisive input.** ArcCreate/DinoBot-NEAT reports that without speed the AI "never knew about the speed of the game", peaked near 1800, and with speed reached the game's cap of 4800 (STATED, README). This project already has speed; it is not a missing-feature case. It does show that speed-dependent timing is the hard part of the task.

2. **No source found compares time-to-collision (tau) against distance as network inputs.** Lee 1976 argues tau is the visual variable for braking timing (STATED, abstract). A web search for RL papers found TTC used as safety feature, reward term or risk estimate, but no head-to-head "tau vs distance" and no claim of speed invariance (STATED by the search; I found no counter-example). Evidence that a tau feature improves learning speed is therefore **absent, not negative**.

3. **Observation normalisation is standard and reported as important for ES/ARS on MuJoCo, but all of it is for locomotion with unbounded, wide-ranging states.** ARS: V1 (raw states) "cannot train policies for the Humanoid-v1 task"; V2 (running mean/std) can (STATED, via ar5iv text). OpenAI's ES code standardises with a running mean/std and clips to ±5.0 (STATED, official code). A Bullet-locomotion ES study reports significantly lower reward without input normalisation (STATED, Frontiers paper). None of this addresses bounded game features that are already roughly in [-1, 1].

4. **This project's inputs are mostly hand-scaled to about [-1, 1], with one exception: the speed feature runs 0 to 2.29 and the per-obstacle speed feature `(speed+offset)/maxSpeed` also exceeds 1 at v2 speeds.** INFERENCE: on a tanh net with 1/sqrt(fan_in) init this is a mild imbalance, not a pathology. ARS motivates normalisation by coordinates of very different scale ([90,100] vs [-1,1]) (STATED); a 0-2.3 range against 0-1 is nowhere near that gap.

5. **Running-stat normalisation is risky for this project specifically.** INFERENCE: it adds non-parameter state (mean/std) that must be stored in `model.js`, duplicated in the CSS network, and updated by workers. ARS V2 needs worker stat merging (STATED, `filter.py` merges buffers). Fixed analytic scaling (e.g. dividing by known maxima) gets most of the benefit with none of that machinery.

6. **Clipping is unnecessary here.** Andrychowicz et al. 2020 found "little evidence that clipping normalized observations" helps; it "might be worth using if there is a risk of extremely high observations due to simulator divergence" (STATED). The dino observations are bounded by construction (x < 600).

7. **The structural gap that matters for late crashes is lookahead in time, not in pixels.** INFERENCE: at ~22 px/frame a 600 px view is ~27 frames; the same 600 px is ~100 frames at speed 6. The relative-x feature changes by 0.037 per frame at top speed and 0.010 at low speed, so the same network must act on a feature whose meaning (frames to impact) varies 3.7x. A frames-to-impact feature (`dx/speed`) would make that explicit. No source tests this, so it is a candidate to probe, not a known fix.

8. **Frame stacking / recurrence is not indicated for a Markov state.** DRQN's authors report recurrence gives no systematic advantage in learning (STATED, abstract); a velocity-obstacle-avoidance study finds stacking and recurrence "cannot consistently make up for the missing velocity data" (STATED, abstract) - the lesson is to provide velocities directly, which `observe()` already does (dino vy, per-obstacle on-screen velocity).

9. **Object type as one-hot is conventional; I found no work that compares one-hot against scalar type for small nets.** INFERENCE: one-hot is the right default for unordered classes (cactus vs ptero). The weak spot is the pterodactyl height class (3 ordered levels, one-hot) and `size/3` (ordered, scalar). Ordered levels already have box `y` and `h` features, so the one-hot height is largely redundant with `y/150`.

10. **Redundancy is cheap for ES but not free.** INFERENCE: of the 13 per-obstacle features, `h`, `w` and the height one-hot overlap with type and size. Dropping them would shrink the input layer (44 -> fewer) and thus the parameter count (32 first-layer weights per input), but no source ranks features by value. Test by ablation, not by literature.

11. **Failure modes at high speed are mostly discussed as reaction-time limits, not as a view-distance limit.** I found no paper that discusses "cannot see far enough ahead relative to speed" for a reactive policy (see section 4). The relevant knowledge is physical (jump airtime vs. closing speed), and is project-specific.

12. **The change cost is high and fixed regardless of how small the change is** (section 5): any edit to `observe()` changes the core hash and means fresh training. Bundle any input change into one new run rather than iterating on observations one at a time.

---

## 2. Findings per question

### Q1. Feature design in dino / Flappy / classic-control / RAM agents

Dino projects (all STATED from READMEs fetched or from search snippets, noted):
- **ArcCreate/DinoBot-NEAT** (README fetched): input sets compared were dino Y distance plus (a) obstacle midtop distance, (b) top-left distance, (c) top-left and top-right distance. Green and Purple were statistically the same (p = 0.63); Yellow and Green differed (p = 0.0047). Without speed, Purple "kept jumping too late" and peaked near 1800; with game speed added it reached the game limit of 4800. The remaining cap came from needing to jump immediately after landing at about 4650.
- **dusanerdeljan/neuroevolution-dino** (README fetched): two networks. Ground obstacles: horizontal distance, height, width, dino y, game speed (5 inputs). Birds: same plus obstacle y (6 inputs).
- **aryanjain28/Chrome-s-T-rex...NEAT** and **javirk/Dino-AI**: from search snippet only, not README. Inputs: speed, horizontal distance, vertical distance (aryanjain28); distance, Y, speed, size (javirk). UNVERIFIED beyond the snippet.
- **TensorFlow.js dino demos** (fritz.ai / heartbeat articles): snippet says same three inputs (speed, width, distance). UNVERIFIED; I did not find a Google-authored "T-Rex runner" AI demo.

Flappy Bird (search snippets only; UNVERIFIED at source level): typical inputs are bird y, vertical offsets to the top/bottom pipe or gap, horizontal distance to the next pipe, sometimes bird vy. A genetic-algorithm write-up reportedly kept five inputs (y, y-distances to the two pipes, x-distance, y-velocity) after testing removal of y-velocity and adding ground/ceiling distances. No controlled ablation found.

Classic control and estool:
- Gymnasium CartPole: 4 observations (position, velocity, angle, angular velocity); position and angle have finite nominal bounds but velocities are unbounded, and the docs say the listed ranges are "not reflective of the allowed values of the state space in an unterminated episode" (STATED, docs). Position and velocity are both supplied, i.e. the Markov state is given directly; there is no frame stacking.
- David Ha's "Evolving Stable Strategies" post (fetched): 2-hidden-layer feed-forward tanh nets; "In all of my experiments I use tanh"; the agent takes the environment's observation as input. BipedalWalker is 24 inputs (10 lidar, angles, contacts) with no absolute coordinates; some tasks give relative coordinates (STATED). Normalisation is not mentioned in that post. The estool README says nothing about input handling (fetched).
- Atari RAM (Sygnowski & Michalewski, arXiv 1605.01335, abstract only): networks on Atari RAM matched a convolutional benchmark on three games (STATED). Mixing RAM and screen did not help. I could not confirm the 128-byte size from the fetched text. Relevant lesson: raw machine state with no feature engineering works with moderate nets in DQN-style training (not ES).

What successful small agents have in common (INFERENCE from the above): current obstacle's relative position and size, own vertical state, and speed. None of the found projects used 3 obstacles, one-hots, or 44 features; this project is already richer than the typical dino AI. No project I found used frame stacking for dino.

Time-to-collision vs distance: no head-to-head found (see takeaway 2).

### Q2. Input normalisation conventions

- **ARS (Mania et al. 2018).** V2 trains "linear maps of states normalized by a mean and standard deviation computed online"; justification: "akin to data whitening", it "ensures that policies put equal weight on the different components of the states"; the example is a coordinate in [90, 100] versus [-1, 1]. V1 without it "cannot train policies for the Humanoid-v1 task" (all STATED, from the ar5iv text via summariser; the arXiv abstract page did not contain this, and the PDF was binary). Official `filter.py` (fetched): divides by `std + 1e-8`, sets elements with std < 1e-7 to `inf` (so they become zero), and has **no clipping**. Stats are merged across workers via a running-stat buffer.
- **OpenAI ES starter code (Salimans et al.).** `MujocoPolicy` standardises `(o - ob_mean) / ob_std` and clips to **[-5, 5]** before the network; hidden activation configurable (tanh among options), `normc_initializer(1.0)` for hidden layers and `0.01` for the continuous output (STATED, official code). The paper body (virtual batch norm, frame skip, preprocessing) I could not read: UNVERIFIED. A search snippet quotes the paper as finding virtual batch normalisation made ES "far more reliable" for Atari policies.
- **OpenAI baselines `VecNormalize`** (fetched): `obs = clip((obs - mean) / sqrt(var + 1e-8), -clipob, clipob)` with default `clipob = 10.`, `cliprew = 10.`, `epsilon = 1e-8` (STATED, code). The task brief's "+-5 in baselines" is not the VecNormalize default; +-5 is the ES starter's value, +-10 is baselines'.
- **What Matters in On-Policy RL (Andrychowicz et al. 2020).** Recommendation (Section 3.3, STATED): "Always use observation normalization and check if value function normalization improves performance." Setup swept input clipping over {1, 2, 5, 10, None}. Found "little evidence that clipping normalized observations" helps. Normalisation yields inputs of "approximately zero mean and standard deviation equal to one" (STATED, Appendix B.9). This is PPO-style on-policy gradient RL on continuous control, not ES.
- **Pagliuca et al. 2020 (Frontiers in Robotics and AI, OpenAI-ES comparison).** Virtual-batch input normalisation helped on locomotion; "the lack of input normalization leads to significantly lower cumulative reward" on WalkerBullet and HumanoidBullet (644.1 vs 1232.2 and 223.1 vs 1866.9); contribution is "minor in simple problems" (STATED).
- **LeCun et al. "Efficient BackProp" (1998).** I intended to cite its advice on zero-mean, equal-variance, decorrelated inputs and tanh saturation; the fetch failed (404/ECONNREFUSED). Any claim from it is UNVERIFIED.

Effect of unnormalised or out-of-range inputs on tanh nets:
- INFERENCE (standard reasoning, no source fetched): a feature of magnitude 2.3 multiplied by weights of std 1/sqrt(44) ≈ 0.15 yields a pre-activation contribution std ≈ 0.34 from that one input, which is modest. Saturation risk comes from accumulated weight growth, which the prior notes (karpathy-training-techniques.md) already tie to ‖θ‖ 8 -> 38 and ~50% of hidden units at |pre| > 2. The input scale and weight norm interact: a larger input scale means a smaller weight is needed to saturate. Whether the speed feature contributes materially is an empirical question for the probe agent.
- STATED (ARS): the harm of unscaled inputs is shown for linear policies on unbounded states. No source I found quantifies it for bounded game inputs on a tanh MLP.

### Q3. Physically meaningful derived features (tau, speed normalisation)

- **Lee 1976**, "A theory of visual control of braking based on information about time-to-collision", Perception 5(4), 437-459, doi 10.1068/p050437. STATED (SAGE page abstract via search): argues the simplest sufficient visual information for braking control is time-to-collision rather than distance, speed or deceleration. The full text was not read. Later bicycle-braking and driving-simulator studies (Rodrigues 2006; Rock & Harris 2006) are reported as broadly supportive but "the tau hypothesis does not provide a complete account of braking control" (STATED, from snippet; UNVERIFIED at source).
- Relevance: Lee's tau is a quantity humans use to time actions, so it is a candidate for "the right variable" for timing a jump. That is a perception claim, not an RL training claim.
- **Evidence it speeds up learning or improves robustness under rising speed:** none found. RL papers that use TTC use it as a risk metric: a collision-risk estimate in the observation reportedly "doubles the performance" in reward in one environment (arXiv 2311.16841, snippet only, UNVERIFIED), and a per-obstacle TTC ranks obstacles in an LSTM+DRL method (snippet only). Neither is a controlled comparison against raw distance.
- **Speed-normalising features** (divide distance by speed): INFERENCE only. Frames-to-impact `dx/speed` is `1/(approach rate)` in px/frame units, i.e. exactly tau in frames. For a net that must output "jump now" at a roughly fixed number of frames before impact (set by jump physics, not speed), a tau feature turns a speed-dependent decision boundary into a fixed one. This is a plausible argument; the dino NEAT README lends indirect support (adding speed fixed the late-jump failure). It does not prove a tau input beats `dx` plus `speed` as separate inputs - an MLP can in principle compute the ratio, but a ratio is awkward for tanh units.
- Hazard: `dx/speed` diverges as dx grows or speed falls; clip or squash (e.g. tanh of tau/T0, or min(tau, T)/T) to keep it bounded. INFERENCE.

### Q4. Partial observability, memory, lookahead

- **Frame stacking vs recurrence.** DRQN abstract (STATED): recurrence is a "viable alternative to stacking a history of frames"; sees one frame per step; "no systematic advantage in learning to play the game"; benefit appears at evaluation when observability changes. The standard DQN stack of four frames (Mnih et al. 2015) I could not read (Nature redirected; arXiv PDF was binary): UNVERIFIED.
- **Karpathy's Pong** (fetched): input is an 80x80 difference image (current frame minus last frame) so the net can see motion (STATED). That is velocity-from-frames; dino's `observe()` supplies velocity explicitly.
- **Missing velocity** (arXiv 2112.12465, abstract): missing velocity in the observation significantly hurts dynamic obstacle avoidance; recurrent networks and frame stacking "cannot consistently make up" for it, but help in simplified scenarios (STATED).
- **Is dino Markov?** INFERENCE from the brief: given dino y, vy, speed, and each visible obstacle's x, size, type, height and velocity, the future is determined except for the invisible spawn sequence. A reactive MLP is sufficient for the visible state. Two non-Markov remnants: (a) what is beyond x = 600 is hidden by design (constraint), (b) the state of animation or timing details not exposed (e.g. a duck-to-stand transition delay, if any - I did not check core.js). Memory could only help with (b).
- **Lookahead vs speed.** I found no paper that analyses "view distance in frames shrinks as speed rises" for a reactive policy. Related general knowledge: the horizon the agent needs is set by the action's dynamics (jump airtime, reaction latency incl. action repeat). INFERENCE: with action repeat n, the controller effectively acts every n frames; at ~27 frames of view and a jump needing k frames of lead, the usable margin is 27 - k - (up to n) frames. If k is a large fraction of 27 at top speed, late crashes would be unavoidable no matter the inputs, and that bound can be computed from core.js physics by the probe agent. I did not compute it.
- Extra obstacles help only if more than one fits in view: with 600 px and gaps that shrink in frames at speed 22, three obstacles may already cover the view. K=3 is probably adequate; I have no evidence either way.

### Q5. One-hot vs scalar encodings

- I found **no source** comparing one-hot to scalar categorical encodings in small tanh/ES networks. Treat everything below as INFERENCE (standard practice).
- Unordered classes (cactus, ptero, ...): one-hot avoids imposing a false order; a scalar code (0, 0.5, 1) forces a monotone relationship between class and downstream units. With 3 types the cost is 3 inputs per slot.
- Ordered levels (ptero height classes, group size): a scalar keeps order; a one-hot costs inputs but lets each level get independent weights. Here the actual box `y` (`y/150`) and `h` already encode height continuously, so the 3-way height one-hot is largely redundant with `y` - at least for pteros. For cacti, `size/3` is a scalar proxy for width (`w/100` carries the same information, since box width is derived from size).
- A thermometer or "binned continuous" encoding is an alternative for the discrete height; no evidence found that it helps.
- The "absent obstacle" encoding (`present = 0`, x = 1) is a convention choice; zero-pad plus presence flag is standard. No source checked.

---

## 3. Candidate changes

Feature indices refer to `observe()` as read: dino features at 0-4; per-obstacle block k at 5 + 13k: present, x, y, w, h, 3 type, size, 3 ptero height, speed.

| Candidate | Rationale and source | What changes in `policy.js` | Risk |
|---|---|---|---|
| Add frames-to-impact `tau = dx / speed` per obstacle (squashed, e.g. min(tau, T)/T) | Lee 1976 (STATED: tau is the braking-timing variable); dino READMEs (STATED: speed fixes late jumps). No ML comparison exists, so rationale is INFERENCE | `PER_OBSTACLE` 13 -> 14 (or replace x); `OBS_DIM` changes; one more line in `observe()` | Medium. Divide-by-small-speed is safe (speed ≥ 6) but redundancy with x and speed may be harmless or helpful; unproven |
| Replace relative x with tau for all obstacles (keep input count) | Same as above; makes the "jump-lead" threshold speed-invariant (INFERENCE) | Same line, no dimension change; but still a core-hash change | Higher: loses raw distance; a net trained on distance geometry cannot warm-start |
| Rescale speed features so they stay within about [0, 1] (divide by true max 22 instead of (speed-6)/7; same for `(speed+offset)/maxSpeed`) | ARS normalisation rationale (STATED) but only mild here; ES starter uses +-5 clip, baselines +-10 (STATED) - neither is needed for 2.3 | Change out[4] and out[b+12] constants only; no dimension change | Low-moderate. Small expected gain (INFERENCE). Cheapest change to try |
| Remove the duplicate speed feature from each obstacle (keep `speedOffset` only, or the global speed) | Redundancy: three copies of `speed + offset` (INFERENCE) | `PER_OBSTACLE` 13 -> 12 or use offset only | Low; but offsets may encode ptero speed difference - check |
| Drop the ptero height one-hot (3) and rely on `y/150`, or drop `h` and `w` where derived | One-hot redundant with continuous box y (INFERENCE); fewer inputs = fewer parameters | `PER_OBSTACLE` 13 -> 10-ish; remove lines | Medium: loses explicit class; need ablation |
| Running mean/std normalisation (ARS V2 / VecNormalize style) | ARS Humanoid result, Andrychowicz "always use" (STATED); but all for unbounded locomotion states | New state in `model.js`, in `policy.js` inference, and in the CSS page; workers must merge stats | High: breaks "analytic features only"; stats must be shipped and kept in sync between trainer and CSS net |
| Clip features to [-5, 5] | ES starter (STATED) | One clamp loop | Very low, but pointless: inputs are bounded (Andrychowicz: little evidence clipping helps) |
| Increase OBS_K from 3 | Typical dino AIs use 1 obstacle; K=3 is already generous (INFERENCE) | `OBS_K`, `OBS_DIM` | Low value unless probe shows >3 obstacles are ever visible |
| Add distance-to-ground / jump-apex or "frames until landing" features | Flappy GA write-up reportedly tested ground/ceiling distances (UNVERIFIED) and kept five inputs | New dino features; needs core physics constants (read-only use) | Medium: derived from visible state, allowed by anti-cheat, but must compute only from visible quantities |
| Frame stacking or an RNN | DRQN, Missing-Velocity abstract (STATED): no systematic gain when velocity is available | New state or history buffer in agent loop; heavier CSS network | High cost, low expected value |
| Time-since-jump-start or action-repeat phase | Non-Markov remnant (INFERENCE) | Agent loop in `policy.js` | Medium; only if probes show timing errors tied to phase |

---

## 4. Cost of changing `observe()` in this repo

Context from `AGENTS.md` only; not verified in code.

- **Core hash.** The hash covers `core.js` and `policy.js`. Any edit to `observe()` changes it and invalidates every existing checkpoint; they `--resume` only with `--force`, and `AGENTS.md` says use `--force` only when decisions are provably unchanged. A feature change alters decisions by definition, so it means a fresh training run.
- **Fresh training.** Dimension changes alter the first-layer shape, so even a warm start from the shipped `runs/v2-h32x32-1m` weights would need surgery (zero-initialised new input columns, re-indexed removed ones). The shipped model took gen 970 to reach a 1M median. A new run needs the full ES budget again; the speed-up to compare against is unknown.
- **CSS network.** `index-css.html` + `css-nn.js` + `model.css` are wired to 44 inputs (`--x0..--x43`) and generated by `trainer/css-model.js`. Any change to input count means editing the generator and the page, then `export-css`. A change that keeps 44 inputs but alters the meaning of features still needs a new `model.css`.
- **Anti-cheat tests.** `test/policy.test.js` enforces that observations use only obstacles with x < 600 and never read `o.gap`, the fairness solver or RNG state. A new derived feature (tau, apex) must be computed from visible fields only; the tests may need new cases for the new features, and any feature reading `speedOffset` or box dimensions is already inside that rule.
- **Browser check.** `node test/browser-check.js` checks that `model.css` matches `model.js` and that the page plays; both must be regenerated after a new model.
- **Parity and net tests.** `test/net.test.js` checks bit-identical decisions for the shipped model; changing `observe()` breaks that for the old model, so the test would need updating or the old model retired. `parity.test.js` should be unaffected (v1 core traces), since `observe()` lives in `policy.js`.
- **Other consumers.** Trainer workers, `eval`, the browser AI mode and the dashboard all share `policy.js`, so one edit propagates to all of them (benefit), but they all need the new model at once.
- **Strategy implication (INFERENCE).** Because every change is expensive, batch the candidate changes into one design (e.g. tau + rescaled speed + removed duplicates), keep the old model for comparison, and decide on the evidence of the empirical probes first.

---

## 5. References

Access date for all: 2026-10-10. Status: VERIFIED = I fetched text that supports the cited claim (in some cases via a summariser, not by reading the raw page myself). UNVERIFIED = not fetched, or fetched but the supporting text was unreadable.

| # | Source | URL | Status |
|---|---|---|---|
| 1 | Mania, Guy, Recht. Simple random search provides a competitive approach to RL (ARS) | https://arxiv.org/abs/1803.07055 (text via https://ar5iv.labs.arxiv.org/html/1803.07055) | VERIFIED (ar5iv text; abs page and PDF alone were not sufficient) |
| 2 | ARS official code, `filter.py` | https://raw.githubusercontent.com/modestyachts/ARS/master/code/filter.py | VERIFIED |
| 3 | ARS official code, `policies.py` | https://raw.githubusercontent.com/modestyachts/ARS/master/code/policies.py | VERIFIED (confirms only that a filter is applied) |
| 4 | Salimans et al. Evolution Strategies as a Scalable Alternative to RL | https://arxiv.org/abs/1703.03864 | VERIFIED for title/authors/abstract only; body text (VBN, frame skip, preprocessing) UNVERIFIED |
| 5 | OpenAI evolution-strategies-starter, `policies.py` | https://raw.githubusercontent.com/openai/evolution-strategies-starter/master/es_distributed/policies.py | VERIFIED (+-5 clip, tanh option) |
| 6 | OpenAI baselines `vec_normalize.py` | https://raw.githubusercontent.com/openai/baselines/master/baselines/common/vec_env/vec_normalize.py | VERIFIED (clipob=10) |
| 7 | Andrychowicz et al. What Matters in On-Policy RL | https://arxiv.org/abs/2006.05990 (text via https://arxiv.org/html/2006.05990v1) | VERIFIED (partial: first 200k chars of 218k; the per-choice analysis figures were not read) |
| 8 | Ha. Evolving Stable Strategies | https://blog.otoro.net/2017/11/12/evolving-stable-strategies/ | VERIFIED |
| 9 | estool README | https://github.com/hardmaru/estool | VERIFIED (says nothing on inputs) |
| 10 | Gymnasium CartPole docs | https://gymnasium.farama.org/environments/classic_control/cart_pole/ | VERIFIED |
| 11 | Sygnowski, Michalewski. Learning from the Memory of Atari 2600 | https://arxiv.org/abs/1605.01335 | VERIFIED (abstract only) |
| 12 | Hausknecht, Stone. Deep Recurrent Q-Learning (DRQN) | https://arxiv.org/abs/1507.06527 | VERIFIED (abstract only) |
| 13 | Missing Velocity in Dynamic Obstacle Avoidance based on Deep RL | https://arxiv.org/abs/2112.12465 | VERIFIED (abstract only) |
| 14 | Karpathy. Deep RL: Pong from Pixels | http://karpathy.github.io/2016/05/31/rl/ | VERIFIED |
| 15 | ArcCreate/DinoBot-NEAT | https://github.com/ArcCreate/DinoBot-NEAT | VERIFIED |
| 16 | dusanerdeljan/neuroevolution-dino | https://github.com/dusanerdeljan/neuroevolution-dino | VERIFIED |
| 17 | Pagliuca, Milano, Nolfi. Efficacy of Modern Neuro-Evolutionary Strategies for Continuous Control Optimization (Frontiers 2020) | https://www.frontiersin.org/articles/10.3389/frobt.2020.00098/full | VERIFIED (via summariser; arXiv 1912.05239 is the preprint) |
| 18 | Lee 1976, A theory of visual control of braking based on information about time-to-collision | https://journals.sagepub.com/doi/10.1068/p050437 | UNVERIFIED (abstract known only through a search-result summary; page not fetched) |
| 19 | Rodrigues et al. 2006, bicycle braking tau study | https://repositorio.ufc.br/handle/riufc/62866 | UNVERIFIED (search snippet) |
| 20 | Rock, Harris 2006, JEP:HPP 32(2), braking and tau | not located | UNVERIFIED (search snippet only) |
| 21 | aryanjain28 Chrome T-rex NEAT | https://github.com/aryanjain28/Chrome-s-T-rex-game-played-by-machine-using-genetic-algorithm-NEAT | UNVERIFIED (search snippet) |
| 22 | javirk/Dino-AI | https://github.com/javirk/Dino-AI | UNVERIFIED (search snippet) |
| 23 | Fritz.ai TensorFlow.js dino | https://fritz.ai/automating-chrome-dinosaur-game/ | UNVERIFIED (search snippet) |
| 24 | Flappy Bird input write-ups (Medium GA post; GitHub NEAT projects; arXiv 2207.14140, 2003.09579) | https://arxiv.org/pdf/2207.14140 , https://arxiv.org/pdf/2003.09579 | UNVERIFIED (search snippets only) |
| 25 | TTC/RL papers (arXiv 1902.00089, 2311.16841, 2609.17347) | https://arxiv.org/pdf/1902.00089 | UNVERIFIED (search snippets only) |
| 26 | Mnih et al. 2013/2015 DQN (frame stacking) | https://arxiv.org/pdf/1312.5602 | UNVERIFIED (PDF returned as binary, Nature page redirected) |
| 27 | LeCun et al. Efficient BackProp | http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf | UNVERIFIED (fetch failed) |

Counts: 17 verified (several only for the abstract or a portion), 10 unverified.
