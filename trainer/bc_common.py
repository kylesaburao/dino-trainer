"""Host-side parts of the python BC training engines (trainer/bc_train.py, numpy; later the MLX
engine, research/plan-gpu-trainer.md §4.1): argument parsing, demo rows, class and transition
weights, distillation targets, the per-seed init, quantization and the flat params layout.
Everything here runs on the CPU with numpy and is shared verbatim by every engine, so a given
--seed means the same init, batch order and recall subsample on each of them. Moved out of
bc_train.py expression for expression: the numpy backend's output is byte-identical.
Must stay Python 3.9 compatible (the repo .venv)."""
import argparse, json, math, os
import numpy as np


def arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demos', required=True)
    ap.add_argument('--obs-dim', type=int, required=True)
    ap.add_argument('--hidden', required=True)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', required=True)
    ap.add_argument('--snapshot-dir', default=None)
    ap.add_argument('--teacher-logits', default=None)
    ap.add_argument('--distill-mode', default='soft', choices=['soft', 'hard'])
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--alpha', type=float, default=0.0)
    ap.add_argument('--quant-bits', type=int, default=0)
    ap.add_argument('--quant-scheme', default='row', choices=['row', 'layer', 'pow2'])
    ap.add_argument('--extra-demos', action='append', default=[])
    ap.add_argument('--transition-weight', type=float, default=None)
    return ap


class Data:
    """The demo rows and everything derived from them that every engine trains on.
    X float32 features; y int64 training labels (the teacher's argmax under hard distillation);
    y_solver the solver labels (recall and the reported loss); w_all float32 row weights (row
    weight x class weight [x transition weight]); Q float32 soft targets or None."""

    def __init__(self, a):
        ROW = a.obs_dim
        D = np.fromfile(a.demos, dtype=np.float32).reshape(-1, ROW + 2)
        if a.extra_demos:
            # --extra-demos (research/prereg-teacher-duck-release-fix.md §4.1): rows appended in order.
            D = np.concatenate([D] + [np.fromfile(f, dtype=np.float32).reshape(-1, ROW + 2) for f in a.extra_demos])
        cols = list(range(ROW))
        X = D[:, cols].astype(np.float32); y = D[:, ROW].astype(np.int64); sw = D[:, ROW + 1].astype(np.float32)
        del D
        N = len(y); counts = np.array([(sw * (y == c)).sum() for c in range(3)])
        cw = np.where(counts > 0, np.sqrt(counts.sum() / (2 * np.maximum(counts, 1))), 0.0)
        w_all = (sw * cw[y]).astype(np.float32)
        self.has_duck = bool(counts[2] > 0)
        self.transitions = None
        if a.transition_weight is not None and a.transition_weight != 1:
            # --transition-weight W: duck-transition rows (trainer/bc.js transitionMask: grounded, and
            # onset = not ducking with label 2, or release = ducking with label != 2) weigh W times
            # more. Applied after the class weights, which stay those of the unmodified weights.
            grounded = X[:, 2] == 0
            onset = grounded & (X[:, 3] == 0) & (y == 2)
            release = grounded & (X[:, 3] == 1) & (y != 2)
            w_all[onset | release] *= np.float32(a.transition_weight)
            self.transitions = dict(onset=int(onset.sum()), release=int(release.sum()), releaseToJump=int((release & (y == 1)).sum()))
            print('transitions: ' + json.dumps(self.transitions), flush=True)
        self.y_solver = y                 # recall and the reported loss always use the solver labels
        self.distill = a.teacher_logits is not None
        self.Q = None
        if self.distill:
            TL = np.fromfile(a.teacher_logits, dtype=np.float64).reshape(-1, 3)
            if len(TL) != N: raise SystemExit('teacher logits: %d rows, demos have %d' % (len(TL), N))
            if a.distill_mode == 'hard':
                y = TL.argmax(1).astype(np.int64)   # first index wins ties, as policy.js
            else:
                zt = TL / a.tau; zt = zt - zt.max(1, keepdims=True); Q = np.exp(zt); Q /= Q.sum(1, keepdims=True)
                self.Q = Q.astype(np.float32)
            del TL
        self.X, self.y, self.w_all, self.N = X, y, w_all, N


def layer_sizes(a, X):
    return [X.shape[1]] + [int(h) for h in a.hidden.split(',')] + [3]


def init_params(rng, layers):
    """float32 (Ws, bs): N(0, 1/fan_in) weights, zero biases; the first draws of a run's rng."""
    Ws = [rng.normal(0, 1 / np.sqrt(layers[i]), (layers[i + 1], layers[i])).astype(np.float32) for i in range(len(layers) - 1)]
    bs = [np.zeros(layers[i + 1], np.float32) for i in range(len(layers) - 1)]
    return Ws, bs


def quantize(Wf, bits, scheme):
    """float32 weight matrix -> (float64 q * scale, float64 scales); trainer/bc.js quantizeInto."""
    qmax = 2 ** (bits - 1) - 1
    W64 = Wf.astype(np.float64)
    mx = np.abs(W64).max(1, keepdims=True) if scheme == 'row' else np.full((W64.shape[0], 1), np.abs(W64).max())
    sc = np.where(mx > 0, mx / qmax, 1.0)
    k = np.clip(np.floor(W64 / sc + 0.5), -qmax, qmax)
    return k * sc, (sc[:, 0] if scheme == 'row' else sc[:1, 0])


def pow2_exponent(m, qmax):
    """Largest integer k with m * 2^k <= qmax (0 for m = 0); trainer/bc.js pow2Exponent."""
    if not m > 0: return 0
    k = math.floor(math.log2(qmax / m))
    while m * 2.0 ** (k + 1) <= qmax: k += 1
    while m * 2.0 ** k > qmax: k -= 1
    return k


def quantize_pow2(Wf, bf, bits):
    """float32 layer (W, b) -> (float64 Wq, float64 bq, k): q * 2^-k with one k per layer;
    trainer/bc.js quantizePow2Into."""
    qmax = 2 ** (bits - 1) - 1
    W64, b64 = Wf.astype(np.float64), bf.astype(np.float64)
    k = pow2_exponent(float(max(np.abs(W64).max(), np.abs(b64).max())), qmax)
    up, down = 2.0 ** k, 2.0 ** -k
    q = lambda x: np.clip(np.floor(x * up + 0.5), -qmax, qmax) * down
    return q(W64), q(b64), k


def flat_params(Ws, bs, bits, scheme):
    """The policy.createNet flat layout (per layer: row-major W, then b) as float64 Python floats;
    on the quantization grid when bits is non-zero."""
    L = len(Ws)
    flat = []
    if bits and scheme == 'pow2':
        for i in range(L):
            wq, bq, _ = quantize_pow2(Ws[i], bs[i], bits)
            flat += wq.ravel().tolist() + bq.tolist()
        return flat
    if bits:
        for i in range(L): flat += quantize(Ws[i], bits, scheme)[0].ravel().tolist() + bs[i].astype(np.float64).tolist()
        return flat
    for i in range(L): flat += Ws[i].astype(np.float64).ravel().tolist() + bs[i].astype(np.float64).tolist()
    return flat


def quant_extras(Ws, bs, bits, scheme):
    """The result's quantization record: {'k': [...]} for pow2, {'scales': [...]} otherwise, {} when off."""
    if bits and scheme == 'pow2': return dict(k=[quantize_pow2(Ws[i], bs[i], bits)[2] for i in range(len(Ws))])
    if bits: return dict(scales=[quantize(W, bits, scheme)[1].tolist() for W in Ws])
    return {}


def recall(pred, ys, has_duck):
    """Per-class recall of pred against the solver labels ys: {none, jump[, duck]}."""
    acc = [float(((pred == c) & (ys == c)).sum() / max(1, (ys == c).sum())) for c in ((0, 1, 2) if has_duck else (0, 1))]
    r = dict(none=acc[0], jump=acc[1])
    if has_duck: r['duck'] = acc[2]
    return r


def write_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f: json.dump(obj, f)
    os.replace(tmp, path)


def write_snapshot(snapshot_dir, epochs, ep, layers, params):
    """<snapshot_dir>/epoch-NN.json ({layers, params, epoch}; NN and epoch 1-based)."""
    name = 'epoch-%0*d.json' % (max(2, len(str(epochs))), ep + 1)
    write_json(os.path.join(snapshot_dir, name), dict(layers=layers, params=params, epoch=ep + 1))
