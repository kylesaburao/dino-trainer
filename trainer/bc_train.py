"""Python/numpy backend for `node trainer/cli.js bc-train --backend python`.

usage: python bc_train.py --demos demos.bin --obs-dim 44 --hidden 32,32 --epochs 15
                          --batch 1024 --lr 0.002 --seed 1 --out result.json

A port of research/scripts/feature-sufficiency/trainB.py (variant base), kept
expression for expression (dtypes, rng call order) so it reproduces that
prototype's params exactly on the same rows. The algorithm is the one in
trainer/bc.js (the node backend). Rows: float32 [obs-dim features, label, weight].
Labels are policy.js action indices (0 none, 1 jump, 2 duck). Class weights cover
all three classes, as in bc.js (absent classes get 0). Prints one JSON line per
epoch ({epoch, loss, recall: {none, jump[, duck]}, seconds}; duck only when the
demos have label-2 rows) and writes {layers, params, log} to --out; Node owns best.json.
--snapshot-dir DIR (optional) also writes DIR/epoch-NN.json ({layers, params, epoch};
NN and epoch 1-based) after each epoch's last optimiser step: a float64 copy only, so it
draws no RNG and changes no computation.
Distillation and quantization (off by default; trainer/bc.js header has the definitions):
--teacher-logits FILE (float64 rows x 3, computed by Node from the teacher) with
--distill-mode soft|hard [--tau T --alpha A]; --quant-bits N [--quant-scheme row|layer]
(QAT: fake-quantized weights in forward and backward, straight-through to the float
weights; the output params are the float64 q * scale, and the output gains `scales`).
Neither draws RNG. Distilled rows add `distillLoss`; `loss` stays the CE vs the solver.
Needs numpy (the repo's .venv; the PYTHON env var overrides the interpreter)."""
import argparse, json, os, time
import numpy as np

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
ap.add_argument('--quant-scheme', default='row', choices=['row', 'layer'])
a = ap.parse_args()

ROW = a.obs_dim
hidden = [int(h) for h in a.hidden.split(',')]
epochs = a.epochs
D = np.fromfile(a.demos, dtype=np.float32).reshape(-1, ROW + 2)
cols = list(range(ROW))
X = D[:, cols].astype(np.float32); y = D[:, ROW].astype(np.int64); sw = D[:, ROW + 1].astype(np.float32)
del D
N = len(y); counts = np.array([(sw * (y == c)).sum() for c in range(3)])
cw = np.where(counts > 0, np.sqrt(counts.sum() / (2 * np.maximum(counts, 1))), 0.0)
w_all = (sw * cw[y]).astype(np.float32)
has_duck = bool(counts[2] > 0)
y_solver = y                      # recall and the reported loss always use the solver labels
distill = a.teacher_logits is not None
Q = None
if distill:
    TL = np.fromfile(a.teacher_logits, dtype=np.float64).reshape(-1, 3)
    if len(TL) != N: raise SystemExit('teacher logits: %d rows, demos have %d' % (len(TL), N))
    if a.distill_mode == 'hard':
        y = TL.argmax(1).astype(np.int64)   # first index wins ties, as policy.js
    else:
        zt = TL / a.tau; zt = zt - zt.max(1, keepdims=True); Q = np.exp(zt); Q /= Q.sum(1, keepdims=True)
        Q = Q.astype(np.float32)
    del TL
QB = a.quant_bits
layers = [X.shape[1]] + hidden + [3]
rng = np.random.default_rng(a.seed)
Ws = [rng.normal(0, 1 / np.sqrt(layers[i]), (layers[i + 1], layers[i])).astype(np.float32) for i in range(len(layers) - 1)]
bs = [np.zeros(layers[i + 1], np.float32) for i in range(len(layers) - 1)]
params = Ws + bs
m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
lr0, B, t = a.lr, a.batch, 0
L = len(Ws)


def forward(xb, W=None):
    W = Ws if W is None else W
    hs = [xb]
    for i in range(L):
        z = hs[-1] @ W[i].T + bs[i]
        hs.append(np.tanh(z) if i < L - 1 else z)
    return hs


def quantize(Wf):
    """float32 weight matrix -> (float64 q * scale, float64 scales); trainer/bc.js quantizeInto."""
    qmax = 2 ** (QB - 1) - 1
    W64 = Wf.astype(np.float64)
    mx = np.abs(W64).max(1, keepdims=True) if a.quant_scheme == 'row' else np.full((W64.shape[0], 1), np.abs(W64).max())
    sc = np.where(mx > 0, mx / qmax, 1.0)
    k = np.clip(np.floor(W64 / sc + 0.5), -qmax, qmax)
    return k * sc, (sc[:, 0] if a.quant_scheme == 'row' else sc[:1, 0])


def flat_params():
    flat = []
    if QB:
        for i in range(L): flat += quantize(Ws[i])[0].ravel().tolist() + bs[i].astype(np.float64).tolist()
        return flat
    for i in range(L): flat += Ws[i].astype(np.float64).ravel().tolist() + bs[i].astype(np.float64).tolist()
    return flat


def distill_grad(zl, p, idx, yb, wb):
    """Distilled output gradient (before the batch-weight scale), the solver-label CE sum and
    the objective sum; definitions in the trainer/bc.js header."""
    ar = np.arange(len(idx))
    ys = y_solver[idx]
    hce = -np.log(p[ar, ys] + 1e-12)
    if Q is None:   # hard: yb holds the teacher's labels
        obj = -np.log(p[ar, yb] + 1e-12)
        g = p.copy(); g[ar, yb] -= 1
    else:
        tau, alpha = a.tau, a.alpha
        zs = zl / tau; zs = zs - zs.max(1, keepdims=True); pt = np.exp(zs); pt /= pt.sum(1, keepdims=True)
        q = Q[idx]
        sce = -(q * np.log(pt + 1e-12)).sum(1)
        obj = alpha * hce + (1 - alpha) * tau * tau * sce
        g = (1 - alpha) * tau * (pt - q)
        if alpha > 0:
            oh = np.zeros_like(p); oh[ar, ys] = 1
            g = alpha * (p - oh) + g
    return g, float((wb * hce).sum()), float((wb * obj).sum())


def write_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f: json.dump(obj, f)
    os.replace(tmp, path)


if a.snapshot_dir: os.makedirs(a.snapshot_dir, exist_ok=True)
log = []
steps = epochs * (N // B)
for ep in range(epochs):
    t0 = time.time()
    perm = rng.permutation(N); tot = 0.0; dtot = 0.0
    for s in range(N // B):
        idx = perm[s * B:(s + 1) * B]
        xb, yb, wb = X[idx], y[idx], w_all[idx]
        Wf = [quantize(W)[0].astype(np.float32) for W in Ws] if QB else Ws
        hs = forward(xb, Wf)
        z = hs[-1]; z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
        if not distill:
            tot += float((wb * -np.log(p[np.arange(B), yb] + 1e-12)).sum())
            g = p; g[np.arange(B), yb] -= 1; g *= (wb / wb.sum())[:, None]
        else:
            g, hsum, osum = distill_grad(hs[-1], p, idx, yb, wb)
            tot += hsum; dtot += osum
            g *= (wb / wb.sum())[:, None]
        gW = [None] * L; gb = [None] * L
        for i in reversed(range(L)):
            gW[i] = g.T @ hs[i]; gb[i] = g.sum(0)
            if i > 0: g = (g @ Wf[i]) * (1 - hs[i] ** 2)
        grads = gW + gb
        t += 1; lr = lr0 * 0.5 * (1 + np.cos(np.pi * t / steps))
        for j, (pp, gg) in enumerate(zip(params, grads)):
            m[j] = 0.9 * m[j] + 0.1 * gg; v[j] = 0.999 * v[j] + 0.001 * gg * gg
            pp -= lr * (m[j] / (1 - 0.9 ** t)) / (np.sqrt(v[j] / (1 - 0.999 ** t)) + 1e-8)
    if a.snapshot_dir:
        name = 'epoch-%0*d.json' % (max(2, len(str(epochs))), ep + 1)
        write_json(os.path.join(a.snapshot_dir, name), dict(layers=layers, params=flat_params(), epoch=ep + 1))
    # train-set recall per class (subsample)
    sub = rng.choice(N, min(N, 200000), replace=False)
    pred = forward(X[sub], [quantize(W)[0].astype(np.float32) for W in Ws] if QB else None)[-1].argmax(1)
    acc = [float(((pred == c) & (y_solver[sub] == c)).sum() / max(1, (y_solver[sub] == c).sum())) for c in ((0, 1, 2) if has_duck else (0, 1))]
    recall = dict(none=acc[0], jump=acc[1])
    if has_duck: recall['duck'] = acc[2]
    row = dict(epoch=ep, loss=float(tot / w_all.sum()), recall=recall, seconds=time.time() - t0)
    if distill: row['distillLoss'] = float(dtot / w_all.sum())
    log.append(row)
    print(json.dumps(row), flush=True)
result = dict(layers=layers, params=flat_params(), log=log)
if QB: result['scales'] = [quantize(W)[1].tolist() for W in Ws]
json.dump(result, open(a.out, 'w'))
