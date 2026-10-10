"""Probe B: behaviour-clone the exact solver into a tanh MLP (policy.js layout).
usage: python trainB.py <variant> <hidden e.g. 32,32> <out.json> <epochs> <data.bin> [more.bin ...]
Rows: float32 [54 features, label, weight]. Inputs are fed RAW (as policy.js would),
no standardisation, so input conditioning is part of what is tested.
Class weights: sqrt-balanced (w_c = sqrt(N / (2 N_c))), times the subsampling weight."""
import sys, json, numpy as np
ROW = 54
VARIANTS = json.load(open('variants.json'))
variant, hidden, out, epochs = sys.argv[1], [int(h) for h in sys.argv[2].split(',')], sys.argv[3], int(sys.argv[4])
files = sys.argv[5:]
D = np.concatenate([np.fromfile(f, dtype=np.float32).reshape(-1, ROW + 2) for f in files])
cols = VARIANTS[variant]
X = D[:, cols].astype(np.float32); y = D[:, ROW].astype(np.int64); sw = D[:, ROW + 1].astype(np.float32)
del D
import os
if os.environ.get('NROWS'):
    keep = np.random.default_rng(99).choice(len(y), int(os.environ['NROWS']), replace=False)
    X, y, sw = X[keep], y[keep], sw[keep]
N = len(y); counts = np.array([(sw * (y == c)).sum() for c in range(3)])
cw = np.where(counts > 0, np.sqrt(counts.sum() / (2 * np.maximum(counts, 1))), 0.0)
w_all = (sw * cw[y]).astype(np.float32)
layers = [X.shape[1]] + hidden + [3]
import os
rng = np.random.default_rng(int(os.environ.get("SEED", "1")))
Ws = [rng.normal(0, 1 / np.sqrt(layers[i]), (layers[i + 1], layers[i])).astype(np.float32) for i in range(len(layers) - 1)]
bs = [np.zeros(layers[i + 1], np.float32) for i in range(len(layers) - 1)]
params = Ws + bs
m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
lr0, B, t = 2e-3, 1024, 0
L = len(Ws)

def forward(xb):
    hs = [xb]
    for i in range(L):
        z = hs[-1] @ Ws[i].T + bs[i]
        hs.append(np.tanh(z) if i < L - 1 else z)
    return hs

steps = epochs * (N // B)
for ep in range(epochs):
    perm = rng.permutation(N); tot = 0.0
    for s in range(N // B):
        idx = perm[s * B:(s + 1) * B]
        xb, yb, wb = X[idx], y[idx], w_all[idx]
        hs = forward(xb)
        z = hs[-1]; z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
        tot += float((wb * -np.log(p[np.arange(B), yb] + 1e-12)).sum())
        g = p; g[np.arange(B), yb] -= 1; g *= (wb / wb.sum())[:, None]
        gW = [None] * L; gb = [None] * L
        for i in reversed(range(L)):
            gW[i] = g.T @ hs[i]; gb[i] = g.sum(0)
            if i > 0: g = (g @ Ws[i]) * (1 - hs[i] ** 2)
        grads = gW + gb
        t += 1; lr = lr0 * 0.5 * (1 + np.cos(np.pi * t / steps))
        for j, (pp, gg) in enumerate(zip(params, grads)):
            m[j] = 0.9 * m[j] + 0.1 * gg; v[j] = 0.999 * v[j] + 0.001 * gg * gg
            pp -= lr * (m[j] / (1 - 0.9 ** t)) / (np.sqrt(v[j] / (1 - 0.999 ** t)) + 1e-8)
    # train-set balanced accuracy (subsample)
    sub = rng.choice(N, min(N, 200000), replace=False)
    pred = forward(X[sub])[-1].argmax(1)
    acc = [float(((pred == c) & (y[sub] == c)).sum() / max(1, (y[sub] == c).sum())) for c in (0, 1)]
    print(f'{variant} ep{ep} loss {tot / w_all.sum():.4f} recall none {acc[0]:.4f} jump {acc[1]:.4f}', flush=True)
flat = []
for i in range(L): flat += Ws[i].astype(np.float64).ravel().tolist() + bs[i].astype(np.float64).tolist()
json.dump(dict(variant=variant, layers=layers, params=flat, recall=acc), open(out, 'w'))
