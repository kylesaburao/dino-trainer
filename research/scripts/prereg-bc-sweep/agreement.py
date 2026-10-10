"""Held-out demonstration agreement per epoch snapshot (research/prereg-bc-sweep.md §4).

For each run dir given, loads <dir>/snapshots/epoch-NN.json ({layers, params} in the
policy.createNet flat layout: per layer, nOut x nIn row-major weights, then nOut biases),
evaluates the tanh MLP in float64 (as policy.js does) on the held-out demos (bc-gen seeds
41-60: float32 rows [44 features, label, weight]) and writes <dir>/agreement.json:
  per epoch: agreement = sum(w * [argmax == label]) / sum(w), and the weighted recall of
  "none" (label 0) and "jump" (label 1); plus the final epoch's predictions on a fixed
  spot-check subset (every STRIDE-th row) for spot-check.js (policy.js argmax).
argmax takes the first index on ties, like policy.js.
Usage: python agreement.py <heldout demos.bin> <run dir> [<run dir> ...]
"""
import json, os, sys, glob
import numpy as np

OBS, ROW, STRIDE, CHUNK = 44, 46, 349, 131072

D = np.fromfile(sys.argv[1], dtype=np.float32).reshape(-1, ROW)
X = D[:, :OBS].astype(np.float64)
y = D[:, OBS].astype(np.int64)
w = D[:, OBS + 1].astype(np.float64)
del D
spot = np.arange(0, len(y), STRIDE)


def unflatten(layers, params):
    p = np.asarray(params, dtype=np.float64)
    Ws, bs, k = [], [], 0
    for l in range(1, len(layers)):
        nin, nout = layers[l - 1], layers[l]
        Ws.append(p[k:k + nin * nout].reshape(nout, nin)); k += nin * nout
        bs.append(p[k:k + nout]); k += nout
    assert k == len(p), (k, len(p))
    return Ws, bs


def predict(Ws, bs, Xc):
    h = Xc
    for i, (W, b) in enumerate(zip(Ws, bs)):
        with np.errstate(all='ignore'):  # Accelerate matmul raises spurious FP flags; finiteness is asserted below
            z = h @ W.T + b
        assert np.isfinite(z).all(), 'non-finite activation'
        h = np.tanh(z) if i < len(Ws) - 1 else z
    return h.argmax(1)


for run in sys.argv[2:]:
    files = sorted(glob.glob(os.path.join(run, 'snapshots', 'epoch-*.json')))
    out = dict(run=os.path.basename(run.rstrip('/')), heldout=sys.argv[1], rows=int(len(y)), weight=float(w.sum()), epochs=[])
    final_pred = None
    for f in files:
        s = json.load(open(f))
        Ws, bs = unflatten(s['layers'], s['params'])
        pred = np.concatenate([predict(Ws, bs, X[i:i + CHUNK]) for i in range(0, len(y), CHUNK)])
        hit = (pred == y)
        rec = {c: float((w * (hit & (y == c))).sum() / (w * (y == c)).sum()) for c in (0, 1)}
        out['epochs'].append(dict(epoch=s['epoch'], agreement=float((w * hit).sum() / w.sum()), recallNone=rec[0], recallJump=rec[1],
                                  unweighted=float(hit.mean()), predCounts=[int((pred == c).sum()) for c in range(3)]))
        out['layers'] = s['layers']
        final_pred = pred
    if final_pred is not None:
        out['spot'] = dict(stride=STRIDE, epoch=out['epochs'][-1]['epoch'], idx=spot.tolist(), pred=final_pred[spot].tolist())
    with open(os.path.join(run, 'agreement.json'), 'w') as fh: json.dump(out, fh)
    e = out['epochs']
    print(out['run'], ' '.join('%.5f' % r['agreement'] for r in e), flush=True)
