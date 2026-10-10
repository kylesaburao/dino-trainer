"""Held-out demonstration agreement per epoch snapshot (research/prereg-bc-sweep.md §4).

For each run dir given, loads <dir>/snapshots/epoch-NN.json ({layers, params} in the
policy.createNet flat layout: per layer, nOut x nIn row-major weights, then nOut biases),
evaluates the tanh MLP in float64 (as policy.js does) on the held-out demos (bc-gen seeds
41-60: float32 rows [44 features, label, weight]) and writes <dir>/agreement.json:
  per epoch: agreement = sum(w * [argmax == label]) / sum(w), and the weighted recall of
  "none" (label 0) and "jump" (label 1), and, when the demos hold duck rows (label 2,
  teacher-duck), recallDuck and duckPredShare (research/prereg-teacher-duck-step2.md); plus the final epoch's predictions on a fixed
  spot-check subset (every STRIDE-th row) for spot-check.js (policy.js argmax).
argmax takes the first index on ties, like policy.js.
--split-x4 T (research/prereg-v3-clones.md §4.0 item 4): each epoch row also gets
  split = {x4: T, le: {...}, gt: {...}}, the same weighted measures (agreement, rows, weight,
  recallNone, recallJump and, with duck rows, recallDuck) on the rows with feature 4 (the speed
  feature (speed - 6) / 7) <= T and > T. T = 2.5 separates rules v2 and the v3 part below the step
  (speed <= 23.5) from the v3 part past the step. A subset with no rows (or no weight for a class)
  reports null. Without the flag the output is unchanged.
Usage: python agreement.py [--split-x4 T] <heldout demos.bin> <run dir> [<run dir> ...]
"""
import json, os, sys, glob
import numpy as np

OBS, ROW, STRIDE, CHUNK = 44, 46, 349, 131072

ARGS = sys.argv[1:]
SPLIT = None
if '--split-x4' in ARGS:
    i = ARGS.index('--split-x4')
    SPLIT = float(ARGS[i + 1])
    del ARGS[i:i + 2]
sys.argv[1:] = ARGS

D = np.fromfile(sys.argv[1], dtype=np.float32).reshape(-1, ROW)
X = D[:, :OBS].astype(np.float64)
y = D[:, OBS].astype(np.int64)
w = D[:, OBS + 1].astype(np.float64)
del D
spot = np.arange(0, len(y), STRIDE)
HAS_DUCK = bool((y == 2).any())
X4 = X[:, 4] if SPLIT is not None else None


def subset(hit, mask):
    """Weighted agreement and recall on the rows in mask (null where there is no weight)."""
    ws = w * mask
    div = lambda num, den: float(num / den) if den > 0 else None
    r = dict(rows=int(mask.sum()), weight=float(ws.sum()), agreement=div((ws * hit).sum(), ws.sum()))
    for c, k in ((0, 'recallNone'), (1, 'recallJump')) + (((2, 'recallDuck'),) if HAS_DUCK else ()):
        r[k] = div((ws * (hit & (y == c))).sum(), (ws * (y == c)).sum())
    return r


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
        row = dict(epoch=s['epoch'], agreement=float((w * hit).sum() / w.sum()), recallNone=rec[0], recallJump=rec[1],
                   unweighted=float(hit.mean()), predCounts=[int((pred == c).sum()) for c in range(3)])
        if HAS_DUCK:  # teacher-duck demos (label 2); keys absent for jump demos, so old outputs are unchanged
            dm = (y == 2)
            row['recallDuck'] = float((w * (hit & dm)).sum() / (w * dm).sum())
            # weighted share of true-duck rows predicted none / jump / duck
            row['duckPredShare'] = [float((w * (dm & (pred == c))).sum() / (w * dm).sum()) for c in range(3)]
        if SPLIT is not None:  # keys absent without --split-x4, so old outputs are unchanged
            row['split'] = dict(x4=SPLIT, le=subset(hit, X4 <= SPLIT), gt=subset(hit, X4 > SPLIT))
        out['epochs'].append(row)
        out['layers'] = s['layers']
        final_pred = pred
    if final_pred is not None:
        out['spot'] = dict(stride=STRIDE, epoch=out['epochs'][-1]['epoch'], idx=spot.tolist(), pred=final_pred[spot].tolist())
    with open(os.path.join(run, 'agreement.json'), 'w') as fh: json.dump(out, fh)
    e = out['epochs']
    print(out['run'], ' '.join('%.5f' % r['agreement'] for r in e), flush=True)
    if SPLIT is not None and e:
        fmt = lambda v: 'n/a' if v is None else '%.5f' % v
        s = e[-1]['split']
        print('  final epoch, x4 <= %g: %s (%d rows); x4 > %g: %s (%d rows)' % (
            SPLIT, fmt(s['le']['agreement']), s['le']['rows'], SPLIT, fmt(s['gt']['agreement']), s['gt']['rows']), flush=True)
