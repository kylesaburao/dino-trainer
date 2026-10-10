"""Probe A separability: crash vs pass at tau-aligned snapshots.
Classifiers: L2 logistic regression (5-fold stratified CV AUC) and LOO 1-NN on
standardised features. Feature sets: raw 44; 44 + engineered (allowed info);
44 + hidden/forbidden full-state extras.
usage: python analyzeA.py"""
import json, glob, numpy as np
rng = np.random.default_rng(0)
W = 600

def auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

def logreg_cv(X, y, lam=1.0, k=5):
    X = X[:, X.std(0) > 1e-6]
    X = (X - X.mean(0)) / X.std(0)
    idx = np.arange(len(y)); s = np.zeros(len(y))
    folds = np.zeros(len(y), int)
    for c in (0, 1):
        ii = idx[y == c]; rng.shuffle(ii); folds[ii] = np.arange(len(ii)) % k
    for f in range(k):
        tr, te = folds != f, folds == f
        w = np.zeros(X.shape[1]); b = 0.0
        pw = (y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())
        sw = np.where(y[tr] == 1, pw, 1.0)
        for _ in range(400):   # Newton-free: gradient descent w/ fixed lr on convex loss
            z = X[tr] @ w + b; p = 1 / (1 + np.exp(-z))
            g = sw * (p - y[tr])
            w -= 0.5 * (X[tr].T @ g / sw.sum() + lam * w / tr.sum())
            b -= 0.5 * g.sum() / sw.sum()
        s[te] = X[te] @ w + b
    return auc(y, s)

def nn_loo(X, y):
    X = X[:, X.std(0) > 1e-6]
    X = (X - X.mean(0)) / X.std(0)
    D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1); np.fill_diagonal(D, np.inf)
    # score = dist to nearest pass - dist to nearest crash (higher -> crash-like)
    dp = np.where(y[None, :] == 0, D, np.inf).min(1); dc = np.where(y[None, :] == 1, D, np.inf).min(1)
    s = np.sqrt(dp) - np.sqrt(dc)
    nn = D.argmin(1); acc1 = (y[nn] == y)[y == 1].mean()
    return auc(y, s), acc1, np.sqrt(D.min(1))

def load(pattern, pred):
    out = []
    for fn in sorted(glob.glob(pattern)):
        for l in open(fn):
            e = json.loads(l)
            if pred(e): out.append(e)
    return out

def tau_feats(x):
    """Engineered from allowed info only: per-obstacle tau in frames (relx*W / v),
    gap between obstacle 0 and 1 in frames, speed/22."""
    f = []
    for k in range(3):
        b = 5 + 13 * k
        f.append(min(x[b + 1] * W / max(x[b + 12] * 13, 1e-6), 80) / 30 if x[b] else 80 / 30)
    if x[5] and x[18]:
        gap_px = x[19] * W - (x[6] * W + x[8] * 100)
        f.append(gap_px / (x[30] * 13) / 30)
    else: f.append(80 / 30)
    f.append((x[4] * 7 + 6) / 22)
    return f

def time_to_land(y, vy):
    t = 0
    while y < 98 and t < 80:
        y += vy; vy += 0.6; t += 1
    return t

def run(name, encs, taus, extra_hidden):
    y = np.array([1 if e['out'] == 'crash' else 0 for e in encs])
    print(f'\n=== {name}: {y.sum()} crash vs {(y == 0).sum()} pass')
    rows = []
    for T in taus:
        keep = [i for i, e in enumerate(encs) if str(T) in e['snaps']]
        if not keep: continue
        E = [encs[i] for i in keep]; yy = y[keep]
        X = np.array([e['snaps'][str(T)]['x'] for e in E])
        A = np.array([e['snaps'][str(T)]['a'] for e in E])
        Xe = np.hstack([X, np.array([tau_feats(r) for r in X])])
        Xh = np.hstack([X, np.array([extra_hidden(e['snaps'][str(T)]) for e in E])])
        la = logreg_cv(X, yy); na, acc1, dmin = nn_loo(X, yy)
        le = logreg_cv(Xe, yy); lh = logreg_cv(Xh, yy)
        jc = (A[yy == 1] == 1).mean(); jp = (A[yy == 0] == 1).mean()
        dc = (A[yy == 1] == 2).mean(); dp = (A[yy == 0] == 2).mean()
        print(f'tau<{T:>2}: n={len(yy)} ({yy.sum()} crash) | logreg AUC raw44 {la:.3f}  +engineered {le:.3f}  +hidden {lh:.3f} | '
              f'1-NN AUC {na:.3f}, crash-NN-is-crash {acc1:.2f} | model JUMP frac crash {jc:.2f} pass {jp:.2f}; DOWN crash {dc:.2f} pass {dp:.2f}')
        rows.append(dict(T=T, n=len(yy), crash=int(yy.sum()), auc_raw=la, auc_eng=le, auc_hidden=lh, nn_auc=na, nn_acc=acc1, jump_c=jc, jump_p=jp, down_c=dc, down_p=dp))
    return rows

hid = lambda s: [s['fs']['keyJump'], s['fs']['released'], s['fs']['speedDrop'], s['fs']['reachedMin'],
                 (s['fs']['req'] or {}).get('from', 0) if s['fs']['req'] else 0]

# Early: small cacti met at 400-750 points whose follower (at the snapshot) is a pterodactyl.
early = load('data/early*.jsonl', lambda e: e.get('type') == 'small' and 'snaps' in e and 400 <= e['spawnScore'] <= 750
             and any((s['fs'].get('next') or {}).get('t') == 'ptero' for s in e['snaps'].values()))
R1 = run('EARLY small cactus followed by a ptero (400-750 pts)', early, [40, 30, 20, 12], hid)

# Early, the subset where the ptero is a LOW one (y class 0 -> must be jumped)
# (ptero y class is in the features of obstacle slot 1 at the snapshot)
def low_follow(e):
    for s in e['snaps'].values():
        x = s['x']
        if x[18] and x[18 + 7] == 1 and x[18 + 9] == 1: return True
    return False
R1b = run('EARLY small cactus followed by a LOW ptero', [e for e in early if low_follow(e)], [40, 30, 20, 12], hid)

# Late: every high (y class 2) pterodactyl at 960k-1M; restrict to encounters where the
# dino is airborne at the tau<20 snapshot (the ptero becomes visible ~22 frames out).
late = load('data/high*.jsonl', lambda e: e.get('yClass') == 2 and 'snaps' in e and '20' in e['snaps'] and e['snaps']['20']['x'][2] == 1)
late_air = [e for e in late if e['snaps']['20']['x'][2] == 1]
n_air_all = len(late_air); n_air_crash = sum(e['out'] == 'crash' for e in late_air)
passes = [e for e in late_air if e['out'] == 'pass']
rng.shuffle(passes)
late_air = [e for e in late_air if e['out'] == 'crash'] + passes[:3000]   # subsample passes for 1-NN
print(f'late airborne high-ptero encounters: {n_air_all} ({n_air_crash} crash); analysed {len(late_air)}')
print(f'\nlate high-ptero encounters: {len(late)}, airborne at tau<20: {len(late_air)}, crashes among airborne: {sum(e["out"] == "crash" for e in late_air)}, crashes among grounded: {sum(e["out"] == "crash" for e in late if e not in late_air)}')
R2 = run('LATE high ptero, dino airborne at tau<20', late_air, [20, 12, 6], hid)

# Physical quantity: free-fall landing time vs ptero arrival at tau<20
print('\nLATE physical margin at tau<20 (tau_ptero - t_land if no input):')
for lab in ('crash', 'pass'):
    m = []
    for e in late_air:
        if e['out'] != lab: continue
        s = e['snaps']['20']; x = s['x']
        yv = 98 - x[0] * 100; vy = x[1] * 10
        m.append(s['tau'] - time_to_land(yv, vy))
    m = np.array(m)
    if len(m): print(f'  {lab}: n={len(m)} margin min {m.min():.2f} p10 {np.percentile(m,10):.2f} median {np.median(m):.2f} max {m.max():.2f}')
    vs = np.array([e['v'] for e in late_air if e['out'] == lab])
    if len(vs): print(f'     ptero v: min {vs.min():.3f} median {np.median(vs):.3f} max {vs.max():.3f}')

json.dump(dict(early=R1, early_low=R1b, late=R2), open('data/analyzeA.json', 'w'), indent=1)
