"""Run after analysis_fixed_lag.py (uses its cached posteriors in data/analysis).
Paired per-song comparison at equal delay: model A decoded with fixed lag LA vs model B with lag LB."""
import sys
from pathlib import Path

import numpy as np

from chordstream.hmm import OnlineHMM
from chordstream.vocab import IGNORE


def decode(hmm, probs, lag):
    lik = (np.clip(probs.astype(np.float64), 1e-8, 1) / hmm.prior) ** hmm.power
    T, n = lik.shape
    alpha = np.empty((T, n)); a = np.zeros(n); a[0] = 1
    for t in range(T):
        a = (a @ hmm.A) * lik[t]; a /= a.sum(); alpha[t] = a
    if lag == 0:
        return alpha.argmax(1)
    b = np.ones((T, n))
    for j in range(lag, 0, -1):
        idx = np.minimum(np.arange(T) + j, T - 1)
        b = (lik[idx] * b) @ hmm.A.T; b /= b.sum(1, keepdims=True)
    return (alpha * b).argmax(1)


def per_song(bundle, cache, lag):
    d = np.load(Path("data/analysis") / f"{Path(bundle).name}__{Path(cache).name}.npz", allow_pickle=True)
    hmm = OnlineHMM.load(Path(bundle) / "hmm.npz")
    out = []
    for p, y in zip(d["probs"], d["labels"]):
        v = y != IGNORE
        out.append((decode(hmm, p, lag)[v] == y[v]).mean())
    return np.array(out)


cache = sys.argv[1]
for a, la, b, lb in [("models/e23a_delay3", 0, "models/v4_clean", 3), ("models/e23b_delay6", 0, "models/v4_clean", 6)]:
    x, y = per_song(a, cache, la), per_song(b, cache, lb)
    d = x - y
    rng = np.random.default_rng(0)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    from math import comb
    k, n = int((d > 0).sum()), int((d != 0).sum())
    p = 2 * sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n
    print(f"{a} vs {b}@lag{lb}: mean {d.mean()*100:+.2f} pt  CI [{np.percentile(boots,2.5)*100:+.2f}, "
          f"{np.percentile(boots,97.5)*100:+.2f}]  better {k}/{n}  sign p={min(1,p):.4f}")
