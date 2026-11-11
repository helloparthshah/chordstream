"""Speculative display simulation: a zero-delay draft model + a label-delay verifier.

At frame t the verifier has settled frame t - d. The displayed chord is the draft's chord when the
draft is confident (HMM posterior >= tau), otherwise the verifier's latest settled chord.
Scores the *displayed* sequence against the labels at display time (what the user sees), plus
how often a shown chord is later contradicted by the verifier ("corrections").

usage: python experiments/analysis_speculative.py data/cache_bb_fix models/v4_clean models/e23b_delay6
(run analysis_fixed_lag.py first; it caches posteriors in data/analysis)
"""
import sys
from pathlib import Path

import numpy as np

from chordstream.evaluate import change_points, match_f
from chordstream.hmm import OnlineHMM
from chordstream.vocab import IGNORE

cache, draft_b, verify_b = sys.argv[1:4]


def load(b):
    d = np.load(Path("data/analysis") / f"{Path(b).name}__{Path(cache).name}.npz", allow_pickle=True)
    return list(d["probs"]), list(d["labels"])


def filt(hmm, probs):
    lik = (np.clip(probs.astype(np.float64), 1e-8, 1) / hmm.prior) ** hmm.power
    a = np.zeros(hmm.n); a[0] = 1
    path, conf = np.empty(len(lik), int), np.empty(len(lik))
    for t in range(len(lik)):
        a = (a @ hmm.A) * lik[t]; a /= a.sum()
        path[t], conf[t] = a.argmax(), a.max()
    return path, conf


import json  # noqa: E402

dp, labels = load(draft_b)
vp, _ = load(verify_b)
dh, vh = OnlineHMM.load(Path(draft_b) / "hmm.npz"), OnlineHMM.load(Path(verify_b) / "hmm.npz")
d = json.loads((Path(verify_b) / "config.json").read_text())["model"]["label_delay"]
D = [filt(dh, p) for p in dp]
V = [filt(vh, p)[0] for p in vp]  # V[s] = verifier's chord for frame s, known at time s + d


def run(tau):
    c = tot = npred = nref = corr = shown = 0
    fs = []
    for (dpath, dconf), v, y in zip(D, V, labels):
        T = len(y)
        settled = np.concatenate([np.zeros(d, int), v[:T - d]])  # verifier's chord available at t
        x = np.where(dconf >= tau, dpath, settled) if tau <= 1 else settled
        if tau > 1:  # pure verifier, shown late: scored at display time
            pass
        m = y != IGNORE
        c += int((x[m] == y[m]).sum()); tot += int(m.sum())
        pc, rc = change_points(x), change_points(y)
        npred += len(pc); nref += len(rc)
        fs.append(match_f(pc, rc, 0.25)[0])
        # corrections: displayed chord at t later contradicted by the verifier's verdict on frame t
        corr += int((x != v[:T]).sum()); shown += T
    return {"tau": tau, "acc_at_display": round(c / tot, 4), "changes": round(npred / nref, 2),
            "change_f": round(float(np.mean(fs)), 3), "contradicted": round(corr / shown, 3)}


rows = [run(0.0)] + [run(t) for t in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)] + [run(2.0)]
print(f"draft {draft_b}  verifier {verify_b} (delay {d} frames)")
print("tau 0 = draft only; tau 2 = verifier only (shown d frames late)")
for r in rows:
    print(r)
# reference: verifier scored at the frames it describes (the chord sheet / history view)
c = tot = 0
for v, y in zip(V, labels):
    m = y != IGNORE
    c += int((v[m] == y[m]).sum()); tot += int(m.sum())
print("verifier at described frames (history / sheet):", round(c / tot, 4))


# Accept rule 2: persistence. Show the verifier's settled chord, but switch early to a draft chord
# once the draft has held it for k consecutive frames (k < d): a draft "token" is accepted when
# it is stable, and the verifier overrides anything it later contradicts.
def run_persist(k, raw_tau=None):
    c = tot = npred = nref = 0
    fs = []
    for (dpath, dconf), v, y, pr in zip(D, V, labels, dp):
        T = len(y)
        settled = np.concatenate([np.zeros(d, int), v[:T - d]])
        x = settled.copy()
        run_len = 1
        for t in range(T):
            if t and dpath[t] == dpath[t - 1]:
                run_len += 1
            else:
                run_len = 1
            ok = run_len >= k
            if raw_tau is not None:
                ok = ok and pr[t][dpath[t]] >= raw_tau
            if ok:
                x[t] = dpath[t]
        m = y != IGNORE
        c += int((x[m] == y[m]).sum()); tot += int(m.sum())
        pc, rc = change_points(x), change_points(y)
        npred += len(pc); nref += len(rc)
        fs.append(match_f(pc, rc, 0.25)[0])
    return {"acc_at_display": round(c / tot, 4), "changes": round(npred / nref, 2), "change_f": round(float(np.mean(fs)), 3)}


for k in range(1, d + 1):
    print("persist", k, run_persist(k))
for rt in (0.5, 0.7, 0.9):
    print("raw prob >=", rt, run_persist(1, rt))
