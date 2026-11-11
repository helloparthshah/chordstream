"""E23/E24 fairness checks: fixed-lag decoding for every model, and the same HMM grid for every model.

usage: python experiments/analysis_fixed_lag.py <cache> <out.json> bundle [bundle ...]
       (bundle:sheet scores an E26 model's sheet head)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from chordstream.data.dataset import split_files
from chordstream.evaluate import ORIGINAL_VAL, change_points, match_f, majmin_view
from chordstream.export import load_net
from chordstream.hmm import OnlineHMM
from chordstream.train import predict_song
from chordstream.vocab import IGNORE

FS = 0.04
cache, out_path, bundles = sys.argv[1], sys.argv[2], sys.argv[3:]
dev = "cuda" if torch.cuda.is_available() else "cpu"
_, val = split_files(cache)


def posteriors(spec):
    bundle, _, head = spec.partition(":")
    head = head or "chord"
    tag = Path(bundle).name + ("" if head == "chord" else f"_{head}")
    p = Path("data/analysis") / f"{tag}__{Path(cache).name}.npz"
    if p.exists():
        d = np.load(p, allow_pickle=True)
        return list(d["probs"]), list(d["labels"]), list(d["ids"])
    net, _ = load_net(Path(bundle))
    net.to(dev)
    probs, labels, ids = [], [], []
    for f in val:
        d = np.load(f)
        probs.append(predict_song(net, d["feats"].astype(np.float32), dev, head=head).astype(np.float32))
        labels.append(d["labels"].astype(np.int64))
        ids.append(f.stem.split("__")[-1])
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, probs=np.array(probs, dtype=object), labels=np.array(labels, dtype=object), ids=np.array(ids))
    return probs, labels, ids


def decode(hmm: OnlineHMM, probs: np.ndarray, lag: int) -> np.ndarray:
    """Forward filter, then (lag > 0) fixed-lag smoothing: frame s is decided at s + lag using
    frames s+1..s+lag. lag 0 = the live zero-lookahead HMM."""
    lik = (np.clip(probs.astype(np.float64), 1e-8, 1) / hmm.prior) ** hmm.power
    T, n = lik.shape
    alpha = np.empty((T, n))
    a = np.zeros(n)
    a[0] = 1.0  # NO_CHORD start, as OnlineHMM.reset
    for t in range(T):
        a = (a @ hmm.A) * lik[t]
        a /= a.sum()
        alpha[t] = a
    if lag == 0:
        return alpha.argmax(1)
    b = np.ones((T, n))
    for j in range(lag, 0, -1):
        idx = np.minimum(np.arange(T) + j, T - 1)
        b = (lik[idx] * b) @ hmm.A.T
        b /= b.sum(1, keepdims=True)
    return (alpha * b).argmax(1)


def score(paths, labels, ids, subset=None, majmin=False):
    c = tot = npred = nref = 0
    fs = []
    for p, y, i in zip(paths, labels, ids):
        if subset and i not in subset:
            continue
        if majmin:
            p, y = majmin_view(p, y)
        v = y != IGNORE
        c += int((p[v] == y[v]).sum()); tot += int(v.sum())
        pc, rc = change_points(p), change_points(y)
        npred += len(pc); nref += len(rc)
        fs.append(match_f(pc, rc, 0.25)[0])
    return {"acc": round(c / tot, 4), "changes": round(npred / nref, 2), "change_f": round(float(np.mean(fs)), 3)}


def hmm_variant(base: OnlineHMM, seconds: float, power: float) -> OnlineHMM:
    p_stay = 1 - FS / seconds
    A = p_stay * np.eye(base.n) + (1 - p_stay) * base.C
    return OnlineHMM(A, base.prior, power)


results = {}
for b in bundles:
    probs, labels, ids = posteriors(b)
    ext = probs[0].shape[1] > 25
    base = OnlineHMM.load(Path(b.partition(":")[0]) / "hmm.npz")
    r = {"bundle_hmm": {"seconds": round(FS / (1 - base.p_stay), 2), "power": base.power}}
    # bundle HMM at lag 0/3/6
    for lag in (0, 3, 6):
        paths = [decode(base, p, lag) for p in probs]
        r[f"lag{lag}"] = {"all": score(paths, labels, ids, majmin=ext),
                          "orig10": score(paths, labels, ids, set(ORIGINAL_VAL), majmin=ext)}
        if ext:
            r[f"lag{lag}"]["full"] = score(paths, labels, ids)
        print(b, f"lag{lag}", r[f"lag{lag}"], flush=True)
    # HMM grid at lag 0
    grid = []
    for sec in (1.0, 2.0, 3.0, 5.0, 8.0):
        for pw in (1.0, 2.0, 3.0):
            h = hmm_variant(base, sec, pw)
            paths = [decode(h, p, 0) for p in probs]
            row = {"seconds": sec, "power": pw, **score(paths, labels, ids, majmin=ext)}
            if ext:
                row["full_acc"] = score(paths, labels, ids)["acc"]
            grid.append(row)
            print(b, row, flush=True)
    r["grid"] = grid
    results[b] = r
Path(out_path).write_text(json.dumps(results, indent=1))
