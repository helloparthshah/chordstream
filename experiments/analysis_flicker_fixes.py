"""F05: display-side flicker fixes, scored at display time on both test sets (no retraining).

Displays (all causal on the live HMM path; scored against the labels of the frames they describe,
so any extra waiting shows up as lag):
  live          the HMM path (Instant mode)
  steady k      the big chord in Steady mode: switches once a chord has held k frames (not backdated)
  evidence k,t  music-aware hold: leave the shown chord A for B at once, unless the network still
                gives A probability >= t, in which case B must hold k frames first
Inputs: plain, or pitch-shift averaging (TTA): posteriors for the CQT shifted by s semitones,
mapped back and averaged (arithmetic) before the HMM.

usage: python experiments/analysis_flicker_fixes.py models/v9
"""
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

from chordstream.data.bollywood import trusted_ids_v2
from chordstream.data.dataset import shift_bins, split_files
from chordstream.evaluate import majmin_view
from chordstream.export import load_net
from chordstream.hmm import OnlineHMM
from chordstream.train import predict_song
from chordstream.vocab import CLASS_NAMES, IGNORE, to_majmin, transpose

FS = 0.04
NC = len(CLASS_NAMES)


def tta_probs(net, feats, dev, shifts, bps):
    acc = 0
    for s in shifts:
        p = predict_song(net, shift_bins(feats, s * bps), dev)
        # posteriors for the audio transposed up by s: class c there is class transpose(c, -s) here
        back = np.zeros_like(p)
        back[:, [transpose(c, -s) for c in range(NC)]] = p
        acc = acc + back
    return acc / len(shifts)


def steady(path, k):
    out = np.empty_like(path); shown, run = path[0], 0
    for t, c in enumerate(path):
        run = run + 1 if t and c == path[t - 1] else 1
        if run >= k:
            shown = c
        out[t] = shown
    return out


def evidence(path, probs, k, thr):
    out = np.empty_like(path); shown, run = path[0], 0
    for t, c in enumerate(path):
        run = run + 1 if t and c == path[t - 1] else 1
        if c != shown and (probs[t, shown] < thr or run >= k):
            shown = c
        out[t] = shown
    return out


def stats(pairs):
    mc = mn = fc = fn = chg = blip = short = seg = 0; mins = 0.0
    for disp, y in pairs:
        p2, y2 = majmin_view(disp, y); m = y2 != IGNORE
        mc += (p2[m] == y2[m]).sum(); mn += m.sum()
        m = y != IGNORE; fc += (disp[m] == y[m]).sum(); fn += m.sum()
        pm = np.array([to_majmin(c) for c in disp]); mins += len(disp) * FS / 60
        b = np.r_[0, np.flatnonzero(np.diff(pm)) + 1, len(pm)]
        lens = np.diff(b); vals = pm[b[:-1]]
        chg += len(lens) - 1; seg += len(lens); short += (lens * FS < 0.5).sum()
        blip += sum(1 for i in range(1, len(lens) - 1) if vals[i - 1] == vals[i + 1] and lens[i] * FS < 0.5)
    return f"majmin {mc / mn:.3f} full {fc / fn:.3f}  {chg / mins:5.1f} ch/min  {blip / mins:4.1f} blips/min  <0.5s {short / seg:.0%}"


def main(b):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, _ = load_net(Path(b)); net.to(dev); hmm = OnlineHMM.load(Path(b) / "hmm.npz")
    bps = 2
    _, val = split_files("data/cache_ext")
    sets = {"Billboard": [(np.load(f), "labels") for f in val]}
    ok = trusted_ids_v2("align_cache_bolly3.jsonl")
    sets["Bollywood"] = [(np.load(f), "labels") for f in sorted(Path("data/cache_bolly3").glob("*.npz"))
                         if zlib.crc32(f.stem.encode()) % 1000 < 300 and f.stem.split("__")[-1] in ok]
    for tta in ([0], [-1, 0, 1], [-2, -1, 0, 1, 2]):
        songs = {n: [] for n in sets}
        for n, items in sets.items():
            for d, key in items:
                pr = tta_probs(net, d["feats"].astype(np.float32), dev, tta, bps)
                songs[n].append((pr, hmm.filter(pr), d[key].astype(np.int64)))
        print(f"-- input shifts {tta}", flush=True)
        variants = [("live", lambda pr, pa: pa)]
        variants += [(f"steady {k}", lambda pr, pa, k=k: steady(pa, k)) for k in (6, 12)]
        variants += [(f"evidence k{k} t{t}", lambda pr, pa, k=k, t=t: evidence(pa, pr, k, t))
                     for k in (6, 12) for t in (0.05, 0.2)]
        for name, fn in variants:
            print(f"   {name:18s}" + "  |  ".join(f"{n}: {stats([(fn(pr, pa), y) for pr, pa, y in songs[n]])}" for n in songs), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
