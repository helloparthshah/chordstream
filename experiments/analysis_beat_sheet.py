"""Chord-sheet accuracy: per-beat chords as the engine makes them (posteriors of the described
frames averaged over each causal-beat-tracker beat, argmax), scored per frame (majmin view and
full vocabulary). Uses posteriors cached by analysis_fixed_lag.py.

Modes: "pool" averages posteriors over each beat (the E16 engine); "vote" takes the majority of the
causal HMM path (the live readout) within each beat; "hmm" is the HMM path itself (no beats).

usage: python experiments/analysis_beat_sheet.py <cache> spec [spec ...]   (spec: models/x or models/x:sheet)
"""
import sys
from pathlib import Path

import numpy as np

from chordstream.beat import BeatTracker
from chordstream.evaluate import majmin_view
from chordstream.hmm import OnlineHMM
from chordstream.vocab import IGNORE
from chordstream.data.dataset import split_files

cache, specs = sys.argv[1], sys.argv[2:]
_, val = split_files(cache)
beats = []
for f in val:  # beat boundaries from the causal tracker on the song's features (as the engine)
    feats = np.load(f)["feats"].astype(np.float32)
    flags = np.array([b.beat for b in BeatTracker().run(feats)])
    beats.append(np.flatnonzero(flags) + 1)  # a flagged frame closes a beat: next beat starts after it


def tag(spec):
    b, _, h = spec.partition(":")
    return Path(b).name + (f"_{h}" if h else "")


for spec in specs:
    d = np.load(Path("data/analysis") / f"{tag(spec)}__{Path(cache).name}.npz", allow_pickle=True)
    hmm = OnlineHMM.load(Path(spec.partition(":")[0]) / "hmm.npz")
    hpaths = [hmm.filter(p) for p in d["probs"]]
    for mode in ("pool", "vote", "hmm"):
        res = {}
        for name, mm in (("majmin", True), ("full", False)):
            c = tot = 0
            for probs, hp, y, bnd in zip(d["probs"], hpaths, d["labels"], beats):
                edges = np.concatenate([[0], bnd[bnd < len(y)], [len(y)]])
                path = hp.copy() if mode == "hmm" else np.empty(len(y), int)
                if mode != "hmm":
                    for a, e in zip(edges[:-1], edges[1:]):
                        if e > a:
                            path[a:e] = (probs[a:e].sum(0).argmax() if mode == "pool"
                                         else np.bincount(hp[a:e]).argmax())
                p, yy = majmin_view(path, y) if mm else (path, y)
                m = yy != IGNORE
                c += int((p[m] == yy[m]).sum()); tot += int(m.sum())
            res[name] = round(c / tot, 4)
        print(f"{spec:32s} {mode:5s} majmin {res['majmin']:.4f}  full {res['full']:.4f}", flush=True)
