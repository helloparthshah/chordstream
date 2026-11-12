"""E21 label cleaning: which aligner changes make sheet-derived labels more accurate? Measured on the
36 held-out Billboard songs (true timings known); the "sheet" is each song's chord sequence.

Variants: bass evidence (weight), beat snapping of chord changes, and a per-frame confidence mask
(accuracy of the frames kept vs share kept). Key given (shift 0) so only timing quality is measured.

usage: python experiments/analysis_label_cleaning.py
"""
import numpy as np

from chordstream.beat import BeatTracker
from chordstream.data.dataset import split_files
from chordstream.data.sheet_align import align_ordered, frame_confidence, snap_to_beats
from chordstream.evaluate import majmin_view
from chordstream.vocab import IGNORE

_, val = split_files("data/cache_ext")
songs = []
for f in val:
    d = np.load(f)
    feats, y = d["feats"].astype(np.float32), d["labels"].astype(np.int64)
    seq = [int(c) for i, c in enumerate(y) if c != IGNORE and (i == 0 or c != y[i - 1])]
    beats = np.flatnonzero([b.beat for b in BeatTracker().run(feats)]) + 1
    songs.append((feats, y, seq, beats))


def score(pairs):
    fc = fn = mc = mn = 0
    for lab, y, keep in pairs:
        m = (y != IGNORE) & keep
        fc += (lab[m] == y[m]).sum(); fn += m.sum()
        p, yy = majmin_view(lab, y); m2 = (yy != IGNORE) & keep
        mc += (p[m2] == yy[m2]).sum(); mn += m2.sum()
    return fc / fn, mc / mn


for bw in (0.0, 0.5, 1.0):
    res = [(align_ordered(feats, seq, shifts=[0], bass_weight=bw), y, beats) for feats, y, seq, beats in songs]
    full = [np.ones(len(y), bool) for _, y, _ in res]
    print(f"bass {bw}: plain full/majmin {score([(r['labels'], y, k) for (r, y, _), k in zip(res, full)])}", flush=True)
    for rad in (4, 8):
        snapped = [snap_to_beats(r["labels"], b, rad) for r, _, b in res]
        print(f"  + beat snap {rad}: {score([(s, y, k) for s, (_, y, _), k in zip(snapped, res, full)])}", flush=True)
    conf = [frame_confidence(r["labels"], r["cos"], r["states"]) for r, _, _ in res]
    allc = np.concatenate(conf)
    for keep_share in (0.9, 0.8, 0.7, 0.6):
        thr = np.quantile(allc, 1 - keep_share)
        print(f"  + keep top {keep_share:.0%} confidence: {score([(r['labels'], y, c >= thr) for (r, y, _), c in zip(res, conf)])}",
              flush=True)
