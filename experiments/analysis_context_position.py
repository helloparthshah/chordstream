"""F12: where does longer attention context help? Accuracy (majmin, live HMM path) by time into the song
and by first vs repeated passages, for models with different context (v10 5 s, v11 16 s, v12 32 s).

A frame is "repeated" when the chord progression around it (its chord and the 3 chords before it, majmin)
already occurred earlier in the song; otherwise "first". Billboard 36 + Bollywood 61 test songs.
usage: python experiments/analysis_context_position.py models/v10 models/v11 models/v12
"""
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

from chordstream.data.bollywood import trusted_ids_v2
from chordstream.data.dataset import split_files
from chordstream.export import load_net
from chordstream.hmm import OnlineHMM
from chordstream.train import predict_song
from chordstream.vocab import IGNORE, to_majmin

dev = torch.device("cpu")
_, val = split_files("data/cache_ext")
ok = trusted_ids_v2("align_cache_bolly3.jsonl")
bolly = [f for f in sorted(Path("data/cache_bolly3").glob("*.npz"))
         if zlib.crc32(f.stem.encode()) % 1000 < 300 and f.stem.split("__")[-1] in ok]
EDGES = [0, 5, 16, 32, 64, 1e9]  # seconds into the song
NAMES = ["0-5s", "5-16s", "16-32s", "32-64s", ">64s"]
FRAME = 0.04


def mm(a):
    return np.array([to_majmin(int(c)) if c >= 0 else IGNORE for c in a])


def repeated(y):
    """Per frame: has the 4-chord progression ending in this frame's chord occurred earlier in the song?"""
    rep = np.zeros(len(y), bool)
    seg_start = np.flatnonzero(np.r_[True, y[1:] != y[:-1]])
    chords = y[seg_start]
    seen = set()
    for i, s in enumerate(seg_start):
        e = seg_start[i + 1] if i + 1 < len(seg_start) else len(y)
        key = tuple(chords[max(0, i - 3): i + 1])
        rep[s:e] = len(key) == 4 and key in seen
        seen.add(key)
    return rep


songs = {}
for f in list(val) + bolly:
    d = np.load(f)
    songs[f] = (d["feats"].astype(np.float32), mm(d["labels"]))

for b in sys.argv[1:]:
    net, _ = load_net(Path(b)); net.eval(); hmm = OnlineHMM.load(Path(b) / "hmm.npz")
    for name, files in (("Billboard", val), ("Bollywood", bolly)):
        pos_hit, pos_n = np.zeros(len(NAMES)), np.zeros(len(NAMES))
        rep_hit, rep_n = np.zeros(2), np.zeros(2)
        for f in files:
            feats, y = songs[f]
            pa = mm(hmm.filter(predict_song(net, feats, dev)))
            n = min(len(pa), len(y)); pa, yy = pa[:n], y[:n]
            m = yy != IGNORE
            hit = (pa == yy) & m
            t = np.arange(n) * FRAME
            for k in range(len(NAMES)):
                sel = m & (t >= EDGES[k]) & (t < EDGES[k + 1])
                pos_hit[k] += hit[sel].sum(); pos_n[k] += sel.sum()
            r = repeated(yy)
            late = t >= 32  # compare first vs repeated only where every model has a full window
            for j, sel in enumerate((m & late & ~r, m & late & r)):
                rep_hit[j] += hit[sel].sum(); rep_n[j] += sel.sum()
        pos = "  ".join(f"{nm} {h / max(c, 1):.3f}" for nm, h, c in zip(NAMES, pos_hit, pos_n))
        print(f"{b} {name}: {pos}  | after 32 s: first {rep_hit[0] / rep_n[0]:.3f} ({rep_n[0] / rep_n.sum():.0%})"
              f"  repeated {rep_hit[1] / rep_n[1]:.3f}", flush=True)
