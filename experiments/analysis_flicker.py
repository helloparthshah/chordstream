"""F04: flicker anatomy of a bundle's live readout (HMM path) on both test sets.

Per test set: changes per minute (prediction vs reference), share of predicted segments shorter than
0.2 / 0.5 s (reference for comparison), "blips" (A -> B -> A with B shorter than 0.5 s), how many
blips are wrong (B not the reference chord), and the most common blip pairs (majmin view).

usage: python experiments/analysis_flicker.py models/v9
"""
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from chordstream.data.bollywood import trusted_ids_v2
from chordstream.data.dataset import split_files
from chordstream.export import load_net
from chordstream.hmm import OnlineHMM
from chordstream.train import predict_song
from chordstream.vocab import CLASS_NAMES, IGNORE, to_majmin

FS = 0.04


def runs(p):
    b = np.flatnonzero(np.diff(p)) + 1
    st = np.r_[0, b]; en = np.r_[b, len(p)]
    return [(int(p[s]), s, e) for s, e in zip(st, en)]


def anatomy(name, pairs):
    tot = Counter(); bl = Counter()
    for path, y in pairs:
        pm = np.array([to_majmin(c) for c in path]); ym = np.array([to_majmin(c) if c != IGNORE else IGNORE for c in y])
        tot["min"] += len(path) * FS / 60
        rp, rr = runs(pm), runs(ym[ym != IGNORE]) if (ym != IGNORE).any() else []
        tot["pc"] += len(rp) - 1; tot["rc"] += max(len(rr) - 1, 0)
        for c, s, e in rp:
            tot["seg"] += 1; tot["s02"] += (e - s) * FS < 0.2; tot["s05"] += (e - s) * FS < 0.5
        for c, s, e in rr:
            tot["rseg"] += 1; tot["rs05"] += (e - s) * FS < 0.5
        for (a, _, _), (b, s, e), (c, _, _) in zip(rp, rp[1:], rp[2:]):
            if a == c and (e - s) * FS < 0.5:
                tot["blip"] += 1
                ref = ym[s:e]; ref = ref[ref != IGNORE]
                right = len(ref) and (ref == b).mean() > 0.5
                tot["blip_right"] += bool(right)
                if not right:
                    bl[(CLASS_NAMES[a], CLASS_NAMES[b])] += 1
    print(f"{name}: {tot['pc'] / tot['min']:.1f} changes/min (ref {tot['rc'] / tot['min']:.1f}); "
          f"segments <0.2 s {tot['s02'] / tot['seg']:.1%}, <0.5 s {tot['s05'] / tot['seg']:.1%} (ref <0.5 s {tot['rs05'] / tot['rseg']:.1%}); "
          f"blips {tot['blip'] / tot['min']:.1f}/min, {1 - tot['blip_right'] / max(tot['blip'], 1):.0%} wrong; "
          f"blips are {tot['blip'] / max(tot['pc'], 1) * 2:.0%} of changes")
    print("   top wrong blips:", ", ".join(f"{a}>{b} {n}" for (a, b), n in bl.most_common(8)))


def main(b):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, _ = load_net(Path(b)); net.to(dev); hmm = OnlineHMM.load(Path(b) / "hmm.npz")
    _, val = split_files("data/cache_ext")
    bb = []
    for f in val:
        d = np.load(f); bb.append((hmm.filter(predict_song(net, d["feats"].astype(np.float32), dev)), d["labels"].astype(np.int64)))
    anatomy("Billboard (36)", bb)
    ok = trusted_ids_v2("align_cache_bolly3.jsonl"); bo = []
    for f in sorted(Path("data/cache_bolly3").glob("*.npz")):
        if zlib.crc32(f.stem.encode()) % 1000 < 300 and f.stem.split("__")[-1] in ok:
            d = np.load(f)
            y = d["labels_full"] if "labels_full" in d.files else d["labels"]
            bo.append((hmm.filter(predict_song(net, d["feats"].astype(np.float32), dev)), y.astype(np.int64)))
    anatomy("Bollywood (61)", bo)


if __name__ == "__main__":
    for b in sys.argv[1:]:
        print("==", b); main(b)
