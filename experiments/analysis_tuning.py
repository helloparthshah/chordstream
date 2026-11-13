"""F11: does retuning the CQT to each recording's measured tuning help?

Live simulation: the first `warm` seconds use the standard CQT; tuning is estimated from those seconds
only, then the rest of the song is analysed with fmin * 2^(cents/1200). Compared with the cached
(untuned) features on Billboard 36 + Bollywood 61, overall and on the songs > 10 cents off.
usage: python experiments/analysis_tuning.py models/v11 [warm_seconds=10]
"""
import dataclasses
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

from chordstream.config import BundleConfig
from chordstream.cqt import offline_cqt
from chordstream.data.bollywood import trusted_ids_v2
from chordstream.data.dataset import split_files
from chordstream.evaluate import majmin_view
from chordstream.export import load_net
from chordstream.hmm import OnlineHMM
from chordstream.train import predict_song
from chordstream.tuning import estimate_tuning
from chordstream.vocab import IGNORE

b = Path(sys.argv[1]); warm = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
dev = "cuda" if torch.cuda.is_available() else "cpu"
net, _ = load_net(b); net.to(dev).eval(); hmm = OnlineHMM.load(b / "hmm.npz")
fcfg = BundleConfig.load(b / "config.json").frontend
_, val = split_files("data/cache_ext"); ok = trusted_ids_v2("align_cache_bolly3.jsonl")
bolly = [f for f in sorted(Path("data/cache_bolly3").glob("*.npz"))
         if zlib.crc32(f.stem.encode()) % 1000 < 300 and f.stem.split("__")[-1] in ok]


def score(feats, y):
    pr = predict_song(net, feats.astype(np.float32), dev); n = min(len(pr), len(y))
    p2, y2 = majmin_view(hmm.filter(pr[:n]), y[:n]); m = y2 != IGNORE
    return (p2[m] == y2[m]).sum(), m.sum()


for name, files in (("Billboard", val), ("Bollywood", bolly)):
    tot = np.zeros((2, 2)); det = np.zeros((2, 2)); nd = 0; rows = []
    for f in files:
        d = np.load(f); y = d["labels"].astype(np.int64)
        audio = np.load(Path("data/raw16k") / f"{f.stem}.npy").astype(np.float32) / 32767
        cents = estimate_tuning(audio[: int(warm * 16000)])
        base = d["feats"].astype(np.float32)
        tuned_cfg = dataclasses.replace(fcfg, fmin=fcfg.fmin * 2 ** (cents / 1200))
        tuned = offline_cqt(audio, tuned_cfg, dev)[: len(base)]
        k = int(warm / fcfg.frame_seconds)
        tuned = np.concatenate([base[:k], tuned[k:]]) if len(tuned) == len(base) else base
        r0, r1 = score(base, y), score(tuned, y)
        tot += [r0, r1]
        if abs(cents) > 10:
            det += [r0, r1]; nd += 1; rows.append(f"{f.stem.split('__')[-1][:18]:18s} {cents:+5.1f}c  {r0[0] / r0[1]:.3f} -> {r1[0] / r1[1]:.3f}")
    print(f"{name}: all {tot[0, 0] / tot[0, 1]:.4f} -> {tot[1, 0] / tot[1, 1]:.4f}   "
          f"songs >10 cents off ({nd}): {det[0, 0] / max(det[0, 1], 1):.3f} -> {det[1, 0] / max(det[1, 1], 1):.3f}", flush=True)
    print("   " + "\n   ".join(rows), flush=True)
