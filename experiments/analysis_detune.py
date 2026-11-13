"""F11: accuracy when the whole recording is detuned by c cents (CQT reference shifted by -c).

usage: python experiments/analysis_detune.py [--billboard-only] models/v11 [models/x ...]
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
from chordstream.vocab import IGNORE

dev = "cuda" if torch.cuda.is_available() else "cpu"
_, val = split_files("data/cache_ext"); ok = trusted_ids_v2("align_cache_bolly3.jsonl")
bolly = [f for f in sorted(Path("data/cache_bolly3").glob("*.npz"))
         if zlib.crc32(f.stem.encode()) % 1000 < 300 and f.stem.split("__")[-1] in ok]
CENTS = (-45, -30, -15, 0, 15, 30, 45)
only_bb = "--billboard-only" in sys.argv
for b in [a for a in sys.argv[1:] if not a.startswith("--")]:
    net, _ = load_net(Path(b)); net.to(dev).eval(); hmm = OnlineHMM.load(Path(b) / "hmm.npz")
    fc = BundleConfig.load(Path(b) / "config.json").frontend  # the model's own CQT (24 or 36 bins/octave)
    for name, files in (("Billboard", val),) + (() if only_bb else (("Bollywood", bolly),)):
        c = np.zeros(len(CENTS)); n = np.zeros(len(CENTS))
        for f in files:
            d = np.load(f); y = d["labels"].astype(np.int64)
            audio = np.load(Path("data/raw16k") / f"{f.stem}.npy").astype(np.float32) / 32767
            for j, ct in enumerate(CENTS):
                x = d["feats"].astype(np.float32) if ct == 0 and d["feats"].shape[1] == fc.n_bins else offline_cqt(audio, dataclasses.replace(fc, fmin=fc.fmin * 2 ** (-ct / 1200)), dev)
                pr = predict_song(net, x, dev); k = min(len(pr), len(y))
                p2, y2 = majmin_view(hmm.filter(pr[:k]), y[:k]); m = y2 != IGNORE
                c[j] += (p2[m] == y2[m]).sum(); n[j] += m.sum()
        print(f"{b} {name}: " + "  ".join(f"{ct:+d}c {c[j] / n[j]:.3f}" for j, ct in enumerate(CENTS)), flush=True)
