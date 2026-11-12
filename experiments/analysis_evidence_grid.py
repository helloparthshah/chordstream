"""F05b: wider grid for the evidence hold (see analysis_flicker_fixes.py), plain input.

usage: python experiments/analysis_evidence_grid.py models/v9
"""
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from analysis_flicker_fixes import evidence, stats  # noqa: E402
from chordstream.data.bollywood import trusted_ids_v2  # noqa: E402
from chordstream.data.dataset import split_files  # noqa: E402
from chordstream.export import load_net  # noqa: E402
from chordstream.hmm import OnlineHMM  # noqa: E402
from chordstream.train import predict_song  # noqa: E402

b = sys.argv[1]
dev = "cuda" if torch.cuda.is_available() else "cpu"
net, _ = load_net(Path(b)); net.to(dev); hmm = OnlineHMM.load(Path(b) / "hmm.npz")
_, val = split_files("data/cache_ext")
ok = trusted_ids_v2("align_cache_bolly3.jsonl")
sets = {"Billboard": val, "Bollywood": [f for f in sorted(Path("data/cache_bolly3").glob("*.npz"))
                                        if zlib.crc32(f.stem.encode()) % 1000 < 300 and f.stem.split("__")[-1] in ok]}
songs = {}
for n, files in sets.items():
    songs[n] = []
    for f in files:
        d = np.load(f); pr = predict_song(net, d["feats"].astype(np.float32), dev)
        songs[n].append((pr, hmm.filter(pr), d["labels"].astype(np.int64)))
for k in (12, 20, 30):
    for t in (0.01, 0.02, 0.05, 0.1):
        print(f"k{k:2d} t{t:<5}" + "  |  ".join(f"{n}: {stats([(evidence(pa, pr, k, t), y) for pr, pa, y in songs[n]])}" for n in songs), flush=True)
