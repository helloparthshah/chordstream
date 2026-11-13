"""E37 pilot: the Billboard extended-label cache (data/cache_ext) recomputed at 36 bins per octave
(3 per semitone, 216 bins, C2..B7) from the cached 16 kHz audio. Labels and targets are copied.

usage: python experiments/build_cqt36_cache.py [out=data/cache_ext36]
"""
import sys
from pathlib import Path

import numpy as np
import torch

from chordstream.config import FrontendConfig
from chordstream.cqt import offline_cqt

out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/cache_ext36"); out.mkdir(parents=True, exist_ok=True)
cfg = FrontendConfig(bins_per_octave=36, n_bins=216); dev = "cuda" if torch.cuda.is_available() else "cpu"
files = sorted(Path("data/cache_ext").glob("*.npz")); print(len(files), "songs", flush=True)
for i, f in enumerate(files):
    dst = out / f.name
    raw = Path("data/raw16k") / f"{f.stem}.npy"
    if dst.exists() or not raw.exists():
        if not raw.exists():
            print("no raw audio", f.stem, flush=True)
        continue
    feats = offline_cqt(np.load(raw).astype(np.float32) / 32767.0, cfg, dev)
    d = np.load(f); n = min(len(feats), len(d["labels"]))
    np.savez_compressed(dst, feats=feats[:n].astype(np.float16), **{k: d[k][:n] for k in d.files if k != "feats"})
    if i % 50 == 0:
        print(i, f.stem, flush=True)
print("done", flush=True)
