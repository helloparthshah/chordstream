"""F09: HMM stickiness vs accuracy, re-tuned for flicker on the deployed model.

Grid over the mean chord duration (self-transition) and emission power; scored on the live HMM path
and the Steady display (evidence hold k20 t0.05), Billboard 36 + Bollywood 61 pooled per set.
usage: python experiments/analysis_hmm_grid.py models/v10
"""
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analysis_flicker_fixes import evidence, stats  # noqa: E402

from chordstream.data.bollywood import trusted_ids_v2
from chordstream.data.dataset import split_files
from chordstream.export import load_net
from chordstream.hmm import OnlineHMM
from chordstream.train import predict_song

b = Path(sys.argv[1]); net, _ = load_net(b); net.eval(); base = OnlineHMM.load(b / "hmm.npz")
_, val = split_files("data/cache_ext")
ok = trusted_ids_v2("align_cache_bolly3.jsonl")
bolly = [f for f in sorted(Path("data/cache_bolly3").glob("*.npz"))
         if zlib.crc32(f.stem.encode()) % 1000 < 300 and f.stem.split("__")[-1] in ok]
sets = {}
for name, files in (("Billboard", val), ("Bollywood", bolly)):
    sets[name] = []
    for f in files:
        d = np.load(f); y = d["labels"].astype(np.int64)
        pr = predict_song(net, d["feats"].astype(np.float32), "cpu"); n = min(len(pr), len(y))
        sets[name].append((pr[:n], y[:n]))
for secs in (3, 5, 8, 12, 20):
    for power in (1.0, 1.5, 2.0, 3.0):
        p = 1 - 0.04 / secs
        hmm = OnlineHMM(p * np.eye(base.n) + (1 - p) * base.C, base.prior, power)
        for name, songs in sets.items():
            paths = [(hmm.filter(pr), pr, y) for pr, y in songs]
            print(f"{secs:3d}s p{power:<4} {name:9s} live {stats([(pa, y) for pa, pr, y in paths])}"
                  f"\n{'':24s}steady {stats([(evidence(pa, pr, 20, 0.05), y) for pa, pr, y in paths])}", flush=True)
