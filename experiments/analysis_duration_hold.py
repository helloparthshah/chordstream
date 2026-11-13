"""F10: duration-aware Steady display (no retraining).

Evidence hold (deployed, F05): leave shown chord A for the live chord B at once if p(A) < thr, else
once B has held k frames. Duration-aware: k grows with how long A has been shown,
k = k0 + slope * min(shown_frames, cap), so a fresh (possibly wrong) chord is easy to replace and a
long-established one needs B to persist longer. Scored on Billboard 36 + Bollywood 61.
usage: python experiments/analysis_duration_hold.py models/v11
"""
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analysis_flicker_fixes import stats  # noqa: E402

from chordstream.data.bollywood import trusted_ids_v2
from chordstream.data.dataset import split_files
from chordstream.export import load_net
from chordstream.hmm import OnlineHMM
from chordstream.train import predict_song


def duration_hold(path, probs, k0, slope, thr, cap=125):
    out = np.empty_like(path); shown, run, age = path[0], 0, 0
    for t, c in enumerate(path):
        run = run + 1 if t and c == path[t - 1] else 1
        age += 1
        k = k0 + slope * min(age, cap)
        if c != shown and (probs[t, shown] < thr or run >= k):
            shown, age = c, 0
        out[t] = shown
    return out


b = Path(sys.argv[1]); net, _ = load_net(b); net.eval(); hmm = OnlineHMM.load(b / "hmm.npz")
dev = "cuda" if __import__("torch").cuda.is_available() else "cpu"; net.to(dev)
_, val = split_files("data/cache_ext")
ok = trusted_ids_v2("align_cache_bolly3.jsonl")
bolly = [f for f in sorted(Path("data/cache_bolly3").glob("*.npz"))
         if zlib.crc32(f.stem.encode()) % 1000 < 300 and f.stem.split("__")[-1] in ok]
sets = {}
for name, files in (("Billboard", val), ("Bollywood", bolly)):
    sets[name] = []
    for f in files:
        d = np.load(f); y = d["labels"].astype(np.int64)
        pr = predict_song(net, d["feats"].astype(np.float32), dev); n = min(len(pr), len(y))
        sets[name].append((pr[:n], hmm.filter(pr[:n]), y[:n]))
grid = [("live", None)] + [(f"k0={k0:2d} slope={s:<4} thr={t}", (k0, s, t))
                           for t in (0.05, 0.2) for k0 in (6, 10, 20) for s in (0, 0.05, 0.1, 0.2)]
for name, g in grid:
    row = []
    for n, songs in sets.items():
        disp = [(pa if g is None else duration_hold(pa, pr, *g), y) for pr, pa, y in songs]
        row.append(f"{n}: {stats(disp)}")
    print(f"{name:28s} " + "  |  ".join(row), flush=True)
