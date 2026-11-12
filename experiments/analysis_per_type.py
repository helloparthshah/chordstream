"""Per-chord-type recall on the 36 clean held-out songs (E18 / E27): for each reference quality,
its share of frames, exact-chord recall and right-root rate, after an HMM (mean chord 5 s, power 2,
built from each bundle's own change matrix, so every model is decoded the same way).
Also the overall majmin / full accuracy and changes. Uses posteriors cached by analysis_fixed_lag.py.

usage: python experiments/analysis_per_type.py models/v7 models/e27_rare
"""
import sys
from pathlib import Path

import numpy as np

from chordstream.evaluate import change_points, majmin_view
from chordstream.hmm import OnlineHMM
from chordstream.vocab import IGNORE, NO_CHORD, QUALITIES, decode_ext

for b in sys.argv[1:]:
    d = np.load(Path("data/analysis") / f"{Path(b).name}__cache_ext.npz", allow_pickle=True)
    base = OnlineHMM.load(Path(b) / "hmm.npz")
    p_stay = 1 - 0.04 / 5.0
    hmm = OnlineHMM(p_stay * np.eye(base.n) + (1 - p_stay) * base.C, base.prior, 2.0)
    stats = {q: [0, 0, 0] for q in QUALITIES}
    fc = fn = mc = mn = npred = nref = 0
    for probs, y in zip(d["probs"], d["labels"]):
        path = hmm.filter(probs)
        m = y != IGNORE
        fc += (path[m] == y[m]).sum(); fn += m.sum()
        p2, y2 = majmin_view(path, y); m2 = y2 != IGNORE
        mc += (p2[m2] == y2[m2]).sum(); mn += m2.sum()
        npred += len(change_points(path)); nref += len(change_points(y))
        for t in np.flatnonzero(m & (y != NO_CHORD)):
            r, q = decode_ext(int(y[t]))
            s = stats[QUALITIES[q]]
            s[0] += 1
            s[1] += path[t] == y[t]
            pd = decode_ext(int(path[t]))
            s[2] += pd is not None and pd[0] == r
    total = sum(s[0] for s in stats.values())
    print(f"{b}: majmin {mc / mn:.4f}  full {fc / fn:.4f}  changes {npred / nref:.2f}x")
    print("   " + "  ".join(f"{q} {s[0] / total * 100:.1f}% {s[1] / max(1, s[0]) * 100:.0f}/{s[2] / max(1, s[0]) * 100:.0f}"
                          for q, s in stats.items()))
