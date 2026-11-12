"""E21 validation: how accurate are sheet-aligned labels? Builds a "sheet" (the chord sequence, no
times) from each held-out Billboard song's annotations, aligns it with data/sheet_align, and scores
the aligned labels against the true timed labels. Also checks transposition recovery: the sheet
is given shifted by a random amount and the aligner must undo it.

usage: python experiments/analysis_sheet_align.py data/cache_ext
"""
import sys

import numpy as np

from chordstream.data.dataset import split_files
from chordstream.data.sheet_align import align_ordered, align_sheet
from chordstream.evaluate import majmin_view
from chordstream.vocab import IGNORE, NO_CHORD, transpose

cache = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else 'grammar'
_, val = split_files(cache)
rng = np.random.default_rng(0)
c_full = c_mm = n_full = n_mm = 0
key_ok = 0
for f in val:
    d = np.load(f)
    feats, y = d["feats"].astype(np.float32), d["labels"].astype(np.int64)
    seq = [int(c) for i, c in enumerate(y) if c != IGNORE and (i == 0 or c != y[i - 1])]
    k = int(rng.integers(12))
    given = [transpose(c, k) if c != NO_CHORD else c for c in seq]
    r = align_sheet(feats, given) if mode == 'grammar' else align_ordered(feats, given)
    key_ok += (r["shift"] + k) % 12 == 0
    lab = r["labels"]
    m = y != IGNORE
    c_full += int((lab[m] == y[m]).sum()); n_full += int(m.sum())
    p, yy = majmin_view(lab, y); m2 = yy != IGNORE
    c_mm += int((p[m2] == yy[m2]).sum()); n_mm += int(m2.sum())
    print(f"{f.stem:22s} acc {np.mean(lab[m] == y[m]):.3f}  margin {r['margin']:.3f}  "
          f"key {'ok' if (r['shift'] + k) % 12 == 0 else 'WRONG'}", flush=True)
print(f"\n{len(val)} songs: full {c_full / n_full:.4f}  majmin {c_mm / n_mm:.4f}  key recovered {key_ok}/{len(val)}")
