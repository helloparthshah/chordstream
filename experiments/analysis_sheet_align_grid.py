"""Grid over aligner settings (ordered mode) on the first 12 held-out Billboard songs."""
import itertools, sys
import numpy as np
from chordstream.data.dataset import split_files
from chordstream.data.sheet_align import align_ordered
from chordstream.evaluate import majmin_view
from chordstream.vocab import IGNORE

_, val = split_files("data/cache_ext")
songs = []
for f in val[:12]:
    d = np.load(f); y = d["labels"].astype(np.int64)
    seq = [int(c) for i, c in enumerate(y) if c != IGNORE and (i == 0 or c != y[i - 1])]
    songs.append((d["feats"].astype(np.float32), y, seq))
for beta, dur in itertools.product((4.0, 8.0, 16.0, 32.0), (1.0, 2.0, 4.0)):
    cf = nf = cm = nm = 0
    for feats, y, seq in songs:
        lab = align_ordered(feats, seq, beta=beta, mean_chord_seconds=dur, shifts=[0])["labels"]
        m = y != IGNORE; cf += (lab[m] == y[m]).sum(); nf += m.sum()
        p, yy = majmin_view(lab, y); m2 = yy != IGNORE; cm += (p[m2] == yy[m2]).sum(); nm += m2.sum()
    print(f"beta {beta:5.1f} dur {dur:.1f}: full {cf/nf:.4f} majmin {cm/nm:.4f}", flush=True)
