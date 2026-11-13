"""Stream one song through the live engine and report flicker of the live chord and the Steady display.

usage: python experiments/song_flicker.py models/v11 song.flac [ref.lab]
"""
import sys

import numpy as np

from chordstream.data.dataset import load_audio
from chordstream.stream import ChordStream
from chordstream.vocab import CLASS_NAMES, to_majmin

eng = ChordStream(sys.argv[1]); y = load_audio(sys.argv[2], eng.sample_rate); hop = eng.cfg.frontend.hop_length
ev = [e for s in range(0, len(y), hop) for e in eng.push(y[s:s + hop])]
FS = eng.cfg.frontend.frame_seconds; idx = {n: i for i, n in enumerate(CLASS_NAMES)}


def report(name, seq):
    pm = np.array([to_majmin(c) for c in seq]); b = np.r_[0, np.flatnonzero(np.diff(pm)) + 1, len(pm)]
    lens, vals = np.diff(b), pm[b[:-1]]; mins = len(pm) * FS / 60
    blips = sum(1 for i in range(1, len(lens) - 1) if vals[i - 1] == vals[i + 1] and lens[i] * FS < 0.5)
    print(f"{name:7s} {(len(lens) - 1) / mins:5.1f} changes/min  {blips / mins:4.1f} blips/min  "
          f"segments < 0.5 s {(lens * FS < 0.5).mean():.0%}  ({len(lens) - 1} changes in {mins:.1f} min)")
    return b, vals


live = np.array([e.index for e in ev]); steady = np.array([idx[e.display_label] for e in ev])
report("live", live); b, vals = report("steady", steady)
if len(sys.argv) > 3:
    from chordstream.data.billboard import read_lab
    from chordstream.data.dataset import frame_labels
    from chordstream.evaluate import majmin_view
    from chordstream.vocab import IGNORE
    ref = frame_labels(read_lab(sys.argv[3]), len(ev), eng.cfg.frontend)
    report("labels", np.where(ref == IGNORE, 0, ref))
    for nm, seq in (("live", live), ("steady", steady)):
        p2, y2 = majmin_view(seq, ref); m = y2 != IGNORE
        print(f"{nm} majmin accuracy {(p2[m] == y2[m]).mean():.3f}")
print("\nSteady display timeline (majmin):")
print("  ".join(f"{b[i] * FS:.1f}s {CLASS_NAMES[v]}" for i, v in enumerate(vals)))
