"""Copy a trained model dir to <dir>_h5p2 with the deployed HMM (5 s self-transition, power 2).

usage: python experiments/hmm55.py models/x
"""
import shutil
import sys

import numpy as np

from chordstream.hmm import OnlineHMM

src = sys.argv[1].rstrip("/")
dst = src + "_h5p2"
shutil.copytree(src, dst, dirs_exist_ok=True)
b = OnlineHMM.load(f"{dst}/hmm.npz")
p = 1 - 0.04 / 5.0
OnlineHMM(p * np.eye(b.n) + (1 - p) * b.C, b.prior, 2.0).save(f"{dst}/hmm.npz")
print(dst)
