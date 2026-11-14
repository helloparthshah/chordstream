"""E42: precompute causal beat features for every cached song (data/beat_cache/<stem>.npy).
usage: python experiments/beat_cache.py <cache_dir> [processes]"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np


def one(path):
    from chordstream.beat import cached_beat_features
    d = np.load(path)
    if len(d["labels"]) >= 8:
        cached_beat_features(path, d["feats"])
    return path


if __name__ == "__main__":
    files = sorted(Path(sys.argv[1]).glob("*.npz"))
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 4) as pool:
        for i, _ in enumerate(pool.imap_unordered(one, [str(f) for f in files], chunksize=4)):
            if i % 100 == 0:
                print(f"{i}/{len(files)}", flush=True)
    print("done", len(files))
