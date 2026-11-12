"""E21: score models on the Bollywood test songs (30% of data/cache_bolly by a stable hash).

Timing-free sheet metrics (from the UG sheet, transposed by the aligner's chosen shift):
  in_sheet   share of chord frames whose predicted chord is one of the sheet's chords
  coverage   share of the sheet's chord-change sequence found in order in the prediction (LCS)
  changes    predicted chord changes / aligned-label changes
Secondary: frame accuracy against the sheet-aligned labels (majmin view and full vocabulary).
Predictions are the live HMM path (what the readout shows), at the frames they describe.

usage: python experiments/analysis_bollywood.py models/v7 [models/x ...]
"""
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

from chordstream.evaluate import majmin_view
from chordstream.export import load_net
from chordstream.hmm import OnlineHMM
from chordstream.train import predict_song
from chordstream.vocab import IGNORE, NO_CHORD, to_majmin, transpose

CACHE, ROOT = Path("data/cache_bolly"), Path("data/bollywood")


def is_test(stem: str) -> bool:
    return (zlib.crc32(stem.encode()) % 1000) < 300


def lcs(a: list[int], b: list[int]) -> int:
    dp = np.zeros((len(a) + 1, len(b) + 1), np.int32)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i, j] = dp[i - 1, j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1, j], dp[i, j - 1])
    return int(dp[-1, -1])


def segs(path: np.ndarray, min_frames: int = 12) -> list[int]:
    """Chord sequence of the sheet-commit view (segments held >= min_frames)."""
    out, start = [], 0
    for t in range(1, len(path) + 1):
        if t == len(path) or path[t] != path[t - 1]:
            c = int(path[start])
            if t - start >= min_frames and c != NO_CHORD and (not out or out[-1] != c):
                out.append(c)
            start = t
    return out


def main(bundles):
    diag = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "align.jsonl").read_text(encoding="utf-8").splitlines()}
    files = [f for f in sorted(CACHE.glob("*.npz")) if is_test(f.stem)]
    print(f"{len(files)} test songs")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for b in bundles:
        net, _ = load_net(Path(b))
        net.to(dev)
        hmm = OnlineHMM.load(Path(b) / "hmm.npz")
        tot = {k: 0.0 for k in ("in", "n", "cov", "seq", "pc", "rc", "mm_c", "mm_n", "f_c", "f_n")}
        for f in files:
            sid = f.stem.split("__")[-1]
            d = np.load(f)
            y = d["labels"].astype(np.int64)
            path = hmm.filter(predict_song(net, d["feats"].astype(np.float32), dev))
            sheet = json.loads((ROOT / "sheets" / f"{sid}.json").read_text(encoding="utf-8"))
            k = diag[sid]["shift"]
            chords = [transpose(c, k) for c in sheet["chords"]]
            mm = lambda seq: [to_majmin(c) for c in seq]  # noqa: E731
            sheet_set = set(mm(chords))
            pm = np.array(mm(path))
            on = pm != NO_CHORD
            tot["in"] += np.isin(pm[on], list(sheet_set)).sum(); tot["n"] += on.sum()
            ref_seq = [c for i, c in enumerate(mm(chords)) if i == 0 or c != mm(chords)[i - 1]]
            pred_seq = [c for i, c in enumerate(mm(segs(path))) if i == 0 or c != mm(segs(path))[i - 1]]
            tot["cov"] += lcs(ref_seq, pred_seq); tot["seq"] += len(ref_seq)
            tot["pc"] += (np.diff(path) != 0).sum(); tot["rc"] += (np.diff(y) != 0).sum()
            p2, y2 = majmin_view(path, y); m = y2 != IGNORE
            tot["mm_c"] += (p2[m] == y2[m]).sum(); tot["mm_n"] += m.sum()
            m = y != IGNORE
            tot["f_c"] += (path[m] == y[m]).sum(); tot["f_n"] += m.sum()
        print(f"{b:28s} in_sheet {tot['in'] / tot['n']:.3f}  coverage {tot['cov'] / tot['seq']:.3f}  "
              f"changes {tot['pc'] / tot['rc']:.2f}x  aligned majmin {tot['mm_c'] / tot['mm_n']:.3f}  "
              f"full {tot['f_c'] / tot['f_n']:.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
