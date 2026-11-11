"""Offline evaluation of the live pipeline on held-out annotated songs.

Everything is computed causally, exactly as the streaming engine would (network posteriors
via the exact chunked forward pass, beat tracker and HMM frame by frame). Metrics:

  acc          frame accuracy (majmin, X excluded)
  change_ratio predicted chord changes / reference changes (1.0 = no over-segmentation)
  change_f     F-measure of predicted vs reference change points within +-tol seconds
  change_dev   median |predicted change - nearest reference change| for hits (seconds)
  beat_f       beat tracker F-measure vs Billboard bar-derived beats (+-70 ms)
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from .beat import BeatTracker, change_scale
from .config import FrontendConfig
from .data.billboard import salami_beats
from .hmm import OnlineHMM
from .vocab import IGNORE

FS = FrontendConfig().frame_seconds


def change_points(seq: np.ndarray) -> np.ndarray:
    return (np.flatnonzero(np.diff(seq) != 0) + 1) * FS


def match_f(pred: np.ndarray, ref: np.ndarray, tol: float) -> tuple[float, list[float]]:
    """Greedy one-to-one matching of event times; returns (F, deviations of hits)."""
    if not len(pred) or not len(ref):
        return 0.0, []
    used = np.zeros(len(ref), bool)
    devs = []
    for t in pred:
        d = np.abs(ref - t)
        d[used] = np.inf
        j = int(d.argmin())
        if d[j] <= tol:
            used[j] = True
            devs.append(float(t - ref[j]))
    hits = len(devs)
    p, r = hits / len(pred), hits / len(ref)
    return (2 * p * r / (p + r) if hits else 0.0), devs


def song_inputs(cache_file: Path, net, device, raw: Path, offsets: dict) -> dict:
    from .train import predict_song

    d = np.load(cache_file)
    feats = d["feats"].astype(np.float32)
    sid = cache_file.stem.split("__")[-1]
    ref_beats = None
    sal = raw / "salami" / "McGill-Billboard" / sid / "salami_chords.txt"
    if sal.exists() and sid in offsets:
        ref_beats = salami_beats(sal.read_text(encoding="utf-8", errors="replace")) + offsets[sid]
    return {"probs": predict_song(net, feats, device), "labels": d["labels"].astype(np.int64),
            "beats": BeatTracker().run(feats), "ref_beats": ref_beats}


def score(songs: list[dict], hmm: OnlineHMM, beat_cfg: dict | None, tol: float = 0.25) -> dict:
    correct = total = n_pred = n_ref = 0
    fs, devs = [], []
    for s in songs:
        scales = None
        if beat_cfg is not None:
            scales = np.array([change_scale(b, **beat_cfg) for b in s["beats"]])
        path = hmm.filter(s["probs"], scales)
        y = s["labels"]
        valid = y != IGNORE
        correct += int((path[valid] == y[valid]).sum())
        total += int(valid.sum())
        pc, rc = change_points(path), change_points(y)
        n_pred, n_ref = n_pred + len(pc), n_ref + len(rc)
        f, dv = match_f(pc, rc, tol)
        fs.append(f)
        devs += dv
    return {"acc": correct / max(1, total), "change_ratio": n_pred / max(1, n_ref),
            "change_f": float(np.mean(fs)), "change_dev": float(np.median(np.abs(devs))) if devs else None}


def beat_quality(songs: list[dict]) -> dict:
    fs, confs = [], []
    for s in songs:
        if s["ref_beats"] is None:
            continue
        times = np.array([(i + 1) * FS for i, b in enumerate(s["beats"]) if b.beat and b.confidence > 0.3])
        ref = s["ref_beats"][(s["ref_beats"] > 5) & (s["ref_beats"] < len(s["beats"]) * FS)]
        fs.append(match_f(times[times > 5], ref, 0.07)[0])
        confs.append(float(np.mean([b.confidence for b in s["beats"]])))
    return {"beat_f": float(np.mean(fs)) if fs else None, "mean_confidence": float(np.mean(confs)) if confs else None,
            "songs": len(fs)}


def run(bundle: str, cache: str, raw: str = "data/billboard_raw", manifest: str = "data/billboard/manifest.jsonl",
        grid: bool = True) -> dict:
    import torch

    from .data.dataset import split_files
    from .export import load_net

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, _ = load_net(Path(bundle))
    net.to(dev)
    offsets = {}
    if Path(manifest).exists():
        for line in Path(manifest).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("status") == "accepted":
                offsets[r["id"]] = r["offset"]
    _, val = split_files(cache)
    songs = [song_inputs(f, net, dev, Path(raw), offsets) for f in val]
    hmm = OnlineHMM.load(Path(bundle) / "hmm.npz")

    out = {"songs": len(songs), "beat_tracker": beat_quality(songs), "baseline": score(songs, hmm, None)}
    if grid:
        results = []
        for lag, window, suppress in itertools.product((0, 2, 4, 6), (1, 2), (0.5, 0.25, 0.1)):
            cfg = {"lag": lag, "window": window, "suppress": suppress}
            results.append({**cfg, **score(songs, hmm, cfg)})
        out["grid"] = sorted(results, key=lambda r: -(r["acc"] + r["change_f"]))
    return out
