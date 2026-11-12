"""Audit annotated training songs: do the chord labels actually match the audio?

Independent of the model, per song:
  resid_s      best global offset (s) of labels vs audio chroma, searched +-1 s. Should be ~0
               after the fetch-time alignment; large values mean the alignment slipped.
  drift_s      spread of best local offsets across 30 s windows. Large = the recording is a
               different edit / tempo than the annotation, so labels slide over the song.
  key_margin   template score at the labelled key minus the best wrong transposition.
               Small or negative = transposed / sped-up upload or wrong song.
  tuning_ratio energy between semitones / energy on semitones (CQT has 2 bins per semitone).
               High = recording is far from A440 (e.g. a quarter tone off), notes smear.
  score        mean chroma-template agreement (dense mixes score lower; weak signal alone).
and, if a model is given:
  model_acc    frame accuracy of the model on the song (low on a training song = suspicious).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..vocab import IGNORE, NO_CHORD
from .youtube import chroma_from_cqt, offset_curve, score_at, templates

FPS = 25


def audit_song(feats: np.ndarray, labels: np.ndarray, bins_per_semitone: int = 2) -> dict:
    chroma = chroma_from_cqt(feats, bins_per_semitone)
    tpl = templates(labels)
    curve = offset_curve(chroma, tpl, FPS)  # +-1 s
    best = int(curve.argmax())
    out = {"resid_s": (best - FPS) / FPS, "score": float(curve[FPS])}

    # Local alignment in 30 s windows (15 s hop): drift = how much the best offset moves.
    win, hop, local = 30 * FPS, 15 * FPS, []
    for s in range(0, max(1, len(labels) - win + 1), hop):
        lab = np.full_like(labels, NO_CHORD)
        lab[s:s + win] = labels[s:s + win]
        if ((lab != NO_CHORD) & (lab != IGNORE)).sum() < 10 * FPS:
            continue
        c = offset_curve(chroma, templates(lab), int(1.5 * FPS))
        local.append((int(c.argmax()) - int(1.5 * FPS)) / FPS)
    out["drift_s"] = float(max(local) - min(local)) if len(local) >= 2 else 0.0
    out["local_offsets"] = local

    wrong = max(score_at(chroma, templates(labels, k), 0) for k in range(1, 12))
    out["key_margin"] = float(curve[FPS] - wrong)

    lin = np.expm1(feats.astype(np.float64))  # features are log1p(100*|CQT|)
    e = (lin ** 2).sum(0)[12:120]  # mid range, skip the extreme bins
    on, between = e[0::2].sum(), e[1::2].sum()  # bin 0 = C (on a semitone)
    out["tuning_ratio"] = float(between / max(on, 1e-9))
    out["chord_frac"] = float(((labels != NO_CHORD) & (labels != IGNORE)).mean())
    return out


def run(cache: str, out_csv: str, bundle: str | None = None, manifest: str = "data/billboard/manifest.jsonl") -> list[dict]:
    from .dataset import split_files

    train, val = split_files(cache)
    meta = {}
    if Path(manifest).exists():
        for line in Path(manifest).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            meta[r["id"]] = r
    net = dev = None
    if bundle:
        import torch

        from ..export import load_net

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        net, _ = load_net(Path(bundle))
        net.to(dev)

    rows = []
    for split, files in (("train", train), ("val", val)):
        for f in files:
            d = np.load(f)
            feats, labels = d["feats"].astype(np.float32), d["labels"].astype(np.int64)
            sid = f.stem.split("__")[-1]
            r = {"id": sid, "split": split, "artist": meta.get(sid, {}).get("artist", ""),
                 "title": meta.get(sid, {}).get("title", ""), "video": meta.get(sid, {}).get("video", ""),
                 **audit_song(feats, labels)}
            if net is not None:
                from ..train import predict_song

                pred = predict_song(net, feats, dev).argmax(1)
                v = labels != IGNORE
                r["model_acc"] = float((pred[v] == labels[v]).mean())
            r.pop("local_offsets")
            rows.append(r)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def local_offsets(feats: np.ndarray, labels: np.ndarray, win_s: float = 20.0, hop_s: float = 5.0,
                  search_s: float = 3.0, bins_per_semitone: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """(centres, offsets) in frames: best label->audio offset per window, searched +-search_s.

    Uses a shorter window and finer hop than the audit so edits and tempo differences
    can be followed. offset s means annotation time t sounds at audio time t + s.
    """
    chroma = chroma_from_cqt(feats, bins_per_semitone)
    win, hop, search = int(win_s * FPS), int(hop_s * FPS), int(search_s * FPS)
    centres, offs = [], []
    for s in range(0, max(1, len(labels) - win + 1), hop):
        lab = np.full_like(labels, NO_CHORD)
        lab[s:s + win] = labels[s:s + win]
        if ((lab != NO_CHORD) & (lab != IGNORE)).sum() < win // 3:
            continue
        c = offset_curve(chroma, templates(lab), search)
        if c.max() <= 0:
            continue
        centres.append(s + win // 2)
        offs.append(int(c.argmax()) - search)
    return np.array(centres), np.array(offs, dtype=np.float64)


def realign_map(feats: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """(source frame per audio frame, out-of-range mask) of the re-timing, or None if unchanged."""
    centres, offs = local_offsets(feats, labels)
    if len(offs) < 3:
        return None
    # Median-filter the offsets (one bad window must not bend the curve), then interpolate.
    k = 2
    sm = np.array([np.median(offs[max(0, i - k): i + k + 1]) for i in range(len(offs))])
    t = np.arange(len(labels))
    shift = np.interp(t, centres, sm)  # audio frame t holds annotation frame t - shift
    src = np.clip(np.round(t - shift).astype(int), 0, len(labels) - 1)
    return src, (t - shift < 0) | (t - shift >= len(labels))


def apply_map(arr: np.ndarray, m, fill) -> np.ndarray:
    if m is None:
        return arr
    src, outside = m
    out = arr[src].copy()
    out[outside] = fill
    return out


def realign(feats: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Re-time labels with a smooth, piecewise-linear offset curve fitted to local alignments."""
    return apply_map(labels, realign_map(feats, labels), NO_CHORD)
