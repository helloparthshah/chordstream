"""Gold-label annotation API (E21): hand-corrected chord timings for the Bollywood test songs.

The page (web/annotate.html) loads a test song's audio and its sheet-aligned chord segments, lets a
person fix chord names and boundaries while listening, and saves the result to
data/bollywood/gold/<id>.json. Gold songs then replace the aligned labels in the Bollywood benchmark.
"""

from __future__ import annotations

import json
import time
import zlib
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .vocab import CLASS_NAMES, IGNORE, NO_CHORD, pretty_name, transpose

router = APIRouter(prefix="/api/gold")
ROOT = Path("data/bollywood")
CACHE = Path("data/cache_bolly2")
GOLD = ROOT / "gold"
FS = 0.04


def _is_test(stem: str) -> bool:
    return (zlib.crc32(stem.encode()) % 1000) < 300


def _songs() -> dict[str, dict]:
    return {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "songs.jsonl").read_text(encoding="utf-8").splitlines()}


def _diag() -> dict[str, dict]:
    p = ROOT / "align2.jsonl"
    return {json.loads(l)["id"]: json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}


def _segments(labels: np.ndarray) -> list[dict]:
    segs, start = [], 0
    for t in range(1, len(labels) + 1):
        if t == len(labels) or labels[t] != labels[t - 1]:
            c = int(labels[start])
            segs.append({"start": round(start * FS, 3), "end": round(t * FS, 3),
                         "label": CLASS_NAMES[c] if c >= 0 else "X"})
            start = t
    return segs


@router.get("/songs")
def songs():
    meta, diag = _songs(), _diag()
    out = []
    for f in sorted(CACHE.glob("*.npz")):
        sid = f.stem.split("__")[-1]
        if not _is_test(f.stem) or not diag.get(sid, {}).get("kept"):
            continue
        g = GOLD / f"{sid}.json"
        done = json.loads(g.read_text(encoding="utf-8")).get("status") if g.exists() else None
        out.append({"id": sid, "song": meta[sid]["song"], "movie": meta[sid].get("movie", ""), "status": done})
    return out


@router.get("/{sid}")
def song(sid: str):
    meta, diag = _songs(), _diag()
    if sid not in meta or not (CACHE / f"bolly__{sid}.npz").exists():
        raise HTTPException(404)
    g = GOLD / f"{sid}.json"
    sheet = json.loads((ROOT / "sheets" / f"{sid}.json").read_text(encoding="utf-8"))
    capo = diag[sid]["capo"]
    chords = sorted({transpose(c, capo) for c in sheet["chords"]})
    base = {"id": sid, "song": meta[sid]["song"], "movie": meta[sid].get("movie", ""), "capo": capo,
            "sheet_chords": [{"label": CLASS_NAMES[c], "name": pretty_name(c)} for c in chords],
            "all_chords": [{"label": CLASS_NAMES[c], "name": pretty_name(c)} for c in range(len(CLASS_NAMES))]}
    if g.exists():
        return {**base, **json.loads(g.read_text(encoding="utf-8"))}
    d = np.load(CACHE / f"bolly__{sid}.npz")
    lab = d["labels_full"] if "labels_full" in d.files else d["labels"]
    return {**base, "segments": _segments(lab.astype(np.int64)), "status": None}


@router.get("/{sid}/audio")
def audio(sid: str):
    p = ROOT / "audio" / f"{sid}.flac"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="audio/flac")


@router.post("/{sid}")
def save(sid: str, body: dict):
    segs = body.get("segments")
    if not isinstance(segs, list) or not segs:
        raise HTTPException(400, "segments required")
    valid = set(CLASS_NAMES) | {"X"}
    for s in segs:
        if s.get("label") not in valid or not (0 <= float(s["start"]) < float(s["end"])):
            raise HTTPException(400, f"bad segment {s}")
    GOLD.mkdir(parents=True, exist_ok=True)
    rec = {"segments": segs, "status": body.get("status", "in progress"), "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
    (GOLD / f"{sid}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return {"ok": True, "saved": rec["saved"]}


def gold_labels(sid: str, n_frames: int) -> np.ndarray | None:
    """Frame labels from a finished gold file (None if the song isn't marked done)."""
    g = GOLD / f"{sid}.json"
    if not g.exists():
        return None
    rec = json.loads(g.read_text(encoding="utf-8"))
    if rec.get("status") != "done":
        return None
    out = np.full(n_frames, IGNORE, np.int64)
    t = (np.arange(n_frames) + 0.5) * FS
    for s in rec["segments"]:
        c = IGNORE if s["label"] == "X" else (NO_CHORD if s["label"] == "N" else CLASS_NAMES.index(s["label"]))
        out[(t >= s["start"]) & (t < s["end"])] = c
    return out
