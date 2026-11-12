"""Fetch audio for McGill Billboard annotations from YouTube and verify alignment.

YouTube uploads are often not the exact recording that was annotated (different
master, radio edit, extra intro, video skits). Every downloaded song is therefore
checked against its annotation before being used:

  1. Candidate choice: duration must match the annotated length (±tolerance); prefer
     "- Topic" (label auto-uploads of the album master) and "official audio"; reject
     live / cover / karaoke / remix / slowed / sped-up / nightcore / instrumental.
  2. Offset search: CQT-chroma vs. triad templates rendered from the annotation,
     cross-correlated over ±10 s to find the global offset.
  3. Verification: the correct transposition must beat all 11 wrong ones by a clear
     margin (rejects re-pitched uploads and wrong songs), and the offset found on the
     first and second halves must agree (rejects edits and tempo drift).

Downloading from YouTube is against its terms of service and the audio is
copyrighted; keep it local and do not redistribute.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import FrontendConfig
from ..cqt import offline_cqt
from ..vocab import IGNORE, NO_CHORD, decode, transpose
from .billboard import read_lab, write_lab
from .dataset import frame_labels, load_audio

BAD_WORDS = re.compile(
    r"\b(live|cover|karaoke|remix|slowed|sped|nightcore|instrumental|reverb|8d|acoustic|"
    r"tutorial|lesson|piano version|reaction|concert|tribute|medley|mashup|extended mix)\b",
    re.I,
)


@dataclass
class Entry:
    id: str
    title: str
    artist: str
    duration: float
    lab: Path


def load_catalog(raw: str | Path) -> list[Entry]:
    """Unique (artist, title) songs that have both salami_chords.txt and full.lab."""
    raw = Path(raw)
    seen, out = set(), []
    for d in sorted((raw / "salami" / "McGill-Billboard").iterdir()):
        sal, lab = d / "salami_chords.txt", raw / "lab" / "McGill-Billboard" / d.name / "full.lab"
        if not (sal.exists() and lab.exists()):
            continue
        meta, last_t = {}, 0.0
        for line in sal.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("#") and ":" in line:
                k, v = line[1:].split(":", 1)
                meta[k.strip()] = v.strip()
            elif line.strip():
                try:
                    last_t = float(line.split()[0])
                except ValueError:
                    pass
        key = (meta.get("artist", "").lower(), meta.get("title", "").lower())
        if key in seen or not all(key):
            continue
        seen.add(key)
        out.append(Entry(d.name, meta["title"], meta["artist"], last_t, lab))
    return out


# ---------------------------------------------------------------- search + download


def choose_candidate(entry: Entry, results: list[dict], tolerance: float) -> dict | None:
    best, best_score = None, float("inf")
    for r in results:
        dur = r.get("duration")
        if not dur:
            continue
        diff = abs(dur - entry.duration)
        name = f"{r.get('title', '')} {r.get('channel', '') or r.get('uploader', '')}"
        if diff > tolerance or (BAD_WORDS.search(name) and not BAD_WORDS.search(entry.title)):
            continue
        score = diff
        channel = r.get("channel") or r.get("uploader") or ""
        if channel.endswith("- Topic"):
            score -= 3.0
        if re.search(r"official (audio|video)", name, re.I):
            score -= 1.0
        if score < best_score:
            best, best_score = r, score
    return best


YDL_EXTRA: dict = {}  # e.g. {"cookiesfrombrowser": ("firefox",)} set by fetch_billboard


class BotCheckError(RuntimeError):
    pass


def search(query: str, n: int = 8) -> list[dict]:
    import yt_dlp

    opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True, **YDL_EXTRA}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{n}:{query}", download=False)
    return [e for e in (info or {}).get("entries", []) if e]


def download_audio(video_id: str, dst: Path, tmp_dir: Path, sr: int) -> None:
    import imageio_ffmpeg
    import yt_dlp

    tmp_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "quiet": True, "no_warnings": True, "noprogress": True, "format": "bestaudio[ext=m4a]/bestaudio",
        "outtmpl": str(tmp_dir / f"{video_id}.%(ext)s"), "noplaylist": True, **YDL_EXTRA,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
            src = Path(ydl.prepare_filename(info))
    except yt_dlp.utils.DownloadError as ex:
        msg = str(ex).lower()
        if "not a bot" in msg or "try again later" in msg or "rate-limit" in msg or "429" in msg:
            raise BotCheckError("YouTube bot check") from ex
        raise
    try:
        subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
             "-ac", "1", "-ar", str(sr), "-c:a", "flac", str(dst)],
            check=True,
        )
    finally:
        src.unlink(missing_ok=True)


# ---------------------------------------------------------------- alignment


def chroma_from_cqt(feats: np.ndarray, bins_per_semitone: int) -> np.ndarray:
    T, F = feats.shape
    pcs = (np.arange(F) // bins_per_semitone) % 12  # fmin = C
    chroma = np.zeros((T, 12), np.float32)
    for pc in range(12):
        chroma[:, pc] = feats[:, pcs == pc].sum(1)
    chroma -= chroma.mean(1, keepdims=True)  # remove broadband energy
    return chroma / (np.linalg.norm(chroma, axis=1, keepdims=True) + 1e-6)


def templates(labels: np.ndarray, shift: int = 0) -> np.ndarray:
    tpl = np.zeros((len(labels), 12), np.float32)
    for i, c in enumerate(labels):
        if c in (IGNORE, NO_CHORD):
            continue
        root, minor = decode(transpose(int(c), shift))
        for iv in (0, 3 if minor else 4, 7):
            tpl[i, (root + iv) % 12] = 1.0
    tpl -= np.where(tpl.any(1, keepdims=True), tpl.mean(1, keepdims=True), 0)
    return tpl / (np.linalg.norm(tpl, axis=1, keepdims=True) + 1e-6)


def score_at(chroma: np.ndarray, tpl: np.ndarray, s: int) -> float:
    """Mean cosine of chroma[t + s] vs tpl[t] over annotated chord frames."""
    active = np.flatnonzero(np.abs(tpl).sum(1) > 0)
    t = active[(active + s >= 0) & (active + s < len(chroma))]
    return float((chroma[t + s] * tpl[t]).sum(1).mean()) if len(t) > 50 else -1.0


def offset_curve(chroma: np.ndarray, tpl: np.ndarray, max_shift: int) -> np.ndarray:
    return np.array([score_at(chroma, tpl, s) for s in range(-max_shift, max_shift + 1)])


def align(audio: np.ndarray, segs, cfg: FrontendConfig, max_shift_s: float = 10.0) -> dict:
    feats = offline_cqt(audio, cfg, device="cpu")  # keep the GPU free
    chroma = chroma_from_cqt(feats, cfg.bins_per_semitone)
    n_ann = int(max(e for _, e, _ in segs) / cfg.frame_seconds) + 1
    labels = frame_labels(segs, n_ann, cfg)
    max_shift = int(max_shift_s / cfg.frame_seconds)

    curve = offset_curve(chroma, templates(labels), max_shift)
    best = int(curve.argmax())
    shift = best - max_shift
    score = float(curve[best])
    # Wrong-key controls at the chosen offset (also catches re-pitched / sped-up uploads).
    wrong = [score_at(chroma, templates(labels, k), shift) for k in range(1, 12)]
    margin = score - float(np.max(wrong))
    # Drift check: align each half independently.
    half = n_ann // 2
    l1, l2 = labels.copy(), labels.copy()
    l1[half:], l2[:half] = NO_CHORD, NO_CHORD
    s1 = int(offset_curve(chroma, templates(l1), max_shift).argmax()) - max_shift
    s2 = int(offset_curve(chroma, templates(l2), max_shift).argmax()) - max_shift
    return {
        "offset": shift * cfg.frame_seconds, "score": score, "key_margin": margin,
        "drift": abs(s1 - s2) * cfg.frame_seconds,
    }


def accept(a: dict, min_score: float, min_margin: float, max_drift: float) -> bool:
    return a["score"] >= min_score and a["key_margin"] >= min_margin and a["drift"] <= max_drift


# ---------------------------------------------------------------- driver


def fetch_billboard(
    raw: str | Path, out: str | Path, limit: int | None = None, tolerance: float = 4.0,
    min_score: float = 0.2, min_margin: float = 0.08, max_drift: float = 0.3, sleep: float = 2.0,
    cookies_from_browser: str | None = None, max_bot_errors: int = 3,
) -> None:
    if cookies_from_browser:
        YDL_EXTRA["cookiesfrombrowser"] = (cookies_from_browser,)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rejected = out / "_rejected"
    rejected.mkdir(exist_ok=True)
    manifest = out / "manifest.jsonl"
    done = set()
    if manifest.exists():
        done = {json.loads(l)["id"] for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()}
    cfg = FrontendConfig()
    catalog = [e for e in load_catalog(raw) if e.id not in done]
    if limit:
        catalog = catalog[:limit]
    print(f"{len(catalog)} songs to fetch ({len(done)} already in manifest)")
    stats = {"accepted": 0, "rejected": 0, "no_candidate": 0, "error": 0}
    bot_errors = 0

    for i, e in enumerate(catalog, 1):
        rec = {"id": e.id, "title": e.title, "artist": e.artist, "duration": e.duration}
        try:
            cand = choose_candidate(e, search(f"{e.artist} - {e.title}"), tolerance)
            if cand is None:
                rec["status"] = "no_candidate"
            else:
                rec.update(video=cand["id"], video_title=cand.get("title"), channel=cand.get("channel"),
                           video_duration=cand.get("duration"))
                wav = out / f"{e.id}.flac"
                download_audio(cand["id"], wav, out / "_tmp", cfg.sample_rate)
                segs = read_lab(e.lab)
                a = align(load_audio(wav, cfg.sample_rate), segs, cfg)
                rec.update(a)
                if accept(a, min_score, min_margin, max_drift):
                    write_lab([(s + a["offset"], t + a["offset"], c) for s, t, c in segs], out / f"{e.id}.lab")
                    rec["status"] = "accepted"
                else:
                    wav.replace(rejected / wav.name)  # kept for threshold tuning
                    rec["status"] = "rejected"
        except BotCheckError:
            bot_errors += 1
            print(f"[{i}/{len(catalog)}] bot-check    {e.artist} - {e.title}", flush=True)
            if bot_errors >= max_bot_errors:
                print("YouTube is blocking downloads (bot check). Stopping instead of retrying; "
                      "see --cookies-from-browser.")
                break
            continue  # not written to the manifest, so it is retried next run
        except Exception as ex:  # network / extractor errors: log and move on
            rec["status"], rec["error"] = "error", f"{type(ex).__name__}: {ex}"[:300]
        stats[rec["status"]] += 1
        with open(manifest, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        extra = (f"off {rec['offset']:+.2f}s score {rec['score']:.2f} margin {rec['key_margin']:.2f} "
                 f"drift {rec['drift']:.2f}") if "score" in rec else rec.get("error", "")[:80]
        print(f"[{i}/{len(catalog)}] {rec['status']:12s} {e.artist} - {e.title}  {extra}", flush=True)
        time.sleep(sleep)
    print(stats)



def rescore(raw: str | Path, out: str | Path, min_score: float = 0.2, min_margin: float = 0.08,
            max_drift: float = 0.3) -> dict:
    """Re-apply acceptance thresholds to already-downloaded songs (no network)."""
    out = Path(out)
    labs = {e.id: e.lab for e in load_catalog(raw)}
    manifest = out / "manifest.jsonl"
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        if r["status"] not in ("accepted", "rejected"):
            continue
        ok = accept(r, min_score, min_margin, max_drift)
        good, bad = out / f"{r['id']}.flac", out / "_rejected" / f"{r['id']}.flac"
        if ok:
            if bad.exists():
                bad.replace(good)
            segs = read_lab(labs[r["id"]])
            write_lab([(a + r["offset"], b + r["offset"], c) for a, b, c in segs], out / f"{r['id']}.lab")
        else:
            if good.exists():
                good.replace(bad)
            (out / f"{r['id']}.lab").unlink(missing_ok=True)
        r["status"] = "accepted" if ok else "rejected"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    stats: dict = {}
    for r in rows:
        stats[r["status"]] = stats.get(r["status"], 0) + 1
    return stats
