"""W06 lyric-sync eval set: timed lyrics (LRCLIB) for the held-out test songs we have audio for.
Kept only when LRCLIB's track length matches our recording within 3 s (same version, so the line
timestamps line up). Written to data/lyric_eval/<id>.json (local only; lyrics are copyrighted).
usage: python experiments/lyric_eval_set.py
"""
import csv, json, time, urllib.parse, urllib.request, zlib
from pathlib import Path

import soundfile as sf

from chordstream.data.bollywood import trusted_ids_v2
from chordstream.data.dataset import split_files

API, OUT = "https://lrclib.net/api", Path("data/lyric_eval")
UA = {"User-Agent": "rubato-eval/0.1 (chord recognition research)"}


def get(path, **params):
    url = f"{API}/{path}?{urllib.parse.urlencode({k: v for k, v in params.items() if v})}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=10) as r:
            return json.loads(r.read())
    except Exception:
        return None


def pick(results, dur):
    good = [r for r in results or [] if r.get("syncedLyrics") and abs((r.get("duration") or 0) - dur) <= 3]
    return min(good, key=lambda r: abs(r["duration"] - dur)) if good else None


songs = []
meta = {int(r["id"]): r for r in csv.DictReader(open("data/billboard_raw/index.csv", encoding="utf-8")) if r["title"]}
_, val = split_files("data/cache_ext")
for f in val:
    i = int(f.stem.split("__")[1]); m = meta.get(i)
    if m: songs.append(("en", f"bb{i:04d}", Path(f"data/billboard/{i:04d}.flac"), m["artist"].split(",")[0], m["title"]))
ok = trusted_ids_v2("align_cache_bolly3.jsonl")
bmeta = {json.loads(l)["id"]: json.loads(l) for l in open("data/bollywood/songs.jsonl", encoding="utf-8")}
for f in sorted(Path("data/cache_bolly3").glob("*.npz")):
    yid = f.stem.split("__")[-1]
    if zlib.crc32(f.stem.encode()) % 1000 < 300 and yid in ok and yid in bmeta:
        songs.append(("hi", f"bo_{yid}", Path(f"data/bollywood/audio/{yid}.flac"), "", bmeta[yid]["song"]))
OUT.mkdir(parents=True, exist_ok=True)
kept = 0
for lang, sid, audio, artist, title in songs:
    if not audio.exists():
        continue
    dur = sf.info(str(audio)).duration
    rec = pick(get("search", track_name=title, artist_name=artist) if artist else get("search", q=title), dur)
    time.sleep(0.3)
    if rec:
        kept += 1
        (OUT / f"{sid}.json").write_text(json.dumps({"lang": lang, "audio": str(audio), "title": title, "duration": dur,
                                                     "lrc": rec["syncedLyrics"]}, ensure_ascii=False), encoding="utf-8")
    print(f"{sid:16s} {'ok ' if rec else '-- '} {artist} - {title}", flush=True)
print(f"kept {kept} of {len(songs)}")
