"""Time-synced lyrics from LRCLIB (https://lrclib.net): free, no API key.

Results are cached on disk so a song is only looked up once.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://lrclib.net/api"
USER_AGENT = "chordstream/0.1 (local chord recognition app)"
CACHE = Path("data/lyrics_cache")
_LRC_LINE = re.compile(r"^\[(\d+):(\d+(?:\.\d+)?)\](.*)$")
# YouTube-style title noise that hurts matching.
_NOISE = re.compile(r"\s*[\(\[](official|lyric|lyrics|audio|video|hd|4k|remaster|visualizer)[^\)\]]*[\)\]]", re.I)


def _get(path: str, params: dict) -> object:
    url = f"{API}/{path}?{urllib.parse.urlencode({k: v for k, v in params.items() if v})}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def parse_lrc(text: str) -> list[dict]:
    lines = []
    for raw in (text or "").splitlines():
        m = _LRC_LINE.match(raw.strip())
        if m:
            lines.append({"t": int(m.group(1)) * 60 + float(m.group(2)), "text": m.group(3).strip()})
    return sorted(lines, key=lambda l: l["t"])


def clean_title(title: str) -> str:
    return _NOISE.sub("", title).strip()


def _record(r: dict) -> dict:
    return {
        "id": r.get("id"), "artist": r.get("artistName"), "title": r.get("trackName"),
        "album": r.get("albumName"), "duration": r.get("duration"),
        "synced": parse_lrc(r.get("syncedLyrics") or ""), "plain": r.get("plainLyrics") or "",
        "instrumental": bool(r.get("instrumental")),
    }


def find_lyrics(artist: str, title: str, duration: float | None = None) -> dict | None:
    title = clean_title(title)
    key = hashlib.sha1(f"{artist.lower()}|{title.lower()}|{round(duration or 0)}".encode()).hexdigest()
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    best = None
    exact = _get("get", {"artist_name": artist, "track_name": title, "duration": round(duration) if duration else None})
    if isinstance(exact, dict) and exact.get("syncedLyrics"):
        best = exact
    else:
        results = _get("search", {"artist_name": artist, "track_name": title}) or []
        if not results:
            results = _get("search", {"q": f"{artist} {title}"}) or []
        synced = [r for r in results if r.get("syncedLyrics")]
        pool = synced or results
        if pool:
            # Prefer the entry whose length matches what is playing.
            best = min(pool, key=lambda r: abs((r.get("duration") or 0) - (duration or 0)) if duration else 0)
    out = _record(best) if best else None
    if out:  # misses are not cached, so they are retried next time
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def search(query: str) -> list[dict]:
    results = _get("search", {"q": query}) or []
    return [_record(r) for r in results[:10]]


def by_id(lyrics_id: int) -> dict | None:
    r = _get(f"get/{int(lyrics_id)}", {})
    return _record(r) if isinstance(r, dict) else None
