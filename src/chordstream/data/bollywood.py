"""E21: Bollywood songs with human chord labels from Ultimate Guitar sheets.

Stages (each resumable; state in data/bollywood/):
  1. songs     YouTube playlists of each year's top Bollywood songs -> songs.jsonl
  2. sheets    match each song to a UG "Chords" sheet (search, title overlap, rating x votes) and
               parse it into an ordered chord list with line starts -> sheets/<yt id>.json
  3. audio     download audio for songs that have a sheet -> audio/<yt id>.flac
  4. cache     CQT features + sheet jump-alignment (data/sheet_align, no neural model) -> extended
               cache files with labels, plus per-song alignment diagnostics

Requests are spaced out (UG ~1.5 s, YouTube via yt-dlp with the user's Firefox session).
"""

from __future__ import annotations

import html
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from ..vocab import IGNORE, NO_CHORD, parse_freetext_ext

ROOT = Path("data/bollywood")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
LABELS = ("t-series", "sony music india", "zee music company", "tips official", "saregama music", "yrf",
          "eros now music", "speed records", "times music")


def _ydl(extra: dict | None = None):
    import yt_dlp

    from .youtube import YDL_EXTRA

    return yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True,
                             **YDL_EXTRA, **(extra or {})})


def find_playlist(year: int) -> dict | None:
    q = urllib.parse.quote_plus(f"top bollywood songs {year}")
    with _ydl({"playlistend": 15}) as y:
        info = y.extract_info(f"https://www.youtube.com/results?search_query={q}&sp=EgIQAw%253D%253D", download=False)
    entries = [e for e in info.get("entries", []) if e.get("id", "").startswith("PL")]
    for e in entries:  # prefer a record label's own playlist, then any whose title names the year
        if any(l in (e.get("channel") or e.get("uploader") or "").lower() for l in LABELS) and str(year) in e.get("title", ""):
            return e
    return next((e for e in entries if str(year) in e.get("title", "")), None)


def playlist_entries(pid: str, limit: int) -> list[dict]:
    with _ydl({"playlistend": limit}) as y:
        info = y.extract_info(f"https://www.youtube.com/playlist?list={pid}", download=False)
    return [{"id": e["id"], "title": e["title"], "duration": e.get("duration")}
            for e in info.get("entries", []) if e and e.get("id") and e.get("title")
            and e["title"] not in ("[Deleted video]", "[Private video]")]


_NOISE = re.compile(r"\b(full|video|song|songs|lyrical|lyric|lyrics|official|audio|hd|4k|new|latest|hindi|"
                    r"bollywood|movie|film|feat|ft|with|from|the|title|track|remix|version|teaser|trailer)\b", re.I)


def clean_title(title: str) -> tuple[str, str]:
    """('song name', 'movie or rest') from a YouTube title like 'Full Video: Raat Bhar | Heropanti | ...'."""
    t = unicodedata.normalize("NFKD", title)
    t = re.sub(r"\(.*?\)|\[.*?\]", " ", t)
    parts = [p.strip() for p in re.split(r"[|:\-–—]", t) if p.strip()]
    parts = [p for p in parts if _NOISE.sub("", p).strip()]
    if not parts:
        return title, ""
    song = _NOISE.sub("", parts[0]).strip()
    rest = _NOISE.sub("", parts[1]).strip() if len(parts) > 1 else ""
    return re.sub(r"\s+", " ", song), re.sub(r"\s+", " ", rest)


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def _store(page: str) -> dict | None:
    m = re.search(r'class="js-store" data-content="([^"]+)"', page)
    return json.loads(html.unescape(m.group(1))) if m else None


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 1}


def ug_search(song: str, movie: str) -> dict | None:
    """Best UG 'Chords' result whose song name shares most words with the song title."""
    best, best_key = None, None
    for q in ([f"{song} {movie}"] if movie else []) + [song]:
        st = _store(_get("https://www.ultimate-guitar.com/search.php?search_type=title&value=" + urllib.parse.quote(q)))
        time.sleep(1.5)
        results = (st or {}).get("store", {}).get("page", {}).get("data", {}).get("results", []) or []
        want = _tokens(song)
        for r in results:
            if r.get("type") != "Chords" or not r.get("tab_url"):
                continue
            have = _tokens(r.get("song_name", ""))
            overlap = len(want & have) / max(1, len(want))
            if overlap < 0.6:
                continue
            key = (overlap, (r.get("rating") or 0) * np.log1p(r.get("votes") or 0))
            if best_key is None or key > best_key:
                best, best_key = r, key
        if best is not None:
            break
    return best


_CH = re.compile(r"\[ch\](.*?)\[/ch\]")


def parse_sheet(content: str) -> dict:
    """Ordered chords and the index of each line's first chord (jump targets for alignment)."""
    chords, line_starts, sections, raw_bad = [], [], [], 0
    for line in content.splitlines():
        if re.match(r"^\s*\[[A-Za-z][^\]]*\]\s*$", line) and "[ch]" not in line and "[tab]" not in line:
            sections.append(len(chords))
        found = _CH.findall(line)
        if not found:
            continue
        line_starts.append(len(chords))
        for raw in found:
            c = parse_freetext_ext(raw)
            if c is None or c == IGNORE:
                raw_bad += 1
                continue
            if c != NO_CHORD:
                chords.append(int(c))
    return {"chords": chords, "line_starts": sorted(set(i for i in line_starts if i < len(chords))),
            "sections": sorted(set(i for i in sections if i < len(chords))), "unparsed": raw_bad}


def ug_sheet(tab_url: str) -> dict | None:
    st = _store(_get(tab_url))
    time.sleep(1.5)
    if not st:
        return None
    data = st.get("store", {}).get("page", {}).get("data", {})
    content = (data.get("tab_view", {}) or {}).get("wiki_tab", {}).get("content")
    if not content:
        return None
    tab = data.get("tab", {}) or {}
    meta = (data.get("tab_view", {}) or {}).get("meta", {}) or {}
    out = parse_sheet(content)
    out.update({"url": tab_url, "song_name": tab.get("song_name"), "artist_name": tab.get("artist_name"),
                "rating": tab.get("rating"), "votes": tab.get("votes"), "capo": meta.get("capo"),
                "tonality": tab.get("tonality_name")})
    return out


# ---------------------------------------------------------------- stages

def stage_songs(years: range, per_year: int) -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "songs.jsonl"
    seen = {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
    n = 0
    with path.open("a", encoding="utf-8") as fh:
        for year in years:
            pl = find_playlist(year)
            if not pl:
                print(f"{year}: no playlist")
                continue
            ents = playlist_entries(pl["id"], per_year)
            print(f"{year}: {pl.get('title')} ({pl.get('channel') or pl.get('uploader')}) -> {len(ents)} videos", flush=True)
            for e in ents:
                if e["id"] in seen or (e["duration"] and not 120 <= e["duration"] <= 480):
                    continue
                seen.add(e["id"])
                song, movie = clean_title(e["title"])
                fh.write(json.dumps({**e, "year": year, "playlist": pl["id"], "song": song, "movie": movie}) + "\n")
                n += 1
    return n


def songs() -> list[dict]:
    return [json.loads(l) for l in (ROOT / "songs.jsonl").read_text(encoding="utf-8").splitlines()]


def stage_sheets(min_chords: int = 8) -> dict:
    out = ROOT / "sheets"
    out.mkdir(parents=True, exist_ok=True)
    stats = {"matched": 0, "none": 0, "short": 0, "done": 0}
    for s in songs():
        dst = out / f"{s['id']}.json"
        if dst.exists() or (out / f"{s['id']}.none").exists():
            stats["done"] += 1
            continue
        try:
            r = ug_search(s["song"], s["movie"])
            sheet = ug_sheet(r["tab_url"]) if r else None
        except Exception as ex:  # network hiccup: retry on the next run
            print(f"  {s['song']}: {ex}")
            continue
        if not sheet:
            (out / f"{s['id']}.none").touch()
            stats["none"] += 1
            continue
        if len(sheet["chords"]) < min_chords:
            (out / f"{s['id']}.none").touch()
            stats["short"] += 1
            continue
        if not plausible_match(s, sheet):
            (out / f"{s['id']}.none").touch()
            stats["none"] += 1
            continue
        dst.write_text(json.dumps(sheet), encoding="utf-8")
        stats["matched"] += 1
        print(f"  {s['song']!r} -> {sheet['song_name']} ({sheet['votes']} votes, {len(sheet['chords'])} chords)", flush=True)
    return stats


def plausible_match(song: dict, sheet: dict) -> bool:
    """Reject look-alike titles (e.g. 'You Are My Soniya' -> 'You Are My Sunshine') and same-named
    western songs: every word of UG's song name must be in the YouTube title, and the sheet must be
    filed as a soundtrack or by an artist named in the title."""
    title = _tokens(song["title"])
    ug_song = _tokens(re.sub(r"\(.*?\)", " ", sheet.get("song_name") or ""))
    artist = (sheet.get("artist_name") or "").lower()
    if not ug_song or not ug_song <= title | {"ver", "version", "acoustic", "unplugged"}:
        return False
    return "soundtrack" in artist or bool(_tokens(artist) & title)


def stage_validate() -> dict:
    out = ROOT / "sheets"
    stats = {"kept": 0, "rejected": 0, "duplicate": 0}
    seen: set[str] = set()
    for s in songs():
        f = out / f"{s['id']}.json"
        if not f.exists():
            continue
        sheet = json.loads(f.read_text(encoding="utf-8"))
        if sheet["url"] in seen:  # same song from another playlist: one copy only (no train/test leak)
            f.rename(out / f"{s['id']}.duplicate")
            stats["duplicate"] += 1
        elif plausible_match(s, sheet):
            seen.add(sheet["url"])
            stats["kept"] += 1
        else:
            f.rename(out / f"{s['id']}.rejected")
            stats["rejected"] += 1
    return stats


def stage_audio(cookies_from_browser: str | None = "firefox", sr: int = 16000) -> dict:
    from . import youtube

    if cookies_from_browser:
        youtube.YDL_EXTRA["cookiesfrombrowser"] = (cookies_from_browser,)
    out = ROOT / "audio"
    out.mkdir(parents=True, exist_ok=True)
    stats = {"downloaded": 0, "have": 0, "failed": 0}
    bot = 0
    for s in songs():
        if not (ROOT / "sheets" / f"{s['id']}.json").exists():
            continue
        dst = out / f"{s['id']}.flac"
        if dst.exists():
            stats["have"] += 1
            continue
        try:
            youtube.download_audio(s["id"], dst, ROOT / "_tmp", sr)
            stats["downloaded"] += 1
            time.sleep(2)
        except youtube.BotCheckError:
            bot += 1
            print("  bot check")
            if bot >= 3:
                print("stopping: repeated YouTube bot checks")
                break
            time.sleep(30)
        except Exception as ex:
            stats["failed"] += 1
            print(f"  {s['id']}: {ex}")
    return stats


def stage_cache(out: str | Path = "data/cache_bolly", device: str | None = None) -> dict:
    """Features + jump-aligned labels. Diagnostics (fit, transposition, margin) go to align.jsonl."""
    from ..config import FrontendConfig
    from ..cqt import offline_cqt
    from ..vocab import CLASS_NAMES, NO_ROOT, parse_harte_ext, to_majmin
    from .dataset import load_audio
    from .sheet_align import align_ordered

    cfg = FrontendConfig()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    diag = ROOT / "align.jsonl"
    done = {json.loads(l)["id"] for l in diag.read_text(encoding="utf-8").splitlines()} if diag.exists() else set()
    n = 0
    for s in songs():
        a, sh = ROOT / "audio" / f"{s['id']}.flac", ROOT / "sheets" / f"{s['id']}.json"
        if s["id"] in done or not (a.exists() and sh.exists()):
            continue
        sheet = json.loads(sh.read_text(encoding="utf-8"))
        feats = offline_cqt(load_audio(a, cfg.sample_rate), cfg, device).astype(np.float32)
        r = align_ordered(feats, sheet["chords"], jumps=sheet["line_starts"])
        lab = r["labels"].astype(np.int64)
        ext = [parse_harte_ext(CLASS_NAMES[c]) if c != NO_CHORD else (NO_CHORD, NO_ROOT, NO_ROOT, 0) for c in range(max(lab) + 1)]
        root = np.array([ext[c][1] for c in lab], np.int16)
        bass = np.array([ext[c][2] for c in lab], np.int16)
        notes = np.array([ext[c][3] for c in lab], np.int32)
        mm = np.array([to_majmin(c) for c in lab], np.int16)
        np.savez_compressed(out / f"bolly__{s['id']}.npz", feats=feats.astype(np.float16), labels=lab.astype(np.int16),
                            root=root, bass=bass, notes=notes, labels_mm=mm)
        rec = {"id": s["id"], "song": s["song"], "year": s["year"], "shift": r["shift"], "margin": r["margin"],
               "fit": r.get("fit"), "n_chords": len(sheet["chords"]), "capo": sheet.get("capo")}
        with diag.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        n += 1
        print(f"  {s['song']}: shift {r['shift']} margin {r['margin']:.3f}", flush=True)
    return {"cached": n}


def trusted_ids() -> set[str]:
    """Songs whose aligned transposition equals the sheet's capo (no capo = 0): an independent check
    that the alignment found the right key. 60 / 98 at first run; most others are off by a fifth,
    which chroma confuses."""
    rows = [json.loads(l) for l in (ROOT / "align.jsonl").read_text(encoding="utf-8").splitlines()]
    return {r["id"] for r in rows if r["shift"] == (r["capo"] or 0) % 12}


def mask_boundaries(src: str | Path = "data/cache_bolly", out: str | Path = "data/cache_bolly_masked",
                    frames: int = 8) -> int:
    """Copy of the cache with labels ignored within +-frames of every aligned chord change: the
    chord identities (human) still train, the uncertain timing (aligner) does not. E21c/d."""
    from ..vocab import IGNORE

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(Path(src).glob("*.npz")):
        d = dict(np.load(f))
        y = d["labels"].astype(np.int64)
        near = np.zeros(len(y), bool)
        for c in np.flatnonzero(np.diff(y) != 0) + 1:
            near[max(0, c - frames): c + frames] = True
        for k in ("labels", "root", "bass", "labels_mm"):
            d[k] = np.where(near, IGNORE, d[k]).astype(d[k].dtype)
        d["notes"] = np.where(near, -1, d["notes"]).astype(d["notes"].dtype)
        np.savez_compressed(out / f.name, **d)
        n += 1
    return n
