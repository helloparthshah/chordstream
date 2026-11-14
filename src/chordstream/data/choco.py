"""E40: more human-labelled songs from ChoCo's audio-aligned partitions (Isophonics, USPop2002,
Robbie Williams, JAAH). ChoCo (github.com/smashub/choco) converts each dataset's time-aligned chord
annotations to JAMS; here they become .lab files plus a catalog for youtube.fetch_catalog, which finds,
downloads, aligns and verifies the matching recording.

Excluded: songs that are also among the Billboard held-out test songs (same artist + title) and the
single-song test track (Bohemian Rhapsody). All songs are train-only (prefix "choco__").
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from .billboard import write_lab
from .youtube import Entry

PARTITIONS = ("isophonics", "uspop2002", "robbie-williams", "jaah")
EXCLUDE_TITLES = {("queen", "bohemian rhapsody")}


def norm(s: str) -> str:
    s = s.lower().replace("&", "and")
    s = re.sub(r"\(.*?\)|\[.*?\]", "", s)
    s = re.sub(r"^the\s+", "", s.strip())
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _meta(row: dict) -> tuple[str, str]:
    artist = row.get("file_performer") or row.get("track_performer") or ""
    title = row.get("track_title") or row.get("file_title") or ""
    return artist.strip(), title.replace("_", " ").strip()


def load_choco_catalog(root: str | Path, lab_dir: str | Path, exclude: set[tuple[str, str]] = frozenset(),
                       partitions=PARTITIONS) -> list[Entry]:
    """Write one .lab per annotated song to lab_dir and return catalog entries (id = ChoCo id)."""
    root, lab_dir = Path(root), Path(lab_dir)
    lab_dir.mkdir(parents=True, exist_ok=True)
    skip = {(norm(a), norm(t)) for a, t in set(exclude) | EXCLUDE_TITLES}
    out, seen = [], set()
    for part in partitions:
        base = root / "partitions" / part / "choco"
        with open(base / "meta.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            artist, title = _meta(row)
            key = (norm(artist), norm(title))
            if not all(key) or key in skip or key in seen:
                continue
            jp = base / "jams" / f"{row['id']}.jams"
            if not jp.exists():  # e.g. Isophonics Michael Jackson: segments/keys only, no chords
                continue
            jams = json.loads(jp.read_text(encoding="utf-8"))
            chords = next((a for a in jams["annotations"] if a["namespace"].startswith("chord")), None)
            if chords is None or len(chords["data"]) < 8:
                continue
            segs = [(float(o["time"]), float(o["time"]) + float(o["duration"]), str(o["value"])) for o in chords["data"]]
            segs = [s for s in segs if s[1] > s[0]]
            dur = float(jams["file_metadata"].get("duration") or segs[-1][1])
            lab = lab_dir / f"{row['id']}.lab"
            write_lab(segs, lab)
            seen.add(key)
            out.append(Entry(row["id"], title, artist, dur, lab))
    return out
