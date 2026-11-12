"""Extract chord progressions from saved Ultimate Guitar tab pages.

Input is a directory of HTML pages you have saved locally. This module only parses
them; it does not download anything. Check UG's terms of service before collecting
pages at scale.

Two page layouts are supported:
  * legacy:  <script> window.UGAPP.store.page = {...}; </script>
  * current: <div class="js-store" data-content="{...html-escaped json...}">
Chords are the [ch]...[/ch] tags in the tab's wiki content.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Iterator
from pathlib import Path

from ..vocab import CLASS_NAMES, IGNORE, parse_freetext, transpose

_STORE_RE = re.compile(r"window\.UGAPP\.store\.page\s*=\s*")
_JS_STORE_RE = re.compile(r'class="js-store"\s+data-content="([^"]*)"')
_CH_RE = re.compile(r"\[ch\](.*?)\[/ch\]")


def extract_store(page_html: str) -> dict | None:
    m = _STORE_RE.search(page_html)
    if m:
        try:
            obj, _ = json.JSONDecoder().raw_decode(page_html, m.end())
            return obj
        except json.JSONDecodeError:
            pass
    m = _JS_STORE_RE.search(page_html)
    if m:
        try:
            store = json.loads(html.unescape(m.group(1)))
            return store.get("store", {}).get("page", store)
        except json.JSONDecodeError:
            pass
    return None


def _find(obj, key: str):
    if isinstance(obj, dict):
        if key in obj:
            yield obj[key]
        for v in obj.values():
            yield from _find(v, key)
    elif isinstance(obj, list):
        for v in obj:
            yield from _find(v, key)


def chords_from_store(store: dict) -> tuple[list[str], dict]:
    content = next((c for c in _find(store, "content") if isinstance(c, str) and "[ch]" in c), "")
    meta = {}
    for k in ("song_name", "artist_name", "tonality_name", "type"):
        v = next(_find(store, k), None)
        if isinstance(v, str):
            meta[k] = v
    return _CH_RE.findall(content), meta


def normalize(raw: list[str], collapse_repeats: bool = True) -> list[int]:
    """Raw UG strings -> class indices (IGNORE kept as a placeholder, garbage dropped)."""
    out: list[int] = []
    for s in raw:
        c = parse_freetext(s)
        if c is None:
            continue
        if collapse_repeats and out and out[-1] == c:
            continue
        out.append(c)
    return out


def parse_directory(root: str | Path, min_chords: int = 4) -> Iterator[dict]:
    for path in sorted(Path(root).rglob("*.htm*")):
        store = extract_store(path.read_text(encoding="utf-8", errors="replace"))
        if store is None:
            continue
        raw, meta = chords_from_store(store)
        prog = normalize(raw)
        if sum(c != IGNORE for c in prog) < min_chords:
            continue
        yield {"source": path.name, **meta, "raw": raw, "chords": [encode(c) for c in prog]}


def encode(c: int) -> str:
    return "X" if c == IGNORE else CLASS_NAMES[c]


def decode_label(s: str) -> int:
    return IGNORE if s == "X" else CLASS_NAMES.index(s)


def transposed(record: dict) -> Iterator[dict]:
    idx = [decode_label(s) for s in record["chords"]]
    for k in range(12):
        yield {**record, "transpose": k, "chords": [encode(transpose(c, k)) for c in idx]}


def load_progressions(jsonl: str | Path) -> list[list[int]]:
    with open(jsonl, encoding="utf-8") as f:
        return [[decode_label(s) for s in json.loads(line)["chords"]] for line in f if line.strip()]
