"""Local gold-label annotation app (E21): `chordstream serve` -> http://127.0.0.1:8000/annotate.

Hand-corrected chord labels for the Bollywood test songs are written to data/ (chordstream.gold).
Live chord recognition runs in the browser instead (web/engine.js, GitHub Pages site); the
Python live engine remains available as `chordstream transcribe` / `chordstream mic`.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse

from .gold import router as gold_router

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="chordstream annotate")
app.include_router(gold_router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return RedirectResponse("/annotate")


@app.get("/annotate", response_class=HTMLResponse)
def annotate():
    return HTMLResponse((WEB_DIR / "annotate.html").read_text(encoding="utf-8"), headers={"Cache-Control": "no-cache"})
