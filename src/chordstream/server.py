"""FastAPI server for network streaming.

WebSocket  /ws : send binary frames of little-endian float32 mono PCM at the
                 bundle's sample rate (16 kHz); receive JSON chord events.
                 Send the text message "reset" to clear state between songs.
GET /          : browser microphone demo.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import lyrics
from .nowplaying import now_playing
from .stream import ChordStream
from .vocab import CLASS_NAMES

BUNDLE = os.environ.get("CHORDSTREAM_BUNDLE", "models/v7")
ONLY_CHANGES = os.environ.get("CHORDSTREAM_ONLY_CHANGES", "0") == "1"
ASR_MODEL = os.environ.get("CHORDSTREAM_ASR", "off")  # e.g. "small"; "off" disables vocal alignment
WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="chordstream")
ASR = None
if ASR_MODEL != "off":
    from .asr import VocalTranscriber

    ASR = VocalTranscriber(ASR_MODEL)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

from .gold import router as gold_router  # noqa: E402  (E21 gold-label annotation)

app.include_router(gold_router)


@app.get("/health")
def health():
    return {"status": "ok", "bundle": BUNDLE}


@app.get("/vocab")
def vocab():
    return {"classes": CLASS_NAMES}


@app.get("/api/nowplaying")
async def api_nowplaying():
    try:
        return await now_playing()
    except Exception as e:  # WinRT errors should never take the page down
        return {"available": False, "reason": f"{type(e).__name__}: {e}"}


@app.get("/api/lyrics")
async def api_lyrics(artist: str = "", title: str = "", duration: float | None = None, id: int | None = None):
    try:
        if id is not None:
            rec = await asyncio.to_thread(lyrics.by_id, id)
        else:
            rec = await asyncio.to_thread(lyrics.find_lyrics, artist, title, duration)
    except OSError as e:
        raise HTTPException(502, f"lyrics lookup failed: {e}") from e
    return rec or {"synced": [], "plain": "", "missing": True}


@app.get("/api/lyrics/search")
async def api_lyrics_search(q: str):
    try:
        return await asyncio.to_thread(lyrics.search, q)
    except OSError as e:
        raise HTTPException(502, f"lyrics search failed: {e}") from e


@app.get("/", response_class=HTMLResponse)
def index():
    # Version every static asset by its modification time so a browser holding an old cached
    # copy (seen on phones) can never pair a new page with an old script.
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    for asset in sorted(WEB_DIR.glob("*.js")):
        html = html.replace(f"/static/{asset.name}\"", f"/static/{asset.name}?v={int(asset.stat().st_mtime)}\"")
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


@app.get("/annotate", response_class=HTMLResponse)
def annotate():
    return HTMLResponse((WEB_DIR / "annotate.html").read_text(encoding="utf-8"), headers={"Cache-Control": "no-cache"})


@app.get("/api/asr")
def api_asr():
    if ASR is None:
        return {"enabled": False}
    return {"enabled": True, "ready": ASR.ready, "model": ASR.model_size, "device": ASR.device, "error": ASR.error}


async def _asr_loop(sock: WebSocket, send, state: dict) -> None:
    """Every STEP_S, transcribe the newest WINDOW_S of audio and send word timestamps."""
    from .asr import STEP_S, WINDOW_S

    sr = 16000
    while True:
        await asyncio.sleep(STEP_S)
        if ASR is None or not ASR.ready or state["total"] < sr * 2:
            continue
        audio = np.concatenate(state["ring"])[-int(WINDOW_S * sr):] if state["ring"] else np.zeros(0, np.float32)
        t1 = state["total"] / sr
        t0 = t1 - len(audio) / sr
        if float(np.sqrt(np.mean(audio**2))) < 1e-3:
            continue  # silence: nothing to hear
        words, lang = await asyncio.to_thread(ASR.transcribe, audio, sr, state["prompt"], state["lang"], state["latin"])
        # Hindustani sung in Urdu script can't be matched to romanized lyrics (no vowels): pin Hindi.
        if state["latin"] and lang == "ur" and not state["lang"]:
            state["lang"] = "hi"
        for w in words:
            w["start"] = round(t0 + w["start"], 3)
            w["end"] = round(t0 + w["end"], 3)
        await send({"type": "words", "t0": round(t0, 3), "t1": round(t1, 3), "words": words})


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    engine = ChordStream(BUNDLE)  # one engine per client: independent state
    lock = asyncio.Lock()

    async def send(obj):
        async with lock:  # chord events and ASR results share the socket
            await sock.send_json(obj)

    await send({"type": "hello", "sample_rate": engine.sample_rate, "frame_seconds": engine.cfg.frontend.frame_seconds,
                "asr": ASR is not None})
    state = {"ring": [], "ring_len": 0, "total": 0, "prompt": None, "lang": None, "latin": False}
    asr_task = asyncio.create_task(_asr_loop(sock, send, state)) if ASR is not None else None
    try:
        while True:
            msg = await sock.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            text = msg.get("text")
            if text == "reset":
                engine.reset()
                state.update(ring=[], ring_len=0, total=0)
                continue
            if text:
                try:
                    ctl = json.loads(text)
                except ValueError:
                    continue
                if ctl.get("type") == "context":  # upcoming lyric lines -> Whisper prompt
                    if ctl.get("prompt") and ctl.get("song") != state.get("song"):  # new song: re-detect
                        state["lang"], state["song"] = None, ctl.get("song")
                    state["prompt"] = (ctl.get("prompt") or "")[-600:] or None
                    state["latin"] = bool(ctl.get("latin"))
                continue
            data = msg.get("bytes")
            if not data:
                continue
            audio = np.frombuffer(data, dtype="<f4")
            if asr_task is not None:  # keep the newest ~10 s for transcription
                state["ring"].append(audio)
                state["ring_len"] += len(audio)
                state["total"] += len(audio)
                while state["ring_len"] - len(state["ring"][0]) > 16000 * 10:
                    state["ring_len"] -= len(state["ring"].pop(0))
            events = await asyncio.to_thread(engine.push, audio)
            for e in events:
                if ONLY_CHANGES and not e.changed:
                    continue
                await send({"type": "chord", **e.as_dict()})
    except WebSocketDisconnect:
        pass
    finally:
        if asr_task is not None:
            asr_task.cancel()
