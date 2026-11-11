"""What is the system playing right now? (Windows Global System Media Transport Controls)

Reads the same information as the Windows volume/media overlay: title, artist and
playback position from Spotify, YouTube Music, Chrome/Edge/Firefox media sessions, etc.
No audio fingerprinting needed. On other platforms `now_playing()` reports unavailable.
"""

from __future__ import annotations

import datetime as dt
import sys
import time

PLAYING, PAUSED = 4, 5


async def now_playing() -> dict:
    if sys.platform != "win32":
        return {"available": False, "reason": "only supported on Windows"}
    try:
        from winrt.windows.media.control import GlobalSystemMediaTransportControlsSessionManager as Manager
    except ImportError:
        return {"available": False, "reason": "install the 'desktop' extra (winrt packages)"}

    mgr = await Manager.request_async()
    sessions = [mgr.get_sessions().get_at(i) for i in range(mgr.get_sessions().size)]
    if not sessions:
        return {"available": False, "reason": "nothing is playing"}
    # Prefer a session that is actually playing; fall back to Windows' "current" session.
    playing = [s for s in sessions if s.get_playback_info().playback_status == PLAYING]
    session = playing[0] if playing else (mgr.get_current_session() or sessions[0])

    props = await session.try_get_media_properties_async()
    tl = session.get_timeline_properties()
    status = session.get_playback_info().playback_status
    pos = tl.position.total_seconds()
    end = tl.end_time.total_seconds()
    # Players update the position only occasionally; extrapolate from last_updated_time.
    if status == PLAYING and tl.last_updated_time.year > 1601:
        pos += (dt.datetime.now(dt.timezone.utc) - tl.last_updated_time).total_seconds()
    has_timeline = end > 0
    return {
        "available": True,
        "app": session.source_app_user_model_id,
        "title": props.title or "",
        "artist": props.artist or props.album_artist or "",
        "album": props.album_title or "",
        "duration": end if has_timeline else None,
        "position": min(pos, end) if has_timeline else None,
        "playing": status == PLAYING,
        "has_timeline": has_timeline,
        "server_time": time.time(),
    }
