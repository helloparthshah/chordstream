"""W06: which small speech model can find where we are in a song from the lyrics (browser vocal sync)?

Mic mode doesn't know the playback position. Task: given 8 s of a song at an unknown point and the
song's timed lyrics (LRCLIB), recognise sung words and vote for the window's start time. Correct when
within 1.5 s of the truth (lyric word times are interpolated within lines, so ~1 s is the floor).
12 windows per song containing a sung line; eval set from experiments/lyric_eval_set.py.

Engines: Whisper tiny/base/small (faster-whisper, CPU int8, language given, no prompt: position
unknown), Vosk small models with the grammar restricted to the song's lyric words.
usage: python experiments/analysis_lyric_sync.py whisper-tiny whisper-base vosk ...
"""
import glob, json, random, re, sys, time, unicodedata
from pathlib import Path

import numpy as np

from chordstream.data.dataset import load_audio

SR, WIN, TOL, PER_SONG, RATE = 16000, 8.0, 1.5, 12, 0.085
DEV = re.compile("[ऀ-ॿ]")
DEV_CONS = re.compile("[क-हक़-य़]")


def romanize(w):
    if w.isascii():
        return w
    if DEV.search(w):
        from indic_transliteration import sanscript
        core = re.sub(r"[^ऀ-ॿ]", "", w)
        out = sanscript.transliterate(core, sanscript.DEVANAGARI, sanscript.IAST)
        if DEV_CONS.search(core) and out.endswith("a"):
            out = out[:-1]
        return out
    from unidecode import unidecode
    return unidecode(w)


def norm(w):  # same phonetic key as the page used (sheet.js): folds romanization variants
    k = unicodedata.normalize("NFKD", romanize(w).lower())
    k = "".join(c for c in k if not unicodedata.combining(c))
    k = re.sub(r"[^a-z0-9]", "", k)
    for a, b in ((r"ee", "i"), (r"oo", "u"), (r"([kgcjtdpbrs])h", r"\1"), (r"ph", "f"), (r"w", "v"), (r"z", "j"),
                 (r"q", "k"), (r"[ae]i", "e"), (r"[ao]u", "o"), (r"(.)\1+", r"\1")):
        k = re.sub(a, b, k)
    return re.sub(r"([aeiou])[mn]$", r"\1", k) if len(k) > 3 else k


def sim(a, b):
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if abs(len(a) - len(b)) > max(len(a), len(b)) / 2:
        return 0.0
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
    r = 1 - d[len(b)] / max(len(a), len(b))
    return r if r >= 0.75 else 0.0


def lyric_tokens(lrc):
    lines = []
    for raw in lrc.splitlines():
        m = re.match(r"^\[(\d+):(\d+(?:\.\d+)?)\](.*)$", raw.strip())
        if m:
            lines.append((int(m[1]) * 60 + float(m[2]), m[3].strip()))
    lines.sort()
    toks = []
    for i, (t, text) in enumerate(lines):
        nxt = lines[i + 1][0] if i + 1 < len(lines) else t + 10
        words, c, n = text.split(), 0, len(text)
        rate = min(RATE, max(0.03, (min(nxt, t + n * RATE + 0.4) - t) / max(1, n)))
        for w in words:
            k = norm(w)
            if len(k) >= 2:
                toks.append((t + c * rate, k, w))
            c += len(w) + 1
    return lines, toks


def locate(words, toks):
    """words: [(t_rel, key)]. Each matched pair votes window start = lyric time - t_rel; best 2 s cluster."""
    votes = []
    for i, (tr, k) in enumerate(words):
        if len(k) < 2:
            continue
        for lt, lk, _ in toks:
            s = sim(k, lk)
            if s:
                votes.append((lt - tr, s, i))
    if not votes:
        return None, 0
    votes.sort()
    best, best_n, best_o = 0, 0, None
    for a in range(len(votes)):
        per = {}
        for o, s, i in votes[a:]:
            if o - votes[a][0] > 2.0:
                break
            per[i] = max(per.get(i, 0), s)
        score = sum(per.values())
        if score > best:
            members = [o for o, s, i in votes[a:] if o - votes[a][0] <= 2.0]
            best, best_n, best_o = score, len(per), float(np.median(members))
    return best_o, best_n


class Whisper:
    def __init__(self, size):
        from faster_whisper import WhisperModel
        self.m = WhisperModel(size, device="cpu", compute_type="int8", cpu_threads=4)

    def words(self, audio, lang, toks):
        segs, _ = self.m.transcribe(audio, language=lang, word_timestamps=True, beam_size=1,
                                    condition_on_previous_text=False, vad_filter=False)
        return [(w.start, norm(w.word.strip())) for s in segs for w in (s.words or [])]


class Vosk:
    MODELS = {"en": "vosk-model-small-en-us-0.15", "hi": "vosk-model-small-hi-0.22"}

    def __init__(self):
        import vosk
        vosk.SetLogLevel(-1)
        self.vosk, self.models = vosk, {}

    def words(self, audio, lang, toks):
        if lang not in self.models:
            self.models[lang] = self.vosk.Model(model_name=self.MODELS[lang])
        vocab = sorted({re.sub(r"[^\w']", "", w.lower()) for _, _, w in toks} - {""})
        rec = self.vosk.KaldiRecognizer(self.models[lang], SR, json.dumps(vocab + ["[unk]"], ensure_ascii=False))
        rec.SetWords(True)
        rec.AcceptWaveform((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
        res = json.loads(rec.FinalResult()).get("result", [])
        return [(r["start"], norm(r["word"])) for r in res if r["word"] != "[unk]"]


def engine(name):
    return Vosk() if name == "vosk" else Whisper(name.split("-", 1)[1])


def load_songs():
    return [json.loads(Path(p).read_text(encoding="utf-8")) | {"id": Path(p).stem} for p in sorted(glob.glob("data/lyric_eval/*.json"))]


def windows(s, y):
    """The eval windows of a song: [t0] (deterministic per song)."""
    lines, _ = lyric_tokens(s["lrc"])
    starts = [t for t, text in lines if text and t + 2 < len(y) / SR]
    rng = random.Random(s["id"])
    return [max(0.0, rng.choice(starts) - rng.uniform(0.5, WIN - 2.5)) for _ in range(min(PER_SONG, len(starts)))]


def main():
    songs = load_songs()
    cache = {}
    for name in sys.argv[1:]:
        eng = engine(name)
        stats = {"en": [0, 0, 0, 0, 0.0], "hi": [0, 0, 0, 0, 0.0]}  # windows, correct, wrong, none, seconds
        for s in songs:
            _, toks = lyric_tokens(s["lrc"])
            if s["id"] not in cache:
                cache[s["id"]] = load_audio(s["audio"], SR)
            y = cache[s["id"]]
            for t0 in windows(s, y):
                win = y[int(t0 * SR): int((t0 + WIN) * SR)].astype(np.float32)
                tic = time.perf_counter()
                words = eng.words(win, s["lang"], toks)
                st = stats[s["lang"]]
                st[4] += time.perf_counter() - tic
                score(st, locate(words, toks), t0)
        report(name, stats)


def score(st, located, t0):
    est, n = located
    st[0] += 1
    if est is None or n < 2:
        st[3] += 1
    elif abs(est - t0) <= TOL:
        st[1] += 1
    else:
        st[2] += 1


def report(name, stats):
    for lang, (n, ok, bad, none, secs) in stats.items():
        if n:
            print(f"{name:14s} {lang}  windows {n:4d}  correct {ok / n:5.1%}  wrong {bad / n:5.1%}  no lock {none / n:5.1%}"
                  f"  {secs / n:5.2f} s/window", flush=True)


if __name__ == "__main__":
    main()
