"""Sung-word timestamps for lyric alignment (faster-whisper).

Every few seconds the last ~8 s of the stream are transcribed with word timestamps. The
client sends the lyric lines it expects next; they are passed as Whisper's prompt, which
strongly biases recognition toward the real lyrics. The client then aligns the recognised
words to the lyric words (see web/sheet.js), so a misheard word only costs one anchor.

Optional: enable with `chordstream serve --asr small` (needs the `asr` extra; GPU recommended).
"""

from __future__ import annotations

import re
import threading
import unicodedata

import numpy as np

WINDOW_S = 8.0
STEP_S = 2.0
MIN_PROB = 0.35  # drop low-confidence words (typically cut off at the window edge, or noise)


_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_DEV_CONSONANT = re.compile(r"[क-हक़-य़]$")


def romanize(word: str) -> str:
    """Latin spelling of a recognised word, so it can match romanized lyrics.

    Devanagari -> IAST with Hindi schwa deletion at word end ("दिल" -> "dil", not "dila").
    Other scripts -> unidecode. The client compares with a phonetic key, so the remaining
    spelling differences (aa/ā, ee/ī, th/t, ...) do not matter.
    """
    if word.isascii():
        return word
    if _DEVANAGARI.search(word):
        from indic_transliteration import sanscript

        core = re.sub(r"[^ऀ-ॿ]", "", word)
        out = sanscript.transliterate(core, sanscript.DEVANAGARI, sanscript.IAST)
        if _DEV_CONSONANT.search(core) and out.endswith("a"):
            out = out[:-1]
        return unicodedata.normalize("NFC", out)
    from unidecode import unidecode

    return unidecode(word)


class VocalTranscriber:
    def __init__(self, model_size: str = "small", device: str = "auto"):
        self.model_size, self.device = model_size, device
        self.model = None
        self.error: str | None = None
        self._lock = threading.Lock()
        self._loaded = threading.Event()
        threading.Thread(target=self._load, daemon=True).start()  # download/load off the request path

    def _load(self) -> None:
        try:
            try:
                import torch  # noqa: F401  (puts torch's bundled cuDNN/cuBLAS DLLs on the search path)

                cuda = torch.cuda.is_available()
            except ImportError:
                cuda = False
            from faster_whisper import WhisperModel

            dev = self.device if self.device != "auto" else ("cuda" if cuda else "cpu")
            self.model = WhisperModel(self.model_size, device=dev, compute_type="float16" if dev == "cuda" else "int8")
            self.device = dev
        except Exception as e:  # missing package / no GPU libs: the app keeps working without ASR
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self._loaded.set()

    @property
    def ready(self) -> bool:
        return self.model is not None

    def transcribe(self, audio: np.ndarray, sr: int, prompt: str | None, language: str | None,
                   romanize_words: bool = False) -> tuple[list[dict], str | None]:
        """(words as [{w, start, end, p[, rom]}] relative to the start of `audio`, detected language)."""
        if not self.ready:
            return [], None
        if sr != 16000:
            raise ValueError("expects 16 kHz audio")
        with self._lock:  # one GPU, many clients
            segments, info = self.model.transcribe(
                audio.astype(np.float32), language=language or None, initial_prompt=prompt or None,
                word_timestamps=True, beam_size=1, temperature=0.0, condition_on_previous_text=False,
                vad_filter=False, no_speech_threshold=0.6,
            )
            words = []
            for seg in segments:
                if seg.no_speech_prob > 0.7:
                    continue
                for w in seg.words or []:
                    if w.probability >= MIN_PROB and w.word.strip():
                        words.append({"w": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3),
                                      "p": round(w.probability, 3)})
        # Whisper sometimes "hears" the prompt during instrumentals; reject a verbatim echo.
        if prompt and len(words) > 4:
            heard = " ".join(w["w"] for w in words).lower()
            if heard.strip(" .,!?") in prompt.lower():
                avg_p = sum(w["p"] for w in words) / len(words)
                if avg_p < 0.6:
                    return [], info.language
        if romanize_words:
            for w in words:
                w["rom"] = romanize(w["w"])
        return words, info.language
