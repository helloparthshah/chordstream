"""E29: room-echo copies of the training songs (Billboard train split + Bollywood v3 training songs).

Each song is re-rendered once with a random synthetic room (RT60 0.25-0.9 s), and half of them also
get the phone-mic band and pink noise at 10-25 dB SNR. The causal CQT is recomputed; labels and
auxiliary targets are copied from the source cache. Output names start with "roomx__", which
split_files treats as train-only. Test songs are never rendered.

usage: python experiments/build_room_cache.py [out=data/cache_room]
"""
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from analysis_degraded import band, fftconvolve, pink, rir  # noqa: E402
from chordstream.config import FrontendConfig  # noqa: E402
from chordstream.cqt import offline_cqt  # noqa: E402
from chordstream.data.bollywood import trusted_ids_v2  # noqa: E402
from chordstream.data.dataset import load_audio, split_files  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/cache_room")
out.mkdir(parents=True, exist_ok=True)
cfg = FrontendConfig()
dev = "cuda" if torch.cuda.is_available() else "cpu"
train, _ = split_files("data/cache_ext")
jobs = [(f, Path("data/billboard") / f"{f.stem.split('__')[-1]}.flac") for f in train if f.stem.startswith("billboard__")]
ok = trusted_ids_v2("align_cache_bolly3.jsonl")
jobs += [(f, Path("data/bollywood/audio") / f"{f.stem.split('__')[-1]}.flac") for f in sorted(Path("data/cache_bolly3").glob("*.npz"))
         if zlib.crc32(f.stem.encode()) % 1000 >= 300 and f.stem.split("__")[-1] in ok]
print(len(jobs), "songs", flush=True)
for i, (src, audio_path) in enumerate(jobs):
    dst = out / f"roomx__{src.stem}.npz"
    if dst.exists():
        continue
    rng = np.random.default_rng(zlib.crc32(src.stem.encode()) + 29)
    y = load_audio(audio_path, cfg.sample_rate)
    x = fftconvolve(y, rir(cfg.sample_rate, rng.uniform(0.25, 0.9), rng))[: len(y)]
    if rng.random() < 0.5:
        x = band(x, cfg.sample_rate, rng.uniform(100, 250), rng.uniform(5000, 8000))
        x = x + pink(len(x), rng) * x.std() / 10 ** (rng.uniform(10, 25) / 20)
    x = (x / (np.abs(x).max() + 1e-9) * 0.5).astype(np.float32)
    feats = offline_cqt(x, cfg, dev)
    d = np.load(src)
    n = min(len(feats), len(d["labels"]))
    arrays = {k: d[k][:n] for k in d.files if k != "feats"}
    np.savez_compressed(dst, feats=feats[:n].astype(np.float16), **arrays)
    if i % 50 == 0:
        print(i, src.stem, flush=True)
print("done", flush=True)
