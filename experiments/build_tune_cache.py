"""E35: detuned copies of the training songs (Billboard train split + Bollywood v3 training songs).

v11 collapses when a recording sits more than ~25 cents sharp of A440 (F11). Each training song gets
two copies: one 15-45 cents flat, one 15-45 cents sharp. Detuning the audio by c cents is computed
exactly by analysing the original audio with the CQT reference shifted the other way
(fmin * 2^(-c/1200)). Labels are unchanged (within +-50 cents the chord names don't change).
Output names start with "tunex__" (train-only in split_files). Test songs are never used.

usage: python experiments/build_tune_cache.py [out=data/cache_tune]
"""
import dataclasses
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

from chordstream.config import FrontendConfig
from chordstream.cqt import offline_cqt
from chordstream.data.bollywood import trusted_ids_v2
from chordstream.data.dataset import split_files
from chordstream.rawfront import prepare_raw

out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/cache_tune"); out.mkdir(parents=True, exist_ok=True)
cfg = FrontendConfig(); dev = "cuda" if torch.cuda.is_available() else "cpu"
train, _ = split_files("data/cache_ext")
jobs = [(f, Path("data/billboard") / f"{f.stem.split('__')[-1]}.flac") for f in train if f.stem.startswith("billboard__")]
ok = trusted_ids_v2("align_cache_bolly3.jsonl")
jobs += [(f, Path("data/bollywood/audio") / f"{f.stem.split('__')[-1]}.flac") for f in sorted(Path("data/cache_bolly3").glob("*.npz"))
         if zlib.crc32(f.stem.encode()) % 1000 >= 300 and f.stem.split("__")[-1] in ok]
print(len(jobs), "songs", flush=True)
for i, (src, audio_path) in enumerate(jobs):
    rng = np.random.default_rng(zlib.crc32(src.stem.encode()) + 35)
    audio = None
    for sign, tag in ((-1, "flat"), (1, "sharp")):
        dst = out / f"tunex__{tag}__{src.stem}.npz"
        if dst.exists():
            continue
        if audio is None:
            audio = np.load(prepare_raw(src.stem, audio_path)).astype(np.float32) / 32767.0
        cents = sign * rng.uniform(15, 45)
        feats = offline_cqt(audio, dataclasses.replace(cfg, fmin=cfg.fmin * 2 ** (-cents / 1200)), dev)
        d = np.load(src); n = min(len(feats), len(d["labels"]))
        np.savez_compressed(dst, feats=feats[:n].astype(np.float16), cents=np.float32(cents),
                            **{k: d[k][:n] for k in d.files if k != "feats"})
    if i % 50 == 0:
        print(i, src.stem, flush=True)
print("done", flush=True)
