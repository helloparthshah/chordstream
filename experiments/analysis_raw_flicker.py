"""F08: flicker of the E33 raw-waveform models vs the CQT model, same test songs and display rules.

Live HMM path and the deployed Steady display (evidence hold k20 t0.05) on Billboard 36 + Bollywood 61.
usage: python experiments/analysis_raw_flicker.py models/v10 models/e33a_raw_init models/e33b_raw_scratch
"""
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from analysis_flicker_fixes import evidence, stats  # noqa: E402

from chordstream.config import BundleConfig
from chordstream.data.bollywood import trusted_ids_v2
from chordstream.data.dataset import split_files
from chordstream.export import load_net
from chordstream.hmm import OnlineHMM
from chordstream.model import ChordNet
from chordstream.rawfront import RawChordNet, prepare_raw, raw_features
from chordstream.train import _predict_song, predict_song

dev = torch.device("cpu")
_, val = split_files("data/cache_ext")
ok = trusted_ids_v2("align_cache_bolly3.jsonl")
bolly = [f for f in sorted(Path("data/cache_bolly3").glob("*.npz"))
         if zlib.crc32(f.stem.encode()) % 1000 < 300 and f.stem.split("__")[-1] in ok]


def audio_for(stem):
    sid = stem.split("__")[-1]
    return Path("data/billboard") / f"{sid}.flac" if stem.startswith("billboard__") else Path("data/bollywood/audio") / f"{sid}.flac"


for b in sys.argv[1:]:
    b = Path(b); raw = (b / "model_raw.pt").exists()
    if raw:
        cfg = BundleConfig.load(b / "config.json"); m = RawChordNet(ChordNet(cfg.model))
        m.load_state_dict(torch.load(b / "model_raw.pt", map_location="cpu")["state_dict"]); m.eval()
        hmm = OnlineHMM.load(Path("models/v9/hmm.npz")); dd = cfg.model.label_delay
    else:
        net, _ = load_net(b); net.eval(); hmm = OnlineHMM.load(b / "hmm.npz")
    out = []
    for name, files in (("Billboard", val), ("Bollywood", bolly)):
        live, steady = [], []
        for f in files:
            d = np.load(f); y = d["labels"].astype(np.int64)
            if raw:
                audio = np.load(prepare_raw(f.stem, audio_for(f.stem))).astype(np.float32) / 32767.0
                pr = _predict_song(m.net, raw_features(m, audio, dev), dev)
                pr = np.concatenate([pr[dd:], np.repeat(pr[-1:], dd, 0)])
            else:
                pr = predict_song(net, d["feats"].astype(np.float32), dev)
            n = min(len(pr), len(y)); pr, y = pr[:n], y[:n]; pa = hmm.filter(pr)
            live.append((pa, y)); steady.append((evidence(pa, pr, 20, 0.05), y))
        out.append(f"  {name}\n    live    {stats(live)}\n    steady  {stats(steady)}")
    print(b, *out, sep="\n", flush=True)
