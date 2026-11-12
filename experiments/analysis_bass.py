"""F07: how much does the model lean on the bass line?

1. Bass removed from the audio: steep high-pass at 150 Hz (most bass guitar fundamentals gone) and
   300 Hz (bass and the low half of the piano/guitar range gone), new causal CQT, live HMM path.
2. Clean audio, frames split by low-register energy (CQT bins C2-F#3): quietest 25% vs the rest.
3. Clean audio, inverted chords (labelled bass != root) vs root position.
Billboard 36 (majmin view) and Bollywood 61 (aligned labels, majmin view).

usage: python experiments/analysis_bass.py models/v9 [models/x ...]
"""
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

from chordstream.config import FrontendConfig
from chordstream.cqt import offline_cqt
from chordstream.data.bollywood import trusted_ids_v2
from chordstream.data.dataset import load_audio, split_files
from chordstream.evaluate import majmin_view
from chordstream.export import load_net
from chordstream.hmm import OnlineHMM
from chordstream.train import predict_song
from chordstream.vocab import IGNORE


def highpass(x, sr, fc):
    f = np.fft.rfftfreq(len(x), 1 / sr)
    g = np.clip((f - fc * 0.8) / (fc * 0.2), 0, 1)  # ramp 0.8 fc -> fc, steep
    return np.fft.irfft(np.fft.rfft(x) * g, len(x)).astype(np.float32)


def acc(path, y, mask=None):
    p, yy = majmin_view(path, y)
    m = yy != IGNORE
    if mask is not None:
        m &= mask[: len(m)]
    return (p[m] == yy[m]).sum(), m.sum()


def main(b):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = FrontendConfig()
    net, _ = load_net(Path(b)); net.to(dev); hmm = OnlineHMM.load(Path(b) / "hmm.npz")
    _, val = split_files("data/cache_ext")
    ok = trusted_ids_v2("align_cache_bolly3.jsonl")
    bolly = [f for f in sorted(Path("data/cache_bolly3").glob("*.npz"))
             if zlib.crc32(f.stem.encode()) % 1000 < 300 and f.stem.split("__")[-1] in ok]
    sets = {"Billboard": [(f, Path("data/billboard") / f"{f.stem.split('__')[-1]}.flac") for f in val],
            "Bollywood": [(f, Path("data/bollywood/audio") / f"{f.stem.split('__')[-1]}.flac") for f in bolly]}
    for name, items in sets.items():
        tot = {k: [0, 0] for k in ("clean", "hp150", "hp300", "low-bass frames", "other frames", "inverted", "root pos")}
        for f, ap in items:
            d = np.load(f); y = d["labels"].astype(np.int64)
            feats = d["feats"].astype(np.float32)
            path = hmm.filter(predict_song(net, feats, dev))
            for k, v in zip(("clean",), (acc(path, y),)):
                tot[k][0] += v[0]; tot[k][1] += v[1]
            low = feats[:, :36].mean(1)
            q = np.quantile(low, 0.25)
            for k, m in (("low-bass frames", low <= q), ("other frames", low > q)):
                c, n = acc(path, y, m); tot[k][0] += c; tot[k][1] += n
            if "bass" in d.files and "root" in d.files:
                r, bs = d["root"].astype(int), d["bass"].astype(int)
                valid = (r >= 0) & (r < 12) & (bs >= 0) & (bs < 12)
                for k, m in (("inverted", valid & (bs != r)), ("root pos", valid & (bs == r))):
                    c, n = acc(path, y, m); tot[k][0] += c; tot[k][1] += n
            audio = load_audio(ap, cfg.sample_rate)
            for fc in (150, 300):
                ff = offline_cqt(highpass(audio, cfg.sample_rate, fc), cfg, dev)
                n = min(len(ff), len(y))
                c, nn = acc(hmm.filter(predict_song(net, ff[:n].astype(np.float32), dev)), y[:n])
                tot[f"hp{fc}"][0] += c; tot[f"hp{fc}"][1] += nn
        print(f"{b} {name}: " + "  ".join(f"{k} {c / n:.3f}" + (f" ({n / 25 / 60:.0f} min)" if k in ("inverted",) else "")
                                         for k, (c, n) in tot.items() if n), flush=True)


if __name__ == "__main__":
    for b in sys.argv[1:]:
        main(b)
