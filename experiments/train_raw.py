"""E33: train ChordNet on a learned raw-waveform front end (chordstream.rawfront) instead of the CQT.

Same data (cache_ext_bolly3x2's training songs: Billboard train split + Bollywood v3 training songs x2),
labels, label delay (3), transition weighting, structured heads and schedule as v9. Pitch shift is done
by resampling the audio. Evaluation: the 36 Billboard and 61 Bollywood test songs with v9's HMM
(5 s / power 2; same training labels, so the same chord prior), live path, majmin view.

usage: python experiments/train_raw.py OUT [--init models/latest] [--epochs 80]
"""
import argparse
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from chordstream.config import BundleConfig
from chordstream.data.bollywood import trusted_ids_v2
from chordstream.data.dataset import split_files
from chordstream.evaluate import majmin_view
from chordstream.hmm import OnlineHMM
from chordstream.model import ChordNet, count_params
from chordstream.rawfront import RawChordNet, RawCropDataset, prepare_raw, raw_features
from chordstream.train import (WARMUP_CONTEXT, _collate, _predict_song, delay_targets, structured_loss,
                               transition_weights)
from chordstream.vocab import IGNORE

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--init", default=None)
ap.add_argument("--epochs", type=int, default=80)
ap.add_argument("--crops", type=int, default=2000)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--kernel", type=int, default=4096)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0); np.random.seed(0)

v9 = BundleConfig.load(Path("models/v9/config.json"))
mcfg = v9.model
v9.save(out / "config.json")


def audio_for(stem: str) -> Path:
    sid = stem.split("__")[1] if stem.startswith("bollytrain__") else stem.split("__")[-1]
    return Path("data/billboard") / f"{sid}.flac" if stem.startswith("billboard__") else Path("data/bollywood/audio") / f"{sid}.flac"


train_files, _ = split_files("data/cache_ext_bolly3x2")
items = []
for f in train_files:
    raw_stem = f.stem if f.stem.startswith("billboard__") else "bolly__" + f.stem.split("__")[1]
    items.append((prepare_raw(raw_stem, audio_for(f.stem)), f))
print(f"train items {len(items)}", flush=True)

net = ChordNet(mcfg)
if a.init:
    sd = torch.load(Path(a.init) / "model.pt", map_location="cpu")["state_dict"]
    own = net.state_dict()
    for k in ("head.weight", "head.bias"):
        if k in sd and sd[k].shape != own[k].shape:
            g = own[k].clone(); g[: sd[k].shape[0]] = sd[k]; sd[k] = g
    net.load_state_dict(sd, strict=False)
model = RawChordNet(net, a.kernel, v9.frontend.hop_length).to(dev)
print(f"params {count_params(model) / 1e6:.2f}M (front end {count_params(model.front) / 1e6:.2f}M)", flush=True)

crop = 2 * mcfg.context_frames
ds = RawCropDataset(items, crop, model.front.context, v9.frontend.hop_length, a.crops)
dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0, collate_fn=_collate, drop_last=True)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.02)
total = a.epochs * len(dl)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

for ep in range(1, a.epochs + 1):
    model.train(); t0, tot, n = time.time(), 0.0, 0
    for x, y, q, aux in dl:
        x, y, q, aux = (t.to(dev, non_blocking=True) for t in (x, y, q, aux))
        y, q, aux = delay_targets(y, q, aux, mcfg.label_delay)
        y[:, :WARMUP_CONTEXT] = IGNORE; aux[:, :WARMUP_CONTEXT, :2] = IGNORE; aux[:, :WARMUP_CONTEXT, 2] = -1
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            o = model.forward_all(x)
        logits = o["chord"]
        ce = F.cross_entropy(logits.float().reshape(-1, mcfg.n_classes), y.reshape(-1), ignore_index=IGNORE,
                             label_smoothing=0.05, reduction="none")
        w = transition_weights(y, 3.0, round(0.5 / v9.frontend.frame_seconds)).reshape(-1) * (y.reshape(-1) != IGNORE)
        loss = (ce * w).sum() / w.sum().clamp_min(1.0) + 0.5 * structured_loss(o, aux)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        tot, n = tot + loss.item(), n + 1
    print(f"epoch {ep:3d}  loss {tot / n:.4f}  {time.time() - t0:.0f}s", flush=True)
    torch.save({"state_dict": model.state_dict()}, out / "model_raw.pt")

# ---- evaluation (live HMM path, majmin view) on both test sets
hmm = OnlineHMM.load(Path("models/v9/hmm.npz"))
_, val = split_files("data/cache_ext")
ok = trusted_ids_v2("align_cache_bolly3.jsonl")
bolly = [f for f in sorted(Path("data/cache_bolly3").glob("*.npz")) if zlib.crc32(f.stem.encode()) % 1000 < 300 and f.stem.split("__")[-1] in ok]
for name, files in (("Billboard", val), ("Bollywood", bolly)):
    c = nn_ = 0
    for f in files:
        d = np.load(f); y = d["labels"].astype(np.int64)
        audio = np.load(prepare_raw(f.stem, audio_for(f.stem))).astype(np.float32) / 32767.0
        feats = raw_features(model, audio, dev)
        probs = _predict_song(model.net, feats, dev)
        dd = mcfg.label_delay
        probs = np.concatenate([probs[dd:], np.repeat(probs[-1:], dd, 0)])
        n = min(len(probs), len(y))
        p2, y2 = majmin_view(hmm.filter(probs[:n]), y[:n]); m = y2 != IGNORE
        c += (p2[m] == y2[m]).sum(); nn_ += m.sum()
    print(f"{name}: majmin {c / nn_:.3f}", flush=True)
sys.exit(0)
