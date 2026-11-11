"""Training loop. Produces a model bundle directory:
    config.json  model.pt  hmm.npz   (+ model.onnx after `chordstream export`)
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import BundleConfig, FrontendConfig, ModelConfig
from .data.dataset import CropDataset, split_files
from .data.ug import load_progressions
from .hmm import OnlineHMM, build_transition, change_matrix_from_progressions
from .model import ChordNet, count_params
from .vocab import IGNORE, N_CLASSES

WARMUP_CONTEXT = 16  # frames at the start of each crop that are context-only (no loss)


def _collate(batch):
    x = torch.from_numpy(np.stack([b[0] for b in batch]))
    y = torch.from_numpy(np.stack([b[1] for b in batch]))
    return x, y


@torch.no_grad()
def predict_song(net: ChordNet, feats: np.ndarray, device, chunk: int = 1024) -> np.ndarray:
    """Exact whole-song causal posteriors, computed in overlapping chunks.

    Each output's dependency cone spans n_layers * (context_frames - 1) + CNN receptive
    field frames, so chunks overlapping by that much reproduce the streaming result.
    """
    net.eval()
    ctx = net.cfg.n_layers * net.cfg.context_frames + 16
    chunk = max(chunk, 2 * ctx)
    x = torch.from_numpy(feats.astype(np.float32)).to(device)
    T = len(feats)
    probs = np.zeros((T, N_CLASSES), np.float32)
    for s in range(0, T, chunk - ctx):
        a = max(0, s - ctx)
        e = min(T, a + chunk)
        p = torch.softmax(net(x[None, a:e]).float(), -1)[0, s - a :].cpu().numpy()
        probs[s : s + len(p)] = p
        if e == T:
            break
    return probs


def evaluate(net: ChordNet, files, device, hmms: list[OnlineHMM] | None = None) -> dict:
    hmms = hmms or []
    correct, total, hmm_correct = 0, 0, np.zeros(len(hmms))
    for f in files:
        d = np.load(f)
        y = d["labels"].astype(np.int64)
        probs = predict_song(net, d["feats"], device)
        valid = y != IGNORE
        correct += int((probs.argmax(1)[valid] == y[valid]).sum())
        total += int(valid.sum())
        for i, h in enumerate(hmms):
            hmm_correct[i] += int((h.filter(probs)[valid] == y[valid]).sum())
    out = {"frame_acc": correct / max(1, total)}
    if hmms:
        out["hmm_accs"] = hmm_correct / max(1, total)
        out["hmm_acc"] = float(out["hmm_accs"].max())
    return out


EMISSION_POWERS = (1.0, 2.0, 3.0)


def fit_hmm(train_files, frontend: FrontendConfig, progressions_jsonl: str | None) -> list[OnlineHMM]:
    """Candidate HMMs (one per emission power); the best on validation is kept."""
    counts = np.ones(N_CLASSES)
    seqs, durations = [], []
    for f in train_files:
        y = np.load(f)["labels"].astype(np.int64)
        counts += np.bincount(y[y >= 0], minlength=N_CLASSES)
        change_pts = np.flatnonzero(np.diff(y)) + 1
        runs = np.split(y, change_pts)
        seqs.append([int(r[0]) for r in runs])
        durations += [len(r) * frontend.frame_seconds for r in runs if r[0] > 0]
    progs = load_progressions(progressions_jsonl) if progressions_jsonl else seqs
    change = change_matrix_from_progressions(progs)
    mean_dur = float(np.median(durations)) if durations else 2.0
    A = build_transition(change, max(mean_dur, 0.5), frontend.frame_seconds)
    # Uniform prior: the network already learned class frequencies; dividing them out again
    # over-boosts rare chords (measured on Billboard val: 72.9% -> 73.9%, half the flicker).
    del counts
    return [OnlineHMM(A, None, emission_power=p) for p in EMISSION_POWERS]


def train(
    cache: str,
    out: str,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 3e-4,
    crops_per_epoch: int = 4000,
    init: str | None = None,
    progressions: str | None = None,
    holdout_prefix: str | None = None,
    workers: int = 2,
    device: str | None = None,
    model_cfg: ModelConfig | None = None,
) -> Path:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if init:
        bundle = BundleConfig.load(Path(init) / "config.json")
    else:
        bundle = BundleConfig(FrontendConfig(), model_cfg or ModelConfig())
    bundle.save(out_dir / "config.json")
    mcfg, fcfg = bundle.model, bundle.frontend
    crop = 2 * mcfg.context_frames  # > window so the banded mask is exercised

    train_files, val_files = split_files(cache, holdout_prefix=holdout_prefix)
    print(f"train songs: {len(train_files)}  val songs: {len(val_files)}  device: {dev}")
    ds = CropDataset(train_files, crop, fcfg.bins_per_semitone, augment=True, crops_per_epoch=crops_per_epoch)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=_collate,
                    drop_last=True, persistent_workers=workers > 0)

    net = ChordNet(mcfg).to(dev)
    if init:
        net.load_state_dict(torch.load(Path(init) / "model.pt", map_location=dev)["state_dict"])
    print(f"params: {count_params(net) / 1e6:.2f}M")
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.02)
    total_steps = epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total_steps))))
    )
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    hmms = fit_hmm(train_files, fcfg, progressions)
    hmms[1].save(out_dir / "hmm.npz")
    best = -1.0
    for ep in range(1, epochs + 1):
        net.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            y[:, :WARMUP_CONTEXT] = IGNORE
            with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                logits = net(x)
            loss = F.cross_entropy(logits.float().reshape(-1, N_CLASSES), y.reshape(-1), ignore_index=IGNORE,
                                   label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot, n = tot + loss.item(), n + 1
        msg = f"epoch {ep:3d}  loss {tot / max(1, n):.4f}  {time.time() - t0:.0f}s"
        if val_files:
            m = evaluate(net, val_files, dev, hmms)
            best_hmm = int(np.argmax(m["hmm_accs"]))
            msg += (f"  val frame_acc {m['frame_acc']:.3f}  hmm_acc {m['hmm_acc']:.3f}"
                    f" (power {EMISSION_POWERS[best_hmm]})")
            score = m["hmm_acc"]
        else:
            score = -tot
        if score > best:
            best = score
            torch.save({"state_dict": net.state_dict()}, out_dir / "model.pt")
            if val_files:
                hmms[best_hmm].save(out_dir / "hmm.npz")
            msg += "  *saved"
        print(msg, flush=True)
    return out_dir
