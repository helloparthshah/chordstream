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


def transition_weights(y: torch.Tensor, weight: float, window: int) -> torch.Tensor:
    """Per-frame loss weights: `weight` for the `window` frames starting at each chord change.

    46% of errors were late switches (the previous chord still shown after a change), so
    those frames are up-weighted to push the causal model to commit to new chords sooner.
    """
    if weight == 1.0 or window <= 0:
        return torch.ones_like(y, dtype=torch.float32)
    change = torch.zeros_like(y, dtype=torch.float32)
    change[:, 1:] = ((y[:, 1:] != y[:, :-1]) & (y[:, 1:] != IGNORE) & (y[:, :-1] != IGNORE)).float()
    near = F.max_pool1d(F.pad(change[:, None], (window - 1, 0)), window, stride=1)[:, 0]
    return 1.0 + (weight - 1.0) * near


def _collate(batch):
    x = torch.from_numpy(np.stack([b[0] for b in batch]))
    y = torch.from_numpy(np.stack([b[1] for b in batch]))
    q = torch.from_numpy(np.stack([b[2] for b in batch]))
    aux = torch.from_numpy(np.stack([b[3] for b in batch]))
    return x, y, q, aux


def delay_targets(y: torch.Tensor, q: torch.Tensor, aux: torch.Tensor, d: int):
    """E23: shift targets d frames later in time, so output t is trained to name frame t - d."""
    if d <= 0:
        return y, q, aux
    y2 = torch.full_like(y, IGNORE)
    y2[:, d:] = y[:, :-d]
    q2 = torch.zeros_like(q)
    q2[:, d:] = q[:, :-d]
    aux2 = aux.clone()
    aux2[:, :d, :2] = IGNORE
    aux2[:, :d, 2] = -1
    aux2[:, d:] = aux[:, :-d]
    return y2, q2, aux2


def stability_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """E22: squared change of the predicted distribution between consecutive frames, counted only
    where the true chord does not change (so real changes are not discouraged)."""
    p = torch.softmax(logits.float(), -1)
    diff = (p[:, 1:] - p[:, :-1]).pow(2).sum(-1)
    same = (y[:, 1:] == y[:, :-1]) & (y[:, 1:] != IGNORE)
    same[:, :WARMUP_CONTEXT] = False
    return diff[same].mean() if same.any() else logits.sum() * 0


def structured_loss(out: dict, aux: torch.Tensor) -> torch.Tensor:
    """Root + bass cross-entropy and per-note binary cross-entropy (unknown targets skipped)."""
    root, bass, notes = aux[..., 0], aux[..., 1], aux[..., 2]
    loss = F.cross_entropy(out["root"].float().reshape(-1, 13), root.reshape(-1), ignore_index=IGNORE)
    loss = loss + F.cross_entropy(out["bass"].float().reshape(-1, 13), bass.reshape(-1), ignore_index=IGNORE)
    known = notes >= 0
    if known.any():
        bits = ((notes[known].unsqueeze(-1) >> torch.arange(12, device=notes.device)) & 1).float()
        loss = loss + F.binary_cross_entropy_with_logits(out["notes"].float()[known], bits)
    return loss


@torch.no_grad()
def predict_song(net: ChordNet, feats: np.ndarray, device, chunk: int = 1024, head: str = "chord") -> np.ndarray:
    """Posteriors aligned to the frames they describe (a label-delay model's output t is frame t - d).
    head="sheet" reads the E26 sheet head instead (delay sheet_delay)."""
    probs = _predict_song(net, feats, device, chunk, head)
    d = getattr(net.cfg, "sheet_delay" if head == "sheet" else "label_delay", 0)
    if d > 0:
        probs = np.concatenate([probs[d:], np.repeat(probs[-1:], d, 0)])
    return probs


@torch.no_grad()
def _predict_song(net: ChordNet, feats: np.ndarray, device, chunk: int = 1024, head: str = "chord") -> np.ndarray:
    """Exact whole-song causal posteriors, computed in overlapping chunks.

    Each output's dependency cone spans n_layers * (context_frames - 1) + CNN receptive
    field frames, so chunks overlapping by that much reproduce the streaming result.
    """
    net.eval()
    fwd = net if head == "chord" else (lambda x: net.sheet_head(net.features(x)))
    ctx = net.cfg.n_layers * net.cfg.context_frames + 16
    if net.cfg.bidirectional:  # needs context on both sides: keep only each chunk's centre
        chunk = max(chunk, 4 * ctx)
        x = torch.from_numpy(feats.astype(np.float32)).to(device)
        T = len(feats)
        probs = np.zeros((T, net.cfg.n_classes), np.float32)
        step = chunk - 2 * ctx
        for s in range(0, T, step):
            a, e = max(0, s - ctx), min(T, s + step + ctx)
            p = torch.softmax(fwd(x[None, a:e]).float(), -1)[0].cpu().numpy()
            probs[s : min(T, s + step)] = p[s - a : s - a + min(step, T - s)]
        return probs
    chunk = max(chunk, 2 * ctx)
    x = torch.from_numpy(feats.astype(np.float32)).to(device)
    T = len(feats)
    probs = np.zeros((T, net.cfg.n_classes), np.float32)
    for s in range(0, T, chunk - ctx):
        a = max(0, s - ctx)
        e = min(T, a + chunk)
        p = torch.softmax(fwd(x[None, a:e]).float(), -1)[0, s - a :].cpu().numpy()
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


def fit_hmm(train_files, frontend: FrontendConfig, progressions_jsonl: str | None,
            n_classes: int = N_CLASSES) -> list[OnlineHMM]:
    """Candidate HMMs (one per emission power); the best on validation is kept."""
    counts = np.ones(n_classes)
    seqs, durations = [], []
    for f in train_files:
        y = np.load(f)["labels"].astype(np.int64)
        counts += np.bincount(y[y >= 0], minlength=n_classes)
        change_pts = np.flatnonzero(np.diff(y)) + 1
        runs = np.split(y, change_pts)
        seqs.append([int(r[0]) for r in runs])
        durations += [len(r) * frontend.frame_seconds for r in runs if r[0] > 0]
    progs = load_progressions(progressions_jsonl) if progressions_jsonl and n_classes == N_CLASSES else seqs
    change = change_matrix_from_progressions(progs, n_classes=n_classes)
    mean_dur = float(np.median(durations)) if durations else 2.0
    A = build_transition(change, max(mean_dur, 0.5), frontend.frame_seconds, n_classes)
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
    transition_weight: float = 1.0,
    transition_window_s: float = 0.5,
    seed: int | None = 0,
    frontend_cfg: FrontendConfig | None = None,
    eval_every: int = 1,
    bidirectional: bool = False,
    distill_weight: float = 1.0,
    aux_weight: float = 0.5,
    stability_weight: float = 0.0,
    bass_dropout: float = 0.0,
    random_eq: float = 0.0,
) -> Path:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if init:
        bundle = BundleConfig.load(Path(init) / "config.json")
        if model_cfg is not None:  # vocabulary/heads may change on top of an init; architecture may not
            bundle.model.n_classes = model_cfg.n_classes
            bundle.model.structured = model_cfg.structured
            bundle.model.key_input = model_cfg.key_input
            bundle.model.label_delay = model_cfg.label_delay
            bundle.model.sheet_delay = model_cfg.sheet_delay
            bundle.model.context_frames = model_cfg.context_frames  # E34: ALiBi, so the band can change
    else:
        bundle = BundleConfig(frontend_cfg or FrontendConfig(), model_cfg or ModelConfig())
    if bidirectional:  # offline teacher (weights from a causal init transfer: same shapes)
        bundle.model.bidirectional = True
    bundle.save(out_dir / "config.json")
    mcfg, fcfg = bundle.model, bundle.frontend
    crop = 2 * mcfg.context_frames  # > window so the banded mask is exercised

    train_files, val_files = split_files(cache, holdout_prefix=holdout_prefix)
    print(f"train songs: {len(train_files)}  val songs: {len(val_files)}  device: {dev}")
    ds = CropDataset(train_files, crop, fcfg.bins_per_semitone, augment=True, crops_per_epoch=crops_per_epoch,
                     n_classes=mcfg.n_classes, bass_dropout=bass_dropout, random_eq=random_eq)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=_collate,
                    drop_last=True, persistent_workers=workers > 0)

    net = ChordNet(mcfg).to(dev)
    if init:
        sd = torch.load(Path(init) / "model.pt", map_location=dev)["state_dict"]
        own = net.state_dict()
        for k in ("head.weight", "head.bias"):  # e.g. 25-class init -> 109-class model: first rows coincide
            if k in sd and sd[k].shape != own[k].shape:
                grown = own[k].clone()
                grown[: sd[k].shape[0]] = sd[k]
                sd[k] = grown
        missing, _ = net.load_state_dict(sd, strict=False)
        if missing:
            print(f"initialised fresh: {sorted({m.split('.')[0] for m in missing})}")
    print(f"params: {count_params(net) / 1e6:.2f}M")
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.02)
    total_steps = epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total_steps))))
    )
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    hmms = fit_hmm(train_files, fcfg, progressions, mcfg.n_classes)
    hmms[1].save(out_dir / "hmm.npz")
    best = -1.0
    for ep in range(1, epochs + 1):
        net.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y, q, aux in dl:
            x, y, q, aux = (t.to(dev, non_blocking=True) for t in (x, y, q, aux))
            y_sheet = delay_targets(y, q, aux, mcfg.sheet_delay)[0] if mcfg.sheet_delay else None
            y, q, aux = delay_targets(y, q, aux, mcfg.label_delay)
            y[:, :WARMUP_CONTEXT] = IGNORE
            aux[:, :WARMUP_CONTEXT, :2] = IGNORE
            aux[:, :WARMUP_CONTEXT, 2] = -1
            with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                out = net.forward_all(x)
            logits = out["chord"]
            ce = F.cross_entropy(logits.float().reshape(-1, mcfg.n_classes), y.reshape(-1), ignore_index=IGNORE,
                                 label_smoothing=0.05, reduction="none")
            w = transition_weights(y, transition_weight, round(transition_window_s / fcfg.frame_seconds)).reshape(-1)
            w = w * (y.reshape(-1) != IGNORE)
            loss = (ce * w).sum() / w.sum().clamp_min(1.0)
            if y_sheet is not None:  # E26: the long-delay sheet head, same weighted loss
                y_sheet[:, :WARMUP_CONTEXT] = IGNORE
                ce2 = F.cross_entropy(out["sheet"].float().reshape(-1, mcfg.n_classes), y_sheet.reshape(-1),
                                      ignore_index=IGNORE, label_smoothing=0.05, reduction="none")
                w2 = transition_weights(y_sheet, transition_weight,
                                        round(transition_window_s / fcfg.frame_seconds)).reshape(-1)
                w2 = w2 * (y_sheet.reshape(-1) != IGNORE)
                loss = loss + (ce2 * w2).sum() / w2.sum().clamp_min(1.0)
            # Distillation: match the offline teacher's posteriors where they exist
            # (unlabelled songs have only these; labelled songs get both).
            has_q = q.sum(-1) > 0
            has_q[:, :WARMUP_CONTEXT] = False
            if distill_weight > 0 and has_q.any():
                logp = F.log_softmax(logits.float(), -1)
                kl = (q * (torch.log(q.clamp_min(1e-8)) - logp)).sum(-1)
                loss = loss + distill_weight * kl[has_q].mean()
            if stability_weight > 0:
                loss = loss + stability_weight * stability_loss(logits, y)
            if mcfg.structured and aux_weight > 0:
                loss = loss + aux_weight * structured_loss(out, aux)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot, n = tot + loss.item(), n + 1
        msg = f"epoch {ep:3d}  loss {tot / max(1, n):.4f}  {time.time() - t0:.0f}s"
        if val_files and ep % eval_every and ep != epochs:  # scoring every song is slower than training
            print(msg, flush=True)
            continue
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
