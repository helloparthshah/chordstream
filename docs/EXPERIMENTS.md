# Experiment log

Every experiment run on chordstream, in order: what was tried, how, what happened and what was
decided. Each entry is reproducible from the command shown. Batch scripts are in
[`experiments/`](../experiments/). Model bundles live in `models/` (git-ignored); the ones in
use are published as [GitHub Releases](https://github.com/helloparthshah/chordstream/releases).

**New experiments:** add an entry under [Queue](#queue) before running, then move it into the
log with its result. One entry per question; record the decision even when the answer is "no gain".

## How results are measured

All numbers come from `chordstream evaluate` (or the same functions in `chordstream.evaluate`).
It is streaming-exact: posteriors from the exact chunked forward pass, then the causal HMM
frame by frame, exactly as the live engine would run.

| Metric | Meaning |
|---|---|
| **acc** | Frame accuracy after the HMM, majmin vocabulary, "X" frames excluded |
| raw | Frame accuracy of the network alone (argmax, no HMM) |
| changes | Predicted chord changes ÷ reference changes (1.0 = no over-segmentation, i.e. no flicker) |
| change-F | F-measure of predicted vs reference change points within ±0.25 s |
| lag | Median signed timing of detected changes (positive = late) |

**Test sets.** The held-out split is a stable hash of each song id (`data/dataset.split_files`,
about 10%), so a song is held out forever. The set grew with the data, so numbers from
different sizes are **not** directly comparable:

| Name | Songs | Used from |
|---|---|---|
| original 10 | 10 (`evaluate.ORIGINAL_VAL`) | E03 onward; the one set comparable across every experiment |
| held-out 19 / 25 / 37 | grows with the fetched data (207 / 291 / 394 songs) | E08–E13 |
| clean held-out 36 | 37 minus one excluded drifting song, cleaned labels | E15 onward |

Comparisons inside one experiment always use the same set. Differences under ~0.5 points on
25–37 songs are within noise.

## Summary

| Model | Live? | Original 10 | Larger held-out set | Changes | Lag | Entry |
|---|---|---|---|---|---|---|
| bb_ft (first fine-tune) | was live | 73.9% | – | 2.0× | +120 ms | E03–E04 |
| v2 | was live | 75.5% | 77.0% (19) | 1.9× | +80 ms | E09 |
| v4 | was live | 78.8% | 80.1% (36 clean) | 1.6× | +80 ms | E15 |
| v5 | was live ([release](https://github.com/helloparthshah/chordstream/releases/tag/model-v5)) | 78.9% | 80.8% (36 clean) | 1.6× | +80 ms | E17 |
| **v6** (extended vocab) | **live** ([release](https://github.com/helloparthshah/chordstream/releases/tag/model-v6)) | **79.9%** | **80.8% (36 clean)**; 70.3% full vocab | 1.6× | +80 ms | E18 |
| teacher (offline, sees ±5 s) | never | 82.0% | 82.9% (36 clean) | 1.04× | – | E17 |

Biggest wins: real data (E03, +28 pts over synthetic only), training budget (E13/E15, +2.4),
more data (E08, +2), distillation (E17, +0.7), transition loss (E09, lag 120 → 40–80 ms).
Things that did **not** help: beat or key priors in the HMM, a faster frontend, a bigger model
at the old learning rate, re-timing labels (for training), and data beyond ~200 songs at the
old training budget.

---

## Log

Dates are the day the experiment ran.

### E01 · Synthetic pretraining · 2026-11-08
- **Question:** can the pipeline learn chords end to end before any real data?
- **Setup:** 320 rendered songs (`chordstream synth`) from 12 built-in progressions, with random
  instruments, voicings, tempo and drums. Causal CQT → CNN → Transformer (2.1M params), 12 epochs.
- **Command:** `chordstream synth --n 300` / `features` / `train --holdout-prefix synth_val__`
- **Result:** 93.6% on 20 held-out *synthetic* songs. Live engine at ~13× real time.
- **Decision:** kept as the initialisation for every later model (`models/latest`).

### E02 · KV-cached streaming step · 2026-11-08
- **Question:** the first engine re-ran the whole 5 s window every 40 ms (30 ms/frame, p99 42 ms,
  over budget). Can it be made incremental?
- **Setup:** banded causal attention + per-layer sliding KV cache; CNN run on its 13-frame
  receptive field only. A test proves the step equals the full forward pass.
- **Result:** 3 ms/frame (p99 5 ms), identical outputs.
- **Decision:** adopted; the basis of all exported models.

### E03 · First real data (Billboard fine-tune) · 2026-11-10
- **Setup:** 86 aligned Billboard songs (76 train / 10 held-out), init from E01, lr 1e-4, 25 epochs.
- **Result:** synthetic-only model on real songs **46%** → fine-tuned **74.3%** raw.
- **Decision:** real data essential; deployed as `bb_ft`.

### E04 · HMM class prior · 2026-11-10
- **Question:** the HMM divided out training class frequencies. Does that help?
- **Result:** uniform prior 73.9% vs 72.9%, and flicker halved (4.2× → 2.0× changes).
- **Decision:** uniform prior is the default.

### E05 · Beat-aware HMM · 2026-11-11
- **Setup:** causal beat tracker (`beat.py`) modulates the HMM change probability near beats;
  grid over lag/window/suppression; also with *oracle* (annotated) beats.
- **Result:** no accuracy gain even with oracle beats (73.8%); changes 1.92× → 1.83×. The beat
  tracker itself: F 0.27 (tempo right on 6/10 songs; half-tempo or off-beat on the rest).
- **Decision:** rejected for recognition. Tracker kept for display (BPM, pulse, timeline).

### E06 · Causal key prior in the HMM · 2026-11-11
- **Setup:** key profile learned from training songs (I 32%, V 19%, IV 18%, vi 12%), causal key
  posterior, chord prior raised to strength γ.
- **Result:** 73.8% at best (γ 0.5), 71.6% at γ 3.
- **Decision:** rejected (`key.py` kept, unused).

### E07 · Error analysis · 2026-11-11
- **Finding:** 53% of errors are fifth-related (C vs F/G), 46% are *late switches* (previous chord
  still shown), 43% fall within 0.4 s of a true change; network lag +40 ms, HMM adds +80 ms.
- **Decision:** the bottleneck is switching speed, not harmony knowledge, which explains E05/E06.

### E08 · More data (207 songs) · 2026-11-11 — `models/exp_a`
- **Result:** 77.3% vs 75.6% (19 held-out); original 10: 75.8% vs 73.8%. Lag unchanged (+120 ms).
- **Decision:** keep fetching data.

### E09 · Transition-weighted loss · 2026-11-11 — `models/exp_b` → `v2`
- **Setup:** ×3 loss weight on the 0.5 s after each chord change (`--transition-weight 3`).
- **Result:** raw lag +120 → **+40 ms**, change-F 0.433 → 0.485, more flicker. With retuned HMM
  (3 s / power 1.5): 77.0% (19 held-out), change-F 0.503, lag +80 ms.
- **Decision:** adopted; deployed as **v2**.

### E10 · Transition-weight sweep (2 / 3 / 5) · 2026-11-11 — `models/exp_w{2,3,5}`
- **Script:** [`experiments/2026-11-11_weight_sweep_and_frontend.ps1`](../experiments/2026-11-11_weight_sweep_and_frontend.ps1)
- **Result (25 held-out):** 80.4% / 80.1% / 79.6%; lag +80 / **+40** / +80 ms.
- **Decision:** 2 and 3 within noise; keep 3 (lowest lag).

### E11 · Faster frontend (`--gamma 10`) · 2026-11-11 — `models/exp_g10`
- **Question:** shorter low-frequency CQT windows (0.2 s → 0.08 s) for faster switching?
- **Result (37 held-out):** 77.9% vs 78.8%, change-F 0.499 vs 0.520, lag identical (+40 ms).
- **Decision:** rejected; blurred bass costs more than the shorter window saves.

### E12 · Data scaling to 394 songs · 2026-11-11 — `models/v3`
- **Result:** tie with v2 (78.3% vs 78.4%, 37 held-out). v3 scores only **79% on its own
  training songs**, so the model under-fits.
- **Decision:** more of the same data won't help; look at capacity and training budget.

### E13 · Capacity · 2026-11-11 — `models/exp_lr3`, `models/exp_big`
- **Script:** [`experiments/2026-11-11_capacity.ps1`](../experiments/2026-11-11_capacity.ps1)
- **Result:** lr 3e-4 (was 1e-4): **79.6%** vs 78.5%, still improving at the end.
  Bigger model (256-d, 6 layers) at lr 1e-4: 78.6%, no gain.
- **Decision:** under-trained, not too small; use the higher LR with longer training.

### E14 · Training-data audit and label re-timing · 2026-11-11
- **Tool:** `chordstream audit` (offset, drift across 30 s windows, key margin, tuning, model agreement).
- **Finding:** 44 songs drift > 1 s (different edit or tempo than annotated); model accuracy
  70.6% on them vs 81.0% on clean songs. Tuning is not an issue.
- **Fix:** piecewise re-timing (`audit.realign`): median drift 0.28 → 0.08 s; 27 songs that still
  drift are excluded, leaving 367 in `data/cache_bb_fix`.

### E15 · Longer, higher-LR training, cleaned vs original labels · 2026-11-11 — `models/v4_*` → `v4`
- **Script:** [`experiments/2026-11-11_v4.ps1`](../experiments/2026-11-11_v4.ps1)
- **Setup:** lr 3e-4, 80 epochs, weight 3; once on cleaned labels, once on original labels.
- **Result (36 clean held-out):** **80.6% raw for both**; original 10: 78.8% (v3: 75.1%).
  HMM 5 s / power 1.5: 80.1%, 1.62× changes, change-F 0.606.
- **Decision:** deployed **v4**. Label cleaning doesn't change the trained model, but it makes the
  evaluation more trustworthy.

### E16 · Beat-synchronous decisions · 2026-11-11
- **Question:** decide chords per beat instead of per 40 ms frame?
- **Result (37 held-out):**

  | | acc | changes |
  |---|---|---|
  | per-frame + HMM (live) | 79.3% | 1.71× |
  | oracle beats, wait for beat end | **83.3%** | 1.22× |
  | oracle beats, causal within beat | 79.0% | 2.00× |
  | tracker beats, wait for beat end | 80.6% | 1.18× |
  | tracker beats, causal within beat | 78.6% | 1.94× |

- **Decision:** the gain comes from *waiting for the beat to end* (lookahead), not from beat
  alignment. Adopted for the chord sheet and progression only (one beat late, written after the
  fact); the live chord stays zero-lookahead. A better beat tracker is worth up to +2.7 pts there.

### E17 · Bidirectional teacher → causal student · 2026-11-11 — `models/teacher`, `models/student` → `v5`
- **Script:** [`experiments/2026-11-11_distill.ps1`](../experiments/2026-11-11_distill.ps1)
- **Setup:** teacher sees ±5 s (`--bidirectional`); its posteriors label 367 labelled songs plus
  299 unlabelled ones (alignment rejects and drifting songs); student = causal model trained on
  human labels + KL to the teacher (`--distill-weight 1`).
- **Result (36 clean held-out):** teacher **82.9%**, 1.04× changes, change-F 0.71 (the cost of zero
  lookahead is about 2 pts plus segmentation quality). Student 81.3% raw / **80.8%** (HMM 3 s /
  power 2), 1.60× changes; original 10: 78.9%. 2 ms/frame.
- **Decision:** deployed **v5** and published it as release `model-v5`.

### E18 · Extended vocabulary + structured heads · 2026-11-12 — `models/ext_teacher`, `models/ext_student` → `v6`
- **Script:** [`experiments/2026-11-12_extended_vocab.ps1`](../experiments/2026-11-12_extended_vocab.ps1)
- **Setup:** 109 classes (12 roots × maj, min, 7, maj7, m7, dim, aug, sus2, sus4 + N; the first
  25 = majmin); training-only root, bass and 12-note heads (McFee & Bello style); labels re-derived
  from full Billboard annotations with E14's re-timing (`data/cache_ext`); teacher → student as E17.
- **Data:** chord-type share: maj 64%, min 12%, 7 11%, m7 7%, maj7 3%, sus4 2.5%, dim/sus2/aug < 0.5%.
- **Success criterion:** majmin view ≥ ~80.5% (no regression vs v5), plus useful accuracy on 7ths/sus.
- **Result (36 clean held-out):** teacher 73.8% full vocabulary. Student (HMM 5 s / power 2):

  | | v5 | ext student |
  |---|---|---|
  | majmin view | 80.8% | **80.8%** |
  | majmin, original 10 | 78.9% | **79.9%** |
  | full vocabulary | 62.5% (can't output 7ths) | **70.3%** |
  | changes / lag | 1.60× / +80 ms | 1.61× / +80 ms |

  Per type (share of frames, exact chord, right root): maj 66% 84% 86% · min 10% 56% 73% ·
  7 9% 44% 86% · m7 8% 52% 75% · maj7 4% 44% 93% · sus4 3% 11% 64% · dim/aug/sus2 < 0.5% ≈0%.
- **Decision:** no majmin regression and real 7th recognition, so deployed as **v6** (release
  [`model-v6`](https://github.com/helloparthshah/chordstream/releases/tag/model-v6)). Sus4 is weak and
  dim/aug/sus2 are too rare to learn from 331 songs. More data for those qualities (or
  synthetic renders of them) is the lever.

---

## Queue

| ID | Experiment | Question | Notes |
|---|---|---|---|
| E19 | Causal harmonic/percussive separation before the CQT | Does suppressing drums help a deep model? | A/B: same recipe, features with and without HPSS (past-only median filter) |
| E20 | Key-signature input, oracle first | Upper bound of giving the Transformer the key | Inject the true key signature (from annotations) as 12 inputs. Proceed to a causal 60 s note profile only if the oracle gains ≥ 0.5 pt. Most errors are in-key (E07), so expect a small gain |
| later | Learned beat head | Better beats for E16 (up to +2.7 pts on the sheet) | Multi-task beat output trained on Billboard bar timings |
| later | Rare chord types (sus, dim, aug) | E18: under 0.5% of frames each, ≈0% recall | Synthetic renders of these qualities mixed into training, or more annotated data |
| later | More unlabelled audio for distillation | E17 used 299 extra songs; the literature uses 1000+ hours | User library or more fetches |

## Model directory registry

| `models/` | From | Status |
|---|---|---|
| `latest` | E01 | init for all models (keep) |
| `bb_ft`, `v2`, `v3`, `v4` | E03, E09, E12, E15 | superseded; kept for comparison |
| `v5` | E17 | previous live model, release `model-v5` |
| `v6` | E18 | **live**, release `model-v6` |
| `teacher` | E17 | offline teacher for majmin distillation |
| `exp_a`, `exp_b`, `exp_w2/3/5`, `exp_g10`, `synth_g10`, `exp_lr3`, `exp_big`, `synth_big`, `v4_clean`, `v4_orig`, `student` | E08–E17 | experiment outputs; safe to delete |
| `ext_teacher` | E18 | offline teacher for extended-vocabulary distillation |
| `ext_student` | E18 | same weights as `v6` (before HMM retune); safe to delete |
