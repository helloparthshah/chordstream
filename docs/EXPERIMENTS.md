# Experiment log

Every experiment run on chordstream, in order: what was tried, how, what happened and what was
decided. Each entry is reproducible from the command shown. Batch scripts are in
[`experiments/`](../experiments/). Model bundles live in `models/` (git-ignored); the ones in
use are published as [GitHub Releases](https://github.com/helloparthshah/rubato/releases).

**New experiments:** add an entry under [Queue](#queue) before running, then move it into the
log with its result. One entry per question; record the decision even when the answer is "no gain".

**Policy (from 2026-11-12): no distillation in training.** New models train on human labels only, so
results reflect the method itself rather than inheriting a teacher's errors. The bidirectional teacher
is kept only as a measuring stick (the cost of zero lookahead). E17/E18's distilled models stay
documented, but new baselines are the non-distilled ones (`v4_clean`, E24).

## How results are measured

**Selection rule (from 2026-11-12): overall accuracy across both test sets.** Live models are chosen
by majmin accuracy over the 36 clean Billboard songs *and* the Bollywood test songs (E21): the mean
of the two domains, with the per-song pooled number alongside. A Billboard drop is acceptable if the
overall number rises. Bollywood accuracy is measured against sheet-aligned labels (noisy), so the
aligner-independent "in sheet" metric must not fall either.

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
| v5 | was live ([release](https://github.com/helloparthshah/rubato/releases/tag/model-v5)) | 78.9% | 80.8% (36 clean) | 1.6× | +80 ms | E17 |
| v6 (extended vocab) | was live ([release](https://github.com/helloparthshah/rubato/releases/tag/model-v6)) | 79.9% | 80.8% (36 clean); 70.3% full vocab | 1.6× | +80 ms | E18 |
| teacher (offline, sees ±5 s) | never | 82.0% | 82.9% (36 clean) | 1.04× | – | E17 |
| E23b (label delay 240 ms, human labels) | candidate | 83.0% | 83.9% (36 clean) | 1.44× | +240 ms display | E23 |
| **v12** (v10 recipe + 32 s attention context) | **live** ([release](https://github.com/helloparthshah/rubato/releases/tag/model-v12)) | – | 82.5% majmin, 72.8% full (36 clean); 80.8 / 73.8% without bass < 150 / 300 Hz; inverted 52.8%; **Bollywood 82.4% / 79.8%** (61), 71.6 / 56.5% without bass | 1.36× | 120 ms display | E38 |
| v11 (v10 + 16 s attention context) | superseded ([release](https://github.com/helloparthshah/rubato/releases/tag/model-v11)) | – | 81.5% majmin, 71.0% full (36 clean); 79.6 / 72.6% without bass < 150 / 300 Hz; inverted 49.7%; **Bollywood 81.0% / 78.4%** (61), 70.5 / 56.9% without bass | 1.56× | 120 ms display | E34b |
| v10 (v9 + random EQ + bass dropout) | superseded ([release](https://github.com/helloparthshah/rubato/releases/tag/model-v10)) | – | 79.5% majmin, 66.8% full (36 clean); 78.1 / 71.0% without bass < 150 / 300 Hz; **Bollywood 79.1% / 76.5%** (61), 71.9 / 58.5% without bass | 1.68× | 120 ms display | E32b |
| v9 (v7 recipe + 152 Bollywood songs ×2, cleaned sheet labels) | superseded ([release](https://github.com/helloparthshah/rubato/releases/tag/model-v9)) | 77.1% majmin | 80.2% majmin, 67.7% full (36 clean); **Bollywood 80.0% / 77.4%** (61) | 1.62× | 120 ms display | E28b |
| v8 (v7 + 47 Bollywood songs, cleaned sheet labels) | superseded ([release](https://github.com/helloparthshah/rubato/releases/tag/model-v8)) | 79.6% majmin | 81.8% majmin, 70.5% full (36 clean); **Bollywood 80.0% / 78.0%** (23) | 1.56× | 120 ms display | E21e |
| E24 (extended, human labels) | baseline | 80.9% majmin | 81.0% majmin, 70.7% full (36 clean) | 1.71× | +80 ms | E24 |
| v7 (extended, 120 ms label delay, human labels) | was live ([release](https://github.com/helloparthshah/rubato/releases/tag/model-v7)) | **83.1%** majmin | **83.3% majmin, 73.0% full (36 clean)** | 1.48× | 120 ms display | E25 |

Biggest wins: real data (E03, +28 pts over synthetic only), 16 → 32 s attention context (E34 + E38, +3.1 overall, +9 on inversions), Bollywood songs with cleaned sheet-aligned labels (E21, +11 pts on Bollywood, +4.8 overall), training budget (E13/E15, +2.4),
a trained-in 120 ms label delay (E23/E25, +2.5 majmin and fewer flips), more data (E08, +2),
transition loss (E09, lag 120 → 40–80 ms), and a 480 ms hold for the chord sheet (F03, changes
1.6× → 1.03×). Distillation gave +0.7 (E17) but is no longer used (policy above; E24 showed no
gain on the extended model).
Things that did **not** help: an early-switch penalty (E36), detuned training copies (E35, fixes the tuning cliff but costs Bollywood), 36 bins/octave (E37 pilot), beat or key priors in the HMM, a faster frontend, a bigger model
at the old learning rate, re-timing labels (for training), data beyond ~200 songs at the old
training budget, drum suppression (E19), a learned raw-waveform front end instead of the CQT (E33, −4), re-tuning HMM stickiness for flicker (F09), a stability loss (E22), draft/verify "speculative"
display (F02), a second longer-delay head (E26), and per-beat averaging once the model has
look-ahead (F03).

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
- **Significance check (paired, per song, v4 vs v5 on the same 36 songs):** raw +0.9 pt, 95% CI
  [+0.4, +1.4] (song bootstrap), better on 24 / worse on 9 songs, sign test p = 0.014. Smoothed
  (as deployed) +0.7 pt, CI [+0.1, +1.4], 23 / 11, p = 0.058. Real but small; the student recovers
  about a third of the teacher's lead. One seed per model, so seed variance isn't included.

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
  [`model-v6`](https://github.com/helloparthshah/rubato/releases/tag/model-v6)). Sus4 is weak and
  dim/aug/sus2 are too rare to learn from 331 songs. More data for those qualities (or
  synthetic renders of them) is the lever.

### E19 · Causal harmonic/percussive separation (HPSS) · 2026-11-12 — `models/e19_hpss`
- **Script:** [`experiments/2026-11-12_hpss_and_key_input.ps1`](../experiments/2026-11-12_hpss_and_key_input.ps1)
- **Question:** does suppressing drums before the network help a deep model?
- **Setup:** `data/variants.hpss_causal`: soft mask H²/(H²+P²) per frame, H = median over the
  current + 4 previous frames, P = median over 17 neighbouring bins; past frames only. Same recipe
  as E15's `v4_clean` (baseline: 80.6% raw on 36 clean held-out).
- **Risk:** the past-only time median may delay new notes; watch lag as well as accuracy.
- **Result (36 clean held-out, smoothed):** 80.4% vs `v4_clean` 80.6%; original 10: 78.9% vs 79.3%;
  changes 2.05× vs 2.01×, change-F 0.558 vs 0.564.
- **Decision:** rejected. No gain; the network already copes with drums.

### E20 · Key-signature input, oracle ceiling · 2026-11-12 — `models/e20_key`
- **Question:** how much could the Transformer gain from knowing the key? (Upper bound.)
- **Setup:** each song's true key signature (from its annotations, via `key.fit_relative_profile`)
  as 12 extra input columns, projected and added to every frame (`--key-input`); rotated with
  pitch-shift augmentation. Same recipe and baseline as E19.
- **Decision rule:** pursue a causal version (decaying 60 s note profile) only if this gains ≥ 0.5 pt.
- **Result (36 clean held-out, smoothed):** 81.1% vs `v4_clean` 80.6% (+0.5); original 10: 79.4% vs
  79.3%; changes 1.98× vs 2.01×, change-F 0.574 vs 0.564.
- **Decision:** not pursued. Even a perfect key gains only +0.5, at the threshold and within noise,
  and a causal estimate can only do worse. This matches E07: most errors are between in-key chords.

### F01 · Flutter diagnosis (decoding only, no retraining) · 2026-11-12
- **Question:** where do the extra chord changes (1.6× the reference) come from, and can decoding fix them?
- **Anatomy (v5, 36 clean held-out):** 1,279 predicted segments < 0.3 s vs 13 in the reference
  (923 vs 239 at 0.3–0.6 s); longer segments match. Flips are between neighbouring chords: D↔E 306,
  A↔D 94, B↔E 76, A↔E 75, C↔G 73, …, concentrated at real chord changes.
- **Decoding variants:**

  | variant | acc | changes | change-F | cost |
  |---|---|---|---|---|
  | deployed | 80.8% | 1.60× | 0.599 | – |
  | hysteresis 2 / 4 / 6 frames | 79.4 / 76.6 / 73.9% | 1.51 / 1.38 / 1.26× | 0.58 / 0.47 / 0.28 | +80–240 ms |
  | oracle change points | 81.6% | 1.48× | 0.636 | (oracle) |
  | fixed-lag HMM 3 / 5 / 8 frames | 82.6 / 82.8 / 82.9% | 1.54 / 1.46 / 1.38× | 0.65 / 0.67 / 0.68 | +120–320 ms |
  | no reversal 5–25 frames | 80.5–78.6% | 1.58–1.60× | ≤ 0.60 | – |
  | teacher (±5 s lookahead) | 82.9% | 1.04× | 0.71 | offline |

- **Conclusion:** the flutter comes from the network (confident flips in the first frames after a
  change); no causal decoding rule removes it. Only future context does. Fixed-lag decoding is a free
  +2 pts of accuracy for 80–120 ms of delay.

### E22 · Zero-lookahead anti-flutter training · 2026-11-12 — queued (runs after E20)
- **Script:** [`experiments/2026-11-12_anti_flutter.ps1`](../experiments/2026-11-12_anti_flutter.ps1)
- **Setup:** human labels only, E15's recipe and data (`data/cache_bb_fix`), compared with `models/v4_clean`
  (80.6% raw). **E22a** `--stability-weight 2`: squared change of the predicted distribution between
  consecutive frames, counted only where the true chord is constant. (E22b, harder distillation, was
  dropped under the no-distillation policy.)
- **Note:** the first run died at epoch 45 on a CUDA "unspecified launch failure" (driver fault; the
  other runs in the batch were fine); rerun from scratch with the same command.
- **Result (36 clean held-out, zero delay):** 80.4% vs `v4_clean` 80.6%; original 10: 79.1% vs 79.3%;
  changes **1.78×** vs 2.01×, change-F **0.590** vs 0.564. With fixed-lag 3 / 6: 81.8% / 81.9%
  (vs 81.7% / 81.8%).
- **Decision:** not adopted. ~11% fewer changes for −0.2 pt, but simply decoding `v4_clean` with a
  flatter HMM (power 1) gives a similar trade (80.0%, 1.61×, change-F 0.602) with no retraining. A
  loss term can't remove flutter without future context; E23 can.

### E23 · Small lookahead via label delay (120 / 240 ms) · 2026-11-12 — `models/e23a_delay3`, `models/e23b_delay6`
- **Setup:** `--label-delay 3` / `6`: the causal network is trained so that output t names the chord
  of frame t − d, giving each decision d frames of "future" context with no architecture change (the
  KV-cached ONNX step is unchanged; the display is d × 40 ms late). `predict_song` realigns outputs,
  so the metrics score the frames being described. Human labels only; same recipe and baseline as E22.
- **Fair baseline:** the same delay spent in the decoder instead: `v4_clean` with a fixed-lag HMM
  (frame s decided at s + d using frames s+1..s+d), as in F01.
- **Result (36 clean held-out, bundle HMM):**

  | delay | model | acc | original 10 | changes | change-F |
  |---|---|---|---|---|---|
  | 0 | `v4_clean` | 80.6% | 79.3% | 2.01× | 0.564 |
  | 120 ms | `v4_clean` + fixed-lag 3 | 81.7% | 80.6% | 1.82× | 0.613 |
  | 120 ms | **E23a** (label delay 3) | **82.9%** | **82.0%** | **1.68×** | **0.650** |
  | 240 ms | `v4_clean` + fixed-lag 6 | 81.8% | 80.9% | 1.70× | 0.633 |
  | 240 ms | **E23b** (label delay 6) | **83.9%** | **83.0%** | **1.44×** | **0.693** |
  | offline | teacher (±5 s) | 82.9% | 82.0% | 1.04× | 0.71 |

  Paired per song at equal delay: E23a +1.24 pt, 95% CI [+0.78, +1.76], better on 29/36, sign test
  p = 0.0003; E23b +2.08 pt, CI [+1.28, +2.99], 30/36, p = 0.0001. Fixed-lag decoding on top of a
  label-delay model adds little (E23b + lag 6: 83.9%, 1.34×). HMM grid (1–8 s × power 1–3) moves
  accuracy < 0.5 pt for every model; power 1 trades ~0.3–0.5 pt for ~15% fewer changes.
- **Engine:** `ChordStream` reads `label_delay` from the bundle; events carry `delay` (seconds) and
  per-beat chords are pooled over the frames they describe. Streaming check on a held-out song:
  per-frame accuracy 76.9% scored at the described frames vs 75.3% if the delay were ignored.
- **Decision:** adopted. Training for the delay is worth +1.2–2.1 pt over spending the same delay in
  the decoder, and E23b beats the offline teacher's accuracy with 240 ms of delay. Flutter is
  reduced (2.01× → 1.44×) but not gone. Next: the same on the extended vocabulary (E25).

### F02 · Speculative display: zero-delay draft + label-delay verifier · 2026-11-12
- **Question (user idea, after LLM speculative decoding):** can a zero-delay draft be shown at once
  and a label-delay model verify/correct it, giving instant changes with the delayed model's stability?
- **Script:** [`experiments/analysis_speculative.py`](../experiments/analysis_speculative.py). Draft
  `v4_clean`, verifier E23a / E23b. Displayed sequence scored against labels at display time.
- **Accept rules and results (36 clean held-out):**

  | live display | acc at display | changes | change-F |
  |---|---|---|---|
  | draft only (zero delay) | 80.6% | 2.01× | 0.564 |
  | draft if HMM confidence ≥ τ, else verifier (τ 0.5–0.99) | 80.6% | 2.0–2.07× | ≤ 0.565 |
  | draft once held k frames, else verifier (E23b, k 2–6) | 79.8–77.0% | 2.06–1.63× | 0.54–0.47 |
  | draft if network prob ≥ 0.9, else verifier (E23a) | 80.2% | 1.70× | 0.591 |
  | **E23a alone, 120 ms late** | 80.0% | **1.68×** | **0.591** |
  | E23b alone, 240 ms late | 77.1% | 1.44× | 0.462 |

  The HMM posterior is > 0.99 almost everywhere, flips included, so confidence gating accepts nearly
  every draft; network confidence helps only marginally.
- **Conclusion:** per-frame draft/verify doesn't pay: the draft's flips are confident errors (as in
  F01), so no accept rule separates them. Showing the 120 ms model directly is as good. The idea does
  fit the **chord sheet**, which is already one beat (~500 ms) late: a longer-delay "verifier" there
  costs no extra visible latency. → E26 (one network, two delay heads: short for the live readout,
  long for the sheet).

### E24 · Extended vocabulary without distillation · 2026-11-12 — queued (runs after E23)
- **Setup:** E18's data and heads (`data/cache_ext`, `--vocab extended`) with E15's recipe, human labels
  only. Compared with v6 (the distilled extended model) in both majmin and full-vocabulary views.
- **Purpose:** a non-distilled replacement for v6, and a measure of what distillation contributed to it.
- **Result (36 clean held-out):** both with v6's deployed decoder (HMM 5 s, power 2):

  | | v6 (distilled) | E24 (human labels only) |
  |---|---|---|
  | majmin view | 80.8% | **81.0%** |
  | full vocabulary | 70.3% | **70.7%** |
  | changes / change-F (majmin) | **1.61×** / 0.594 | 1.71× / **0.604** |

  With its own training-time HMM (power 3): 81.2% majmin, 70.9% full, 1.99×.
- **Decision:** distillation contributed nothing measurable on the extended model, so E24 replaces
  v6 as the non-distilled extended baseline. Not deployed by itself: E25 (with label delay) is the
  deployment candidate.

### E25 · Extended vocabulary + label delay (120 / 240 ms) · 2026-11-12 — `models/e25a_ext_delay3`, `models/e25b_ext_delay6` → `v7`
- **Script:** [`experiments/2026-11-12_ext_delay.ps1`](../experiments/2026-11-12_ext_delay.ps1)
- **Setup:** E24 (extended vocabulary, structured heads, human labels only) + E23's `--label-delay 3` / `6`.
- **Result (36 clean held-out; majmin view / full vocabulary / changes / change-F):**

  | model | decoder | majmin | full | changes | change-F |
  |---|---|---|---|---|---|
  | v6 (distilled, zero delay) | deployed | 80.8% | 70.3% | 1.61× | 0.594 |
  | E24 (zero delay) | bundle | 81.2% | 70.9% | 1.99× | 0.577 |
  | E24 + fixed-lag 3 (120 ms) | bundle | 82.4% | 72.1% | 1.81× | 0.616 |
  | **E25a** (delay 3, 120 ms) | bundle (power 3) | 83.5% | 73.1% | 1.64× | 0.654 |
  | **E25a** | HMM 5 s / power 2 | **83.3%** | **73.0%** | **1.48×** | **0.669** |
  | E24 + fixed-lag 6 (240 ms) | bundle | 82.5% | 72.3% | 1.66× | 0.646 |
  | E25b (delay 6, 240 ms) | HMM 5 s / power 2 | 84.0% | 73.9% | 1.33× | 0.709 |

  Original 10 (majmin): E25a 83.1% (v6 79.9%). Streaming check on a held-out song: 80.5% per frame
  scored at the described frames (78.5% if the delay were ignored).
- **Decision:** deployed **E25a** as **v7** (HMM 5 s / power 2; release
  [`model-v7`](https://github.com/helloparthshah/rubato/releases/tag/model-v7)): +2.5 pt majmin,
  +2.7 pt full vocabulary and 8% fewer changes than v6, with 120 ms display delay and no distillation.
  E25b is better still but 240 ms late on the live readout; E26 tests getting that accuracy on the
  sheet only.

### E26 · Two delay heads (live 120 ms + sheet 320 / 480 ms) · 2026-11-12 — `models/e26a_sheet8`, `models/e26b_sheet12`
- **Script:** [`experiments/2026-11-12_sheet_head.ps1`](../experiments/2026-11-12_sheet_head.ps1)
- **Setup:** E25a's recipe plus a second chord head trained with `--sheet-delay 8` / `12`.
- **Result (36 clean held-out, HMM 5 s / power 2; majmin / full / changes):** live heads 82.7% /
  72.4% / 1.50× (v7 alone: 83.3% / 73.0% / 1.48×, so sharing the trunk costs 0.6 pt); sheet heads
  83.8% / 73.6% / 1.25× (delay 8) and 83.9% / 73.7% / 1.20× (delay 12), no better than E25b's single
  delay-6 head (84.0% / 73.9%). Gains from delay saturate around 240 ms.
- **Decision:** rejected. The code (`--sheet-delay`, optional `sheet_probs` ONNX output) stays, unused.

### F03 · What should the chord sheet be built from? · 2026-11-12
- **Script:** [`experiments/analysis_beat_sheet.py`](../experiments/analysis_beat_sheet.py); 36 clean held-out.
- **Per-beat averaging (E16's engine) no longer helps:** for label-delay models, averaging
  posteriors over causal-tracker beats is *worse* than the HMM path (v7: 82.3% vs 83.3% majmin;
  E25b 82.0% vs 84.0%); a majority vote of the HMM path per beat is no better (81.9–82.3%). It still
  helps the zero-delay v6 (81.8% vs 80.8%), which is why E16 adopted it. Imperfect beat edges cost
  more than the averaging gains once the network already has some future context.
- **Minimum-hold commit (the useful form of F02's "speculative" idea):** the sheet commits a chord
  only once the live HMM chord has held k frames, backdated to where it started:

  | v7, hold k | majmin | full | changes |
  |---|---|---|---|
  | 0 (HMM path) | 83.3% | 73.0% | 1.60× |
  | 5 (200 ms) | 83.3% | 73.0% | 1.35× |
  | 8 (320 ms) | 83.4% | 73.3% | 1.18× |
  | **12 (480 ms)** | **83.4%** | **73.4%** | **1.03×** |

  (changes here on the full vocabulary.) E25b with 12: 84.1% / 74.2% / 1.03×.
- **Decision:** adopted in the engine (`ChordStream(sheet_min_frames=12)`): the live readout shows
  every decision 120 ms late; the sheet commits after 480 ms, which is about as late as the old
  per-beat chords, and has essentially the reference number of chord changes. Streaming check on 4
  held-out songs: 0.99× the reference changes.

### E21 · Bollywood songs with human chord labels from UG sheets · 2026-11-12 — in progress
- **Question:** all labelled songs are US Billboard hits (1958–1991). How well does v7 do on
  Bollywood, and does fine-tuning on Bollywood songs with human (chord-sheet) labels help?
- **Earlier probe:** *Raat Bhar* (Heropanti), v6 vs its UG sheet (E, B, C♯m, A, D): key matches;
  93% of time on the sheet's chords (extra: G♯m 6%); 88% of the sheet's 75 changes found in order;
  over-segmented (136 vs 75 changes, mostly A↔C♯m and C♯m↔B).
- **Pipeline** (`chordstream bollywood <stage>`, `data/bollywood.py`): YouTube playlists of each
  year's top Bollywood songs (2010–2024; 266 songs) → UG "Chords" sheet per song (search, title
  overlap, rating × votes; then a plausibility check: every word of UG's song name must be in the
  video title and the sheet filed as a soundtrack or under an artist named in the title, which
  catches look-alikes such as "You Are My Soniya" → "You Are My Sunshine") → audio → labels.
- **Labels without a neural model** (`data/sheet_align.py`): sheets give chords in order but no
  times. Chroma-template forced alignment of the sheet's chord sequence, allowing jumps to line
  starts (McVicar et al.'s jump alignment; sheets omit repeats), all 12 transpositions tried
  (capo / key). Chord identities are human; timing is signal processing.
- **Aligner validation** (`experiments/analysis_sheet_align.py`; 36 held-out Billboard songs, the
  "sheet" = each song's annotated chord sequence given in a random key):

  | method | full vocab | majmin | key recovered |
  |---|---|---|---|
  | sheet chords + bigram grammar, unordered | 58.8% | 65.5% | 31/36 |
  | **ordered forced alignment** | **77.6%** | **79.6%** | **36/36** |

  Emission sharpness and chord duration barely matter (76.5–77.0% on a 12-song grid); the chroma
  templates are the limit. Real sheets are less complete than this simulation.
- **Plan:** the Bollywood test set is scored mainly with timing-free sheet metrics (key, share of
  time on the sheet's chords, in-order coverage of the sheet's changes), aligned-frame accuracy as
  a secondary measure; fine-tune v7's recipe on Billboard + aligned Bollywood songs (low-margin
  alignments excluded) and compare on the Bollywood test set and the 36 Billboard songs.
- **Data:** 266 playlist songs → 175 UG matches → 106 pass the plausibility check → 98 after
  removing duplicates (same sheet via two playlists) → all 98 downloaded and aligned.
- **Alignment check against the capo:** the transposition the aligner picks equals the sheet's
  capo (none = 0) on **60 / 98** songs, an independent confirmation of the alignment. Most of the
  rest are off by a fifth (e.g. capo 1 → shift 8), which chroma confuses. Only these 60 are used
  (`bollywood.trusted_ids`): 19 test songs (30% by hash of the cache name), 41 for training.
- **Baseline (19 Bollywood test songs, `experiments/analysis_bollywood.py`):**

  | model | in sheet | coverage | changes | aligned majmin / full |
  |---|---|---|---|---|
  | v6 | 88.6% | 70.5% | 0.99× | 54.5% / 47.3% |
  | v7 | 88.2% | 71.7% | 0.96× | 55.8% / 50.0% |

  The chords used are mostly the sheet's (88%), but frame agreement with the aligned labels is far
  below Billboard (83%): partly label noise (the aligner is ~78–80% on Billboard, likely less here),
  partly a domain gap.
- **Training** ([`2026-11-12_bollywood.ps1`](../experiments/2026-11-12_bollywood.ps1)): `data/cache_ext`
  + the 41 training songs linked 3× (prefix `bollytrain__`, never held out); E21a = v7's recipe from
  the synthetic init, E21b = v7 fine-tuned 20 epochs at lr 1e-4.
- **Result:**

  | model | Bollywood: in sheet | coverage | changes | aligned majmin / full | Billboard 36: majmin / full | original 10 |
  |---|---|---|---|---|---|---|
  | v7 | 88.2% | 71.7% | 0.96× | 55.8% / 50.0% | **83.3% / 73.0%** | **83.1%** |
  | E21a (scratch) | 92.5% | 74.3% | 1.35× | **67.4% / 65.1%** | 81.6% / 70.7% | 80.0% |
  | E21b (fine-tune) | **92.7%** | **74.4%** | 1.30× | 65.2% / 62.9% | 81.1% / 70.8% | 80.3% |

  Bollywood improves on the timing-free sheet metrics too (in sheet +4.3, coverage +2.6), so this
  is not just learning the aligner's habits; but Billboard drops 1.7–2.2 pt and Bollywood output
  changes chord more often. Likely cause: the aligned labels' errors (~20%, concentrated at chord
  boundaries) at 25% of the training weight.
- **E21c / E21d** ([`2026-11-12_bollywood_masked.ps1`](../experiments/2026-11-12_bollywood_masked.ps1)):
  Bollywood labels ignored within ±320 ms of every aligned chord change
  (`bollywood.mask_boundaries`), 3× / 1× weight. Running.
- **E21c / E21d result** (v1 benchmark): in sheet 93.0% / 92.0%, coverage 74.1% / 74.0%, aligned
  majmin 64.1% / 60.9%; Billboard 82.4% / 82.6% (bundle HMM). Less Billboard loss than E21a, but
  lower agreement with the v1 labels, which reward copying the aligner's boundary errors.
- **Label cleaning** (user request; [`experiments/analysis_label_cleaning.py`](../experiments/analysis_label_cleaning.py),
  measured on the 36 Billboard songs with true timings, key given):

  | aligner | full | majmin | frames kept |
  |---|---|---|---|
  | E21 aligner | 77.6% | 79.6% | 100% |
  | + bass evidence (weight 0.5) | 78.7% | 80.4% | 100% |
  | + beat snapping (±4 / ±8 frames) | 77.2% / 76.4% | – | worse (causal beats too rough) |
  | bass + keep top 80% / **70%** / 60% by confidence | 81.1 / **82.1** / 83.5% | 82.7 / **83.7** / 84.7% | 80 / 70 / 60% |

  Confidence = the aligned chord's emission minus the best other sheet chord's, smoothed over 0.5 s.
  `chordstream bollywood clean` (`bollywood.stage_cache_clean`): bass 0.5; the capo transposition is
  used whenever its likelihood is within 0.05/frame of the best (fixes the fifth confusion); the
  least-confident 30% of frames (pooled) are ignored → `data/cache_bolly2`, **70 / 98** songs kept
  (was 60). No neural model involved.
- **v2 benchmark** (cleaned labels, 23 test songs; supersedes v1 for selection):

  | model | in sheet | coverage | aligned majmin / full |
  |---|---|---|---|
  | v7 | 89.5% | 73.8% | 68.9% / 61.3% |
  | E27b | 89.1% | 73.3% | 69.0% / 61.7% |
  | E21a | 92.1% | 73.4% | **75.5% / 73.6%** |
  | E21b | 92.6% | 73.9% | 75.0% / 72.9% |
  | E21c | **93.0%** | 73.8% | 73.8% / 70.9% |
  | E21d | 92.4% | **75.0%** | 73.2% / 67.6% |

  v7's agreement rises from 55.8% to 68.9% with cleaner labels: most of the earlier gap was label
  noise. **E21e / E21f** ([`2026-11-12_bollywood_clean.ps1`](../experiments/2026-11-12_bollywood_clean.ps1)):
  train on the cleaned labels (×3; 47 training songs), f also with E27b's 50 rare renders.
- **E21e / E21f result** (v2 benchmark; Billboard with HMM 5 s / power 2):

  | model | Bollywood majmin / full | in sheet | coverage | Billboard majmin / full | **mean** | pooled (59 songs) |
  |---|---|---|---|---|---|---|
  | v7 | 68.9% / 61.3% | 89.5% | 73.8% | **83.3% / 73.0%** | 76.1% | 77.7% |
  | E21a | 75.5% / 73.6% | 92.1% | 73.4% | 81.6% / 70.7% | 78.6% | 79.0% |
  | E21c | 73.8% / 70.9% | 93.0% | 73.8% | 82.4% / 71.4% | 78.1% | 79.0% |
  | **E21e** | **80.0% / 78.0%** | 92.8% | **76.6%** | 81.8% / 70.5% | **80.9%** | **81.1%** |
  | E21f (+ rare) | 79.9% / 77.9% | **93.1%** | 76.1% | 81.4% / 69.9% | 80.7% | 80.8% |

  E21e's aligned-label score may be flattered (trained on labels made the same way), but it also
  leads on in-order coverage, which doesn't use the aligner's timing. E21f learns dim (32% recall)
  at a small cost elsewhere.
- **Decision:** deployed **E21e** as **v8** (HMM 5 s / power 2; release
  [`model-v8`](https://github.com/helloparthshah/rubato/releases/tag/model-v8)): +4.8 pt overall,
  +11 pt majmin on Bollywood, −1.5 pt on Billboard (accepted under the selection rule).
- **Overall (selection rule above, v1 benchmark), majmin:** v7 83.3 / 55.8 → mean **69.6%** (pooled per song
  73.8%); E21a 81.6 / 67.4 → **74.5%** (76.7%); E21b 81.1 / 65.2 → 73.2% (75.6%). E21a leads; final
  choice after E21c/d.

### E27 · Rare chord types from synthetic renders · 2026-11-12 — `models/e27_rare`
- **Script:** [`experiments/2026-11-12_rare_chords.ps1`](../experiments/2026-11-12_rare_chords.ps1)
- **Setup:** `chordstream synth --rare`: diatonic loops with sus2/sus4 that resolve, passing
  diminished chords a semitone below the next chord, augmented links and 7ths, exact labels
  (400 rendered, `data/cache_rare`, training-only prefix `synthx`). 150 of them mixed into
  `data/cache_ext` (~30% of songs); v7's recipe otherwise.
- **Result (36 clean held-out, HMM 5 s / power 2 for both; `experiments/analysis_per_type.py`):**

  | | v7 | E27 |
  |---|---|---|
  | majmin / full | **83.3% / 73.0%** | 82.8% / 72.3% |
  | changes | **1.60×** | 1.66× |
  | dim recall (0.3% of frames) | 4% | **52%** |
  | aug recall (0.3%) | 0% | **24%** |
  | sus4 recall (2.9%) | 15% | **19%** |
  | sus2 (< 0.1%) | 0% | 0% |
  | 7 / maj7 / m7 recall | **44 / 48 / 55%** | 40 / 43 / 51% |

- **Decision:** not deployed. The rare types are learnable from synthetic audio, but at 30% of
  songs the synthetic share costs 0.5–0.7 pt overall (7th chords most).
- **E27b** (50 renders, ~12% of songs; [`2026-11-12_rare_chords_light.ps1`](../experiments/2026-11-12_rare_chords_light.ps1)):
  majmin 83.1% / full 72.8% (v7 83.3% / 73.0%, within noise), original 10 83.3% (v7 83.1%),
  changes 1.65×; dim recall **50%** (v7 4%), aug 5%, sus4 18% (v7 15%), 7 / maj7 / m7 43 / 45 / 54%.
  Bollywood test: level with v7 (in sheet 87.8%, coverage 71.6%).
- **Decision (E27b):** keeps most of the diminished-chord gain at no measurable cost; deployment
  decided together with E21.

### E28 · Scaled Bollywood set (152 training songs) · 2026-11-12 — `models/e28a_bolly3x1`, `models/e28b_bolly3x2`

- **Data:** the E21 scale-up ([`2026-11-12_bollywood_scale.ps1`](../experiments/2026-11-12_bollywood_scale.ps1))
  covered 2000–2024 (3 playlists × 40 songs per year, 1,426 songs). After sheet matching, validation,
  audio, and E21 cleaning, `data/cache_bolly3` has **213 kept songs** (from 70), with 152 for training
  and 61 for testing. The hash split is the same as before, so the 61 test songs include v2's 23 and no
  test song is ever trained on.
- **Benchmark v3:** `analysis_bollywood.py --v3` scores the 61 test songs. From now on it is the
  Bollywood half of the selection rule, and v2 is kept for continuity. The `/annotate` page now lists
  these 61 songs.
- **Runs** ([`2026-11-12_bollywood_scaled_train.ps1`](../experiments/2026-11-12_bollywood_scaled_train.ps1)):
  - Recipe: v8's, from the synthetic init. Each model is evaluated with HMM 5 s / power 2
    (`experiments/hmm55.py` writes `<model>_h5p2`).
  - E28a sees each Bollywood song once, which keeps Bollywood's share about the same as E21e's 47 × 3.
  - E28b sees each song twice.

  | model | Billboard majmin / full | Bollywood v3 majmin / full | in sheet | coverage | changes | v2 majmin | **mean (v3)** | pooled (97 songs) |
  |---|---|---|---|---|---|---|---|---|
  | v7 | **83.3% / 73.0%** | 60.7% / 54.0% | 78.1% | 68.7% | 0.96× | 68.9% | 72.0% | 69.1% |
  | v8 (E21e, 47 songs ×3) | 81.8% / 70.5% | 78.0% / 75.2% | **90.4%** | **78.6%** | 1.16× | 80.0% | 79.9% | 79.4% |
  | E28a (152 ×1) | 81.5% / 70.2% | 78.2% / 75.7% | 90.3% | 78.3% | 1.13× | 80.0% | 79.9% | 79.4% |
  | E28b (152 ×2) | 80.2% / 67.7% | **80.0% / 77.4%** | 89.8% | 78.4% | 1.25× | **81.6%** | **80.1%** | **80.1%** |

- **Findings:**
  - v8 generalizes to the 38 Bollywood test songs it had never been scored on: 78.0% there, against
    v7's 60.7%.
  - Tripling the Bollywood songs buys almost nothing. E28a ties v8. E28b gains +0.2 pt on the mean,
    but only on the aligned-label score, which may be flattered by training on labels made the same
    way.
  - E28b does worse on the measures that don't depend on the aligner's timing: in sheet −0.6 pt,
    coverage −0.2 pt, 1.25× changes (more flicker). It is also −1.6 pt on Billboard.
  - The limit now looks like the quality of the sheet-aligned labels, not the number of songs.
- **Decision:** at first kept v8, because E28b's gain is within noise and the aligner-independent
  metrics fell slightly. The user then chose to apply the selection rule as stated (overall accuracy
  up, Billboard drop accepted). **E28b is deployed as v9** (HMM 5 s / power 2, ONNX parity 2.6e-06,
  release [`model-v9`](https://github.com/helloparthshah/rubato/releases/tag/model-v9)).
  Hand-corrected gold labels should confirm it; if they don't, revert to v8.

### F04 · Flicker anatomy of v9 (live readout) · 2026-11-12

- **Script:** [`experiments/analysis_flicker.py`](../experiments/analysis_flicker.py). Measures the
  live HMM path at the frames it describes, in the majmin view. A "blip" is A → B → A with B lasting
  under 0.5 s.

  | | changes/min (ref) | segments < 0.2 s | segments < 0.5 s (ref) | blips/min | blips wrong |
  |---|---|---|---|---|---|
  | v9, Billboard 36 | 46.4 (26.3) | 11.4% | 32.9% (3.2%) | 6.3 | 77% |
  | v8, Billboard 36 | 44.5 (26.3) | 10.9% | 30.9% (3.2%) | 5.8 | 80% |
  | v9, Bollywood 61 | 60.3 (47.3) | 16.9% | 47.2% (33.9%) | 10.8 | 61% |
  | v8, Bollywood 61 | 55.5 (47.3) | 17.1% | 44.9% (33.9%) | 9.9 | 63% |

- **Findings:**
  - The readout changes about 1.8× as often as the reference on Billboard, and a third of its
    segments are shorter than 0.5 s.
  - Blips account for about 27–36% of the changes, and most of them are wrong.
  - The wrong blips jump almost always to the chord a fourth or fifth away, or to a neighbouring
    diatonic chord: E→D, D→E, B→E, C→G, G→D, F→C, Dm→Am. This looks like a bass or melody note
    briefly outweighing the chord.
  - v9 flickers 4–9% more than v8.
  - The Bollywood reference is itself choppy (34% of segments under 0.5 s), which is aligner noise.
    Billboard is the reliable flicker yardstick.
- **UI:** the big readout shows every decision and plays a pop animation on each change, so every
  blip is visible. Browser echo cancellation, noise suppression and auto-gain are already off.

### F05 · Display-side flicker fixes (no retraining) · 2026-11-12

- **Scripts:** [`experiments/analysis_flicker_fixes.py`](../experiments/analysis_flicker_fixes.py) and
  [`analysis_evidence_grid.py`](../experiments/analysis_evidence_grid.py). Model v9. Each display is
  scored *at display time*, so any waiting counts as lag.
- **Displays tested:**
  - **steady k:** switch once the live chord has held k frames. This is the sheet rule without
    backdating.
  - **evidence hold k / t:** leave the shown chord A for B at once, unless the network still gives A
    at least t probability; then B must hold k frames first.
  - **Pitch-shift averaging:** average the posteriors from inputs shifted ±1 and ±2 semitones, before
    the HMM.

  | Billboard 36 (Bollywood 61 majmin) | majmin | changes/min (ref 26.3) | blips/min | segments < 0.5 s |
  |---|---|---|---|---|
  | live HMM path (Instant) | 80.2% (80.0%) | 46.4 | 6.3 | 33% |
  | steady 6 (240 ms) | 74.3% (70.3%) | 38.0 | 3.1 | 19% |
  | steady 12 (480 ms) | 66.5% (61.4%) | 28.7 | 0.2 | 2% |
  | **evidence hold 20 / 0.05** | **79.6%** (77.0%) | **35.1** | **1.8** | 20% |
  | evidence hold 20 / 0.02 | 77.0% (72.9%) | 29.1 | 0.8 | 12% |
  | evidence hold 12 / 0.05 | 79.7% (77.1%) | 36.9 | 2.2 | 21% |
  | pitch-shift avg ±2, live | 80.3% (80.4%) | 45.9 | 5.9 | 32% |

- **Findings:**
  - The evidence hold dominates plain holds. At the same flicker as steady 12, it is 10.5 pt more
    accurate at display time.
  - At 20 / 0.05 it removes about 70% of blips for 0.6 pt on Billboard.
  - Bollywood loses more (−3 pt), partly because its aligned reference is itself choppy (F04).
  - Pitch-shift averaging does almost nothing: the blips are consistent across shifts, which is the
    confident-error pattern seen in F01 again.
- **Decision:**
  - The engine emits `display_label` / `display_name` (evidence hold 20 / 0.05). The web page shows it
    big by default ("Steady"), with the live chord small underneath ("hearing D…"); the **Instant**
    checkbox restores per-frame display.
  - Streaming check on Billboard 0010 (60 s): the engine display agrees with the offline rule on 97%
    of frames, and changes drop from 65 to 49.
  - An earlier version of the Steady display used the 480 ms sheet rule, and was replaced the same day
    after this table.
  - Pitch-shift averaging is rejected.

### F06 · Does live-like audio flicker more? · 2026-11-12

- **Script:** [`experiments/analysis_degraded.py`](../experiments/analysis_degraded.py). The 36
  Billboard test songs are re-rendered and given a fresh causal CQT, then scored with v9:

  | audio | majmin | full | changes/min | blips/min |
  |---|---|---|---|---|
  | clean | 80.2% | 67.7% | 46.4 | 6.3 |
  | room echo (synthetic RIR, RT60 0.5 s) | **75.4%** | 63.5% | 47.8 | 6.4 |
  | phone-mic band 150 Hz–6 kHz + soft clip | 80.4% | 68.1% | 45.4 | 5.9 |
  | pink noise, 15 dB SNR | 79.7% | 67.0% | 46.6 | 6.2 |
  | all three | 75.0% | 63.2% | 47.0 | 6.4 |

- **Findings:**
  - Flicker is a property of the model, not the room: it hardly changes under any degradation.
  - Room echo, however, costs **4.8 pt** of accuracy. The phone-mic band and noise are harmless.
- **Next:** E29 trains with room-echo copies of the training songs.

### E29 · Room-echo training copies · 2026-11-12 — `models/e29_room`

- **Script:** [`2026-11-12_room_echo.ps1`](../experiments/2026-11-12_room_echo.ps1), using
  [`build_room_cache.py`](../experiments/build_room_cache.py).
- **Data:** v9's mix plus one re-rendered copy of each of the 483 training songs in `data/cache_room`
  (`roomx__` prefix, train-only; test songs are never rendered). Each copy gets a random synthetic room
  (RT60 0.25–0.9 s). Half of the copies also get the phone-mic band and pink noise at 10–25 dB SNR.
- **Result** (HMM 5 s / power 2):

  | | Billboard majmin / full | Bollywood v3 majmin | **overall** | room echo (F06) | all degradations |
  |---|---|---|---|---|---|
  | v9 | 80.2% / 67.7% | **80.0%** | **80.1%** | 75.4% | 75.0% |
  | E29 | 80.2% / 67.3% | 79.2% | 79.7% | **77.1%** | **75.8%** |

  Flicker: 47.9 changes/min on clean Billboard, against 46.4 for v9.
- **Decision:** not deployed. Overall accuracy on the two test sets falls 0.4 pt. The echo gain
  (+1.7 pt) is measured on synthetic rooms like the training ones, so it is probably optimistic. Worth
  revisiting once there is real phone-in-a-room test audio.

### E30 · Stability loss with label delay · 2026-11-12 — `models/e30a_stab2`, `models/e30b_stab5`

- **Script:** [`2026-11-12_stability_delay.ps1`](../experiments/2026-11-12_stability_delay.ps1).
- **Setup:** v9's recipe and data, plus E22's stability loss at weights 2 and 5.
- **Result** (HMM 5 s / power 2):

  | | Billboard majmin (changes) | Bollywood v3 majmin (changes) | **overall** |
  |---|---|---|---|
  | v9 | **80.2%** (1.62×) | **80.0%** (1.25×) | **80.1%** |
  | E30a (w 2) | **80.2%** (1.57×) | 79.5% (1.19×) | 79.9% |
  | E30b (w 5) | 80.0% (1.51×) | 78.9% (1.12×) | 79.5% |

- **Findings:** with look-ahead the loss trims chord changes by 3–7%, at a cost on Bollywood. That is far
  less than the display-side evidence hold achieves (F05).
- **Decision:** not deployed.

### F07 · Bass dependence · 2026-11-12

- **Script:** [`experiments/analysis_bass.py`](../experiments/analysis_bass.py). Prompted by the user:
  "it struggles when there's less or no bass". Chords should come from the harmony, not the bass line.
- **Result for v9** (majmin):

  | | clean | bass removed < 150 Hz | < 300 Hz | quietest-bass 25% of frames | other frames | inverted chords | root position |
  |---|---|---|---|---|---|---|---|
  | Billboard 36 | 80.2% | 76.4% | 65.2% | 76.3% | 81.5% | **46.6%** (13 min) | 85.6% |
  | Bollywood 61 | 80.0% | **66.8%** | **46.8%** | 75.5% | 81.3% | – | 81.4% |

- **Findings:**
  - The model leans heavily on the bass.
  - On Bollywood, removing everything below 150 Hz costs 13 pt, and below 300 Hz costs 33 pt.
  - Inverted chords are mostly named after their bass note.
- **Next:** E31, bass dropout in training.

### E31 · Bass dropout · 2026-11-12 — `models/e31a_bassdrop3`, `models/e31b_bassdrop6`

- **Script:** [`2026-11-12_bass_dropout.ps1`](../experiments/2026-11-12_bass_dropout.ps1).
- **Setup:** v9's recipe and data, plus `--bass-dropout 0.3 / 0.6`. That share of training crops has
  everything below a random cutoff removed or attenuated to at most 30%, with a 2-semitone ramp. The
  cutoff is 92–370 Hz.
- The other crops keep their bass, so the model sees every song both with and without bass. This was
  the user's request.

### E32 · Random EQ (user idea) · 2026-11-12 — `models/e32a_eq5`, `models/e32b_eq5_bass3` → `v10`

- **Script:** [`2026-11-12_random_eq.ps1`](../experiments/2026-11-12_random_eq.ps1). E32a was rerun via
  `_random_eq_a_rerun.ps1`: the first run hit CUDA out-of-memory while an evaluation shared the 4 GB GPU.
- **Setup:** v9's recipe and data, plus `--random-eq 0.5`. Half of the crops pass through a random EQ
  curve, applied exactly on the log1p features: `dataset.random_eq_db` / `apply_gain_db`.
  - low shelf: usually cut, up to 40 dB, corner 70–400 Hz;
  - high shelf: −24…+6 dB, 1–3.5 kHz;
  - tilt: ±6 dB;
  - 0–3 peaks or notches.
- b also uses E31's bass dropout 0.3.
- **Result** (majmin, HMM 5 s / power 2; F07's `analysis_bass.py`):

  | | Billboard clean | Billboard no bass < 150 / < 300 Hz | inverted | Bollywood clean | Bollywood no bass < 150 / < 300 Hz | **overall clean** |
  |---|---|---|---|---|---|---|
  | v9 | **80.2%** | 76.4 / 65.2% | **46.6%** | **80.0%** | 66.8 / 46.8% | **80.1%** |
  | E31a (dropout 0.3) | 79.5% | 77.9 / 69.7% | 45.0% | 79.0% | 70.9 / 55.6% | 79.25% |
  | E31b (dropout 0.6) | 79.5% | 77.9 / 70.3% | 43.5% | 79.1% | 71.0 / 57.0% | 79.3% |
  | E32a (EQ 0.5) | 79.8% | 77.3 / 69.0% | 44.6% | 79.5% | 69.9 / 54.0% | 79.65% |
  | **E32b (EQ 0.5 + dropout 0.3)** | 79.5% | **78.1 / 71.0%** | 43.5% | 79.1% | **71.9 / 58.5%** | 79.3% |

- **Findings:**
  - All variants trade about 0.5–0.8 pt on clean audio for large gains without bass: Bollywood
    +5 / +12 pt for E32b.
  - Random EQ and dropout combine best.
  - Inverted chords don't improve. The model still names a chord after a real non-root bass. That
    needs labelled inversions, not augmentation.
  - Frames that are naturally low in bass don't improve either; these are mostly sparse passages.
- **Decision:** deployed **E32b as v10** (HMM 5 s / power 2, ONNX parity 3.2e-06), under the user's
  cost-benefit rule: "if we're seeing massive improvement in one sector while minor regression
  overall then it's worth switching".

### E33 · Learned raw-waveform front end (user idea) · 2026-11-12 — `models/e33a_raw_init`, `models/e33b_raw_scratch`

- **Code:** [`chordstream/rawfront.py`](../src/chordstream/rawfront.py) and
  [`experiments/train_raw.py`](../experiments/train_raw.py); raw audio is cached by
  `prepare_raw.py` in `data/raw16k`.
- **Setup:** the CQT is replaced by a causal strided Conv1d on the 16 kHz waveform.
  - 2 × 144 filters of 4096 samples (256 ms), stride 640, random init.
  - Each filter pair gives a magnitude, then log1p with a learned scale.
  - Its output feeds the same ChordNet; data, labels and delay are as in v9.
  - Pitch shift is done by resampling, with the labels time-stretched to match.
- **Runs:** a: ChordNet from `models/latest`, front end random; b: everything from scratch.
- **Question:** can a network learn its own pitch analysis as well as the CQT?
- **Result** (majmin, v9's HMM; flicker from F08 below):

  | | Billboard | Bollywood | Steady changes/min (BB / Bolly) | live blips/min (BB / Bolly) |
  |---|---|---|---|---|
  | v9 (CQT) | **80.2%** | **80.0%** | – | – |
  | v10 (CQT) | 79.5% | 79.1% | 36.7 / 38.7 | 6.3 / 10.0 |
  | E33a (raw, ChordNet init) | 76.3% | 76.0% | 37.0 / 38.0 | 7.5 / 10.5 |
  | E33b (raw, scratch) | 75.2% | 75.6% | 37.3 / 39.1 | 7.7 / 10.3 |

- **Findings:** 3–5 pt worse than the CQT, and no steadier (slightly jumpier live; equal after the
  Steady display). With ~600 training songs the CQT's semitone-aligned bins are a strong prior that
  learned filters don't recover. The model's bass reliance is a label shortcut (root = bass in most
  training frames), not a CQT property: the CQT has no loudness weighting and gives every octave 24
  bins.
- **Decision:** keep the CQT. If revisited: initialise the conv filters as the CQT kernels
  (starts at CQT accuracy, learns only where the data supports it).

### F08 · Flicker of the raw-waveform models · 2026-11-12
- **Script:** [`analysis_raw_flicker.py`](../experiments/analysis_raw_flicker.py); live HMM path and the
  deployed Steady display (evidence hold k20 t0.05) on both test sets. Results in the E33 and E34
  tables. Reference: the labels themselves change 28.1 / 41.2 times per minute (Billboard /
  Bollywood; Bollywood's sheet-aligned labels have 29% segments < 0.5 s).

### F09 · HMM stickiness re-tuned for flicker · 2026-11-12
- **Script:** [`analysis_hmm_grid.py`](../experiments/analysis_hmm_grid.py) on v10: mean chord
  3 / 5 / 8 / 12 / 20 s × emission power 1 / 1.5 / 2 / 3.
- **Result:** the Steady display barely moves: 34.5–36.8 changes/min on Billboard and 37.0–38.7 on
  Bollywood across the whole grid. 5 s → 20 s cuts live changes by only ~2/min; the network's
  posteriors are confident enough to swamp the transition prior. Lower power calms the live chord
  (48 → 40/min on Billboard at power 1) but costs ~1.5 pt on Bollywood.
- **Decision:** keep 5 s / power 2. Remaining flicker comes from the network; next idea is a
  duration-aware hold (a chord's exit threshold grows with how long it has played).

### F10 · Duration-aware Steady display · 2026-11-12
- **Script:** [`analysis_duration_hold.py`](../experiments/analysis_duration_hold.py) on v11. The evidence
  hold's k grows with how long the shown chord has been up: k = k0 + slope × min(age, 5 s); k0 6/10/20,
  slope 0–0.2, thr 0.05/0.2.
- **Result:** every setting lies on the same accuracy/flicker frontier as the fixed hold. The deployed
  k20 / t0.05 (Billboard 80.3%, 33.8 changes/min, 1.6 blips/min) is already at its calm end; growing k
  buys ≤ 0.8 changes/min for 0.3 pt. thr 0.2 barely holds at all.
- **Single song** ([`song_flicker.py`](../experiments/song_flicker.py), real streaming engine): Bohemian
  Rhapsody (before reference labels; see F11) live 55.3 / Steady 40.6 changes/min, 8.5 / 2.0 blips/min.
- **Decision:** keep k20 / t0.05. Display-side rules are exhausted (F05, F09, F10); further flicker
  reduction has to come from the network.

### E34 · 16 s attention context · 2026-11-12 — `models/e34_ctx400`, `models/e34b_ctx400_eq` → `v11`

- **Scripts:** [`2026-11-12_raw_and_context.ps1`](../experiments/2026-11-12_raw_and_context.ps1) (E34),
  [`2026-11-12_context_eq.ps1`](../experiments/2026-11-12_context_eq.ps1) (E34b).
- **Setup:** `--context 400` (16 s, about 8 bars at 120 BPM) instead of 128 (5 s), batch 16, same init
  (`models/latest`) and data as v9. Prompted by a pipeline description the user shared that suggests
  4–8 bars of context. E34 uses v9's recipe; E34b adds v10's random EQ 0.5 + bass dropout 0.3.
  (One E34 epoch shows 25267 s: the PC slept; training resumed normally.)
- **Result** (majmin, HMM 5 s / power 2):

  | | Billboard clean | Billboard no bass < 150 / < 300 Hz | inverted | Bollywood clean | Bollywood no bass < 150 / < 300 Hz | **overall clean** | Steady changes/min (BB / Bolly) |
  |---|---|---|---|---|---|---|---|
  | v9 | 80.2% | 76.4 / 65.2% | 46.6% | 80.0% | 66.8 / 46.8% | 80.1% | – |
  | v10 | 79.5% | 78.1 / 71.0% | 43.5% | 79.1% | **71.9 / 58.5%** | 79.3% | 36.7 / 38.7 |
  | E34 | **82.1%** | 78.6 / 65.9% | **52.0%** | **81.8%** | 65.7 / 43.7% | **82.0%** | 33.1 / 46.8 |
  | **E34b** | 81.5% | **79.6 / 72.6%** | 49.7% | 81.0% | 70.5 / 56.9% | 81.3% | 33.8 / 43.3 |

- **Findings:**
  - Longer context is the biggest clean gain since Bollywood data: +1.9 pt over v9 with only the
    context changed, and +6 to +8 pt on inverted chords.
  - Without EQ/dropout it loses v10's no-bass robustness; E34b keeps most of it (Billboard no-bass
    even improves; Bollywood no-bass −1.5 pt vs v10).
  - Billboard flickers less; Bollywood's Steady display changes more often (43 vs 39/min), close to
    the labels' own rate (41/min).
  - CPU per step 4.4 ms vs 3.0 ms (budget 40 ms). Full vocabulary: Billboard 71.0%, Bollywood 78.4%.
- **Decision:** deployed **E34b as v11** (+2.0 pt overall clean, inversions +6 pt, no-bass within
  1.5 pt of v10; ONNX parity 2.86e-06).

### F11 · Tuning robustness of v11 · 2026-11-12
- **Script:** [`analysis_detune.py`](../experiments/analysis_detune.py): every test song analysed with the CQT
  reference moved by −45..+45 cents (detuning the audio by c cents ≡ `fmin · 2^(−c/1200)`), majmin, HMM.
- **Trigger:** Bohemian Rhapsody (Isophonics labels, `data/isophonics_queen`, +1.24 s offset) scored
  68.0% / 40.6 changes/min on v11's Steady display (v10 63.2% / 45.9); retuning the audio by +17 cents
  raised it to **75.4% / 37.4** (reference 36.4 changes/min). The recording is flat of A440.

  | v11 | −45 | −30 | −15 | 0 | +15 | +30 | +45 |
  |---|---|---|---|---|---|---|---|
  | Billboard | 8.8% | 32% | 79% | 81.5% | 82% | 79% | 73% |
  | Bollywood | 13.5% | 14.6% | 77% | 81% | 67% | 64% | 67% |

- **Findings:** a cliff ~25 cents flat (the CQT bins sit halfway between semitones and the model picks
  the wrong neighbour). About 16 of the 97 test songs are > 10 cents off A440. Live tuning estimation
  ([`tuning.py`](../src/chordstream/tuning.py), spectral-peak circular mean) was too fragile: one
  estimate landed on the cliff and sent a Bollywood song to ~4%. So the fix has to be in training.

### E37 · 36 bins/octave pilot (user suggestion) · 2026-11-12 — `models/e37a_bins24`, `e37b_bins36`, `e37c_bins36_micro`
- **Script:** [`2026-11-12_cqt36_pilot.ps1`](../experiments/2026-11-12_cqt36_pilot.ps1)
  ([`build_cqt36_cache.py`](../experiments/build_cqt36_cache.py) → `data/cache_ext36`). Small setup for
  early validation: Billboard only, from scratch, 5 s context, 40 epochs. A = 24 bins (control),
  B = 36 bins/octave (`--bins-per-octave 36`), C = 36 bins + random ±1-bin (±33 cent) shift, labels
  kept (`--micro-shift`).

  | Billboard test | −45 | −30 | −15 | 0 | +15 | +30 | +45 | val |
  |---|---|---|---|---|---|---|---|---|
  | A 24 bins | 24.0% | 76.3% | 82.6% | 83.1% | 82.8% | 79.7% | 71.5% | 72.7% |
  | B 36 bins | 25.8% | 76.5% | 82.8% | 83.2% | 82.8% | 80.6% | 73.2% | 72.9% |
  | C 36 + shift | **42.5%** | **78.1%** | 82.6% | 82.8% | 81.9% | 79.9% | 70.6% | 72.1% |

- **Findings:** 36 bins alone change nothing (≤ 2 pt). The shift augmentation is what helps, and only
  partly (±33 cents doesn't cover 45). These small models are far less tuning-sensitive than v11
  (76% vs 32% at −30): v11's long context / long fine-tuning is what made it brittle.
- **Decision:** not worth rebuilding every cache at 36 bins. Test tuning augmentation on the full recipe (E35).

### E35 · Detuned training copies · 2026-11-12 — `models/e35_tune`, `models/e35b_tune25`
- **Scripts:** [`build_tune_cache.py`](../experiments/build_tune_cache.py) (two copies per training song,
  15–45 cents flat / sharp, `tunex__` prefix, train-only, 966 files),
  [`2026-11-12_detune.ps1`](../experiments/2026-11-12_detune.ps1) (E35, v11 recipe on
  `data/cache_ext_bolly3x2_tune`), [`2026-11-12_detune_share.ps1`](../experiments/2026-11-12_detune_share.ps1)
  (E35b, `--tune-share 0.25`: detuned copies limited to 25% of crops; E35 sampled by duration, ~60%).
- **Memory fix:** `CropDataset` now keeps features in float16 and converts each crop to float32 (the ×2.5
  data set had exhausted 16 GB RAM: 400–1600 s/epoch while sharing the machine with E37).

  | majmin | BB −45 | BB −30 | BB 0 | BB +45 | Bolly −30 | Bolly 0 | Bolly no bass < 150 / < 300 | overall clean |
  |---|---|---|---|---|---|---|---|---|
  | v11 | 8.8% | 32% | **81.5%** | **73%** | 14.6% | **81.0%** | **70.5 / 56.9%** | **81.2%** |
  | E35 | 60.2% | **80.7%** | 81.0% | 60.2% | **73.4%** | 67.8% | 62.3 / 49.0% | 72.8% |
  | E35b (25%) | **65.8%** | 77.0% | 81.2% | 54.7% | 40.1% | 77.4% | 69.7 / 56.4% | 78.8% |

- **Checks:** detuned copies are time-aligned with the originals (lag 0 on 12 songs) and the cached
  features match a fresh CQT to 0.002, so the Bollywood drop is real, not a data bug. Native tuning of
  training songs (sample of 66): Billboard mostly within ±10 cents, Bollywood has a group near −30;
  ~10% of detuned copies end up > 50 cents off (effectively a semitone away with the original labels).
- **Findings:** detuned copies remove the Billboard flat cliff for ≤ 0.5 pt clean, but cost Bollywood
  3.6–13 pt on in-tune audio, scaling with their share. The test sets already contain real-world tuning
  spread, so their clean numbers are the realistic measure.
- **Decision:** not deployed (−2.4 to −8 pt overall). Possible follow-up: choose each copy's shift from the
  song's measured native tuning so no copy passes ±40 cents, and a smaller share (10%).

### E36 · Early-switch penalty (user idea) · 2026-11-12 — `models/e36a_early1`, `models/e36b_early3`
- **Script:** [`2026-11-12_early_switch.ps1`](../experiments/2026-11-12_early_switch.ps1). Extra loss
  `−log(1 − p(next chord))` on the frames shortly before each labelled change (`--early-weight`,
  `--early-window`), 0.5 s window: a weight 1, b weight 3; v11 recipe otherwise.

  | | BB live | BB changes/min | Bolly live | Bolly changes/min | BB / Bolly Steady | Steady changes/min |
  |---|---|---|---|---|---|---|
  | v11 | **81.5%** | 44.7 | **81.0%** | **62.6** | **80.3 / 78.1%** | 33.8 / **43.3** |
  | E36a | 80.8% | **43.8** | 80.4% | 64.1 | 79.5 / 77.8% | **33.7** / 45.7 |
  | E36b | 79.9% | 44.5 | 79.7% | 68.0 | 79.0 / 77.8% | 34.9 / 51.8 |

- **Findings:** −0.6 to −1.6 pt and no less flicker. The
  penalty pushes probability off the upcoming chord without making the current one stickier, so the
  model hedges instead; the heavier weight flickers more on Bollywood. Flicker comes from uncertainty, not anticipation.
- **Decision:** not deployed. With F05/F09/F10 this exhausts loss/display tweaks for flicker.

### E38 · 32 s attention context · 2026-11-13 — `models/e38_ctx800` → `v12`
- **Script:** [`2026-11-13_context800.ps1`](../experiments/2026-11-13_context800.ps1). v11's recipe with
  `--context 800` (32 s, ~16 bars at 120 BPM), batch 8 (same frames per step), 73 s/epoch.

  | majmin | BB clean / full | BB no bass < 150 / < 300 | inverted | Bolly clean / full | Bolly no bass < 150 / < 300 | **overall** | Steady changes/min (BB / Bolly) | Steady blips/min |
  |---|---|---|---|---|---|---|---|---|
  | v11 | 81.5 / 71.0% | 79.6 / 72.6% | 49.7% | 81.0 / 78.4% | 70.5 / **56.9%** | 81.3% | 33.8 / **43.3** | 1.6 / **3.4** |
  | **E38** | **82.5 / 72.8%** | **80.8 / 73.8%** | **52.8%** | **82.4 / 79.8%** | **71.6** / 56.5% | **82.5%** | **32.9** / 46.6 | **1.0** / 4.2 |

- **Findings:** context keeps paying: +1.2 pt overall, +3 pt on inversions, Billboard steadier (blips
  −40%). Bollywood's Steady display changes a little more often (46.6/min vs labels' 41). The tuning
  cliff is unchanged (Billboard −30 cents 33.6%). Bohemian Rhapsody (Isophonics labels) Steady 69.8%
  (v11 68.0%). CPU 3.6 ms/frame vs 2.7 ms for v11 in the same run (p99 5.0 ms, budget 40 ms).
- **Decision:** deployed **E38 as v12** (ONNX parity 3.10e-06).

### F12 · Where does longer context help? · 2026-11-13
- **Script:** [`analysis_context_position.py`](../experiments/analysis_context_position.py). Accuracy by time
  into the song, and (after 32 s, where every model has a full window) on passages whose 4-chord
  progression already occurred earlier ("repeated") vs not ("first").

  | v12 − v10 (pt) | 0–5 s | 5–16 s | 16–32 s | 32–64 s | > 64 s | first / repeated (after 32 s) |
  |---|---|---|---|---|---|---|
  | Billboard | +1.2 | +1.4 | +0.1 | +2.7 | +3.5 | +3.0 / +3.5 |
  | Bollywood | +0.6 | +2.4 | +1.6 | +1.9 | +3.8 | +3.5 / +3.5 |

- **Findings:** the gain grows with time into the song and is the same on first and repeated
  passages. The long window supplies song-wide context (key, timbre, harmonic vocabulary), not
  recall of repeats, which a compact running summary could carry at constant cost (planned E41).

### E39 · 64 s attention context · 2026-11-13 — `models/e39_ctx1600`
- **Script:** [`2026-11-13_context1600.ps1`](../experiments/2026-11-13_context1600.ps1). v12's recipe with
  `--context 1600`, batch 4, ~10 min/epoch (vs 73 s for E38; RAM-bound on this machine).

  | majmin | BB clean / full | BB no bass < 150 / < 300 | inverted | Bolly clean / full | Bolly no bass < 150 / < 300 | **overall** | Steady changes/min (BB / Bolly) | Steady blips/min |
  |---|---|---|---|---|---|---|---|---|
  | v12 | 82.5 / 72.8% | 80.8 / 73.8% | 52.8% | 82.4 / 79.8% | 71.6 / 56.5% | 82.5% | 32.9 / 46.6 | 1.0 / 4.2 |
  | E39 | **83.0 / 73.7%** | **81.0** / 74.7% | **55.3%** | **83.0 / 80.7%** | **72.4 / 58.6%** | **83.0%** | 31.7 / 49.1 | 1.0 / 4.8 |

- **Findings:** still improving but with diminishing returns: 5 → 16 s +2.0 pt overall, 16 → 32 s
  +1.2, 32 → 64 s +0.5. The tuning cliff is unchanged (Billboard −30 cents 36.1%). Per-step attention
  work and KV cache double again.
- **Decision:** not deployed for now: +0.5 pt does not justify doubling per-step cost, mainly in the
  browser (W01). The context direction continues as E41 (short window + running summaries).

### D02 · Tuning-blind alignment check, and ChoCo songs · 2026-11-13
- **Bug:** `youtube.align` verifies a download with a chroma key-margin test, pooling CQT bins
  (2k, 2k+1) per pitch class. For a recording about 50 cents flat the energy straddles pitch classes
  and the margin collapses, so flat recordings were rejected from all training data. **Fix:**
  `align_tuned` tries tunings 0, −50, −25, +25 cents and keeps the best margin; the other checks
  (score, drift) are unchanged. `realign_rejected` re-checks earlier rejects.
- **Billboard:** 83 of 272 rejected songs recovered (tunings −50: 69, −25: 11, +25: 3) into
  `data/billboard_rec` (train-only prefix `bbrec`). Measured tuning of 21 of them: mean −11.9 cents
  (12 below −10), vs +3 for songs that passed before.
- **ChoCo** ([`data/choco.py`](../src/chordstream/data/choco.py), `chordstream fetch-choco`): the
  audio-aligned human annotations of Isophonics, USPop2002, Robbie Williams and JAAH (547 songs after
  removing Billboard test duplicates and the Bohemian Rhapsody test song). Recordings found and
  verified on YouTube: 311 accepted, +50 recovered by the tuning-aware re-check = **361 songs
  (~22 h)**, train-only prefix `choco`. Most remaining rejects fail the drift check (JAAH jazz takes).
- **Next:** E40 = v12's recipe + these 444 songs ([`2026-11-13_choco.ps1`](../experiments/2026-11-13_choco.ps1)).

### W01 · Browser-only version (GitHub Pages) · 2026-11-13
- **What:** the live engine ported to JavaScript ([`web/engine.js`](../src/chordstream/web/engine.js)):
  causal CQT, ONNX step model through onnxruntime-web (WebGPU, WebAssembly fallback) in a Web Worker,
  causal HMM, beat tracker, key spelling, chord-sheet commit and Steady display. The page is the same
  `index.html`, talking to the worker through a small socket-like wrapper; lyrics come from LRCLIB
  directly (`lyrics.js`). `chordstream export-web` writes the bundle;
  `scripts/build_site.py` + `.github/workflows/pages.yml` deploy it.
- **Parity** ([`tests/web/parity.mjs`](../tests/web/parity.mjs), 90 s of a Billboard song, 2250 frames, WebAssembly):
  every chord, raw argmax, Steady display, sheet commit, beat, BPM and key **identical** to the
  Python engine; max abs error CQT 2.4e-6, probabilities 1.0e-6.
- **Speed:** the step model first took 52.6 ms per frame in WebAssembly (Node, 1 thread) vs 3.6 ms
  natively. The streaming step ran the CNN over all 13 input frames although only the newest
  output is used; `StreamingStep.embed_last` now computes just that frame's cone (13 → 9 → 5 → 1
  frames per block; identical output, about 3.5x less CNN work). WebAssembly then: 33 ms per frame in
  Node, **21 ms in Chromium** (18.5 ms in the page while playing a file), inside the 40 ms budget.
  WebGPU (KV caches kept on the GPU between steps): GPU kernel time ~2 ms per step and decisions
  identical to Python (250 frames, max probability error 1.4e-6), but wall time could not be
  measured on this laptop while training held the GPU at 97% (~400 ms per step, waiting behind CUDA).
  On a phone (user report): **WebGPU 40 ms per step vs WebAssembly 15 ms.** One step is tiny (one
  frame, ~2 ms of math, ~150 small kernels), so per-dispatch overhead and the GPU→CPU readback of
  the probabilities every 40 ms dominate; the CPU has neither. The Windows "GPU %" seen while the
  page ran was the E40 training job plus the copy engine that hybrid-graphics laptops use to move
  browser frames to the integrated GPU (the same with the model stopped); the model's own 3D/compute
  use was ~1%.
- **Backend choice:** first an automatic choice (time a few steps on each backend at load, keep the
  faster; 5–10 s to load), then, at the user's request, a CPU / GPU toggle in the model badge
  (default CPU, remembered per device, locked while listening): load 2–3 s. The badge shows the live
  compute time per step, red above the 40 ms budget. `Engine.load(..., {provider: 'auto'})` still
  exists (used by bench.html-style checks).
- **Phones and system audio:** mobile browsers have no `getDisplayMedia`, so a page cannot hear other
  apps there. The button now says so (it used to fail with "Cancelled (TypeError)") and points to
  Mic or Play file.
- CI runs `tests/web/syntax.mjs` (every inline page script parses) before deploying.

### W06 · Can speech-to-text sync the lyric sheet? · 2026-11-14
- Question: can a small speech model hear enough sung words to place the playing position in the
  lyrics? Test (`experiments/analysis_lyric_sync.py`, set from `lyric_eval_set.py`): 8 s windows
  from test songs with synced lyrics (420 English windows, 273 Hindi). Recognised words are matched
  phonetically (Hindi lyrics romanised, IAST with schwa deletion) and vote on an offset. A window is
  correct if it locks within ±1.5 s, wrong if it locks elsewhere, otherwise "no lock".

| Engine (CPU) | English correct / wrong | Hindi correct / wrong | s per 8 s window |
|---|---|---|---|
| Vosk (grammar = song's lyrics) | 17.6% / 9.8% | 0.7% / – | – |
| Web Speech (Chrome, cloud) | 18.8% / 28.1% | 18.8% / 37.5% | live |
| Whisper tiny | 33.6% / 28.1% | 3.3% / 16.1% | 1.6 (en), 5.2 (hi) |
| Whisper base | 41.2% / 36.0% | 4.8% / 30.0% | 3.6, 9.2 |
| Whisper small | 43.8% / 34.3% | 13.6% / 46.5% | 5.4, 15.7 |

- Verdict: **not shipped.** Even the best (Whisper small, 244M parameters, too slow to run
  alongside the chord model in a browser) is wrong about as often as it is right on English, and
  worse than right on Hindi: sung words over a full mix are mostly misheard. A wrong lock moves the
  sheet to the wrong line, which is worse than the current character-position timing. A better
  route, if revisited: match the recognised chord sequence against the song's chord sheet (no
  vocals needed), or use a vocal-separated input.

### W07 · "How it works" section on the site · 2026-11-14
- A card below the app: the pipeline (audio → note energies → CNN → Transformer → HMM → chord), one
  short card per component, how it was trained, how it differs from whole-song tools and classic
  live detectors, and headline numbers (~82% on unseen songs, 109 chord types, ~2M parameters,
  15–25 ms per step). Same content as the README's walkthrough, condensed for visitors.

### W05 · Chord sheet polish · 2026-11-14
- Chords starting up to 0.5 s before a lyric line now go on that line's first word (they used to
  sit alone at the end of the previous line, because chords land a beat ahead of the syllable and
  lyric timestamps are approximate). Progression chips no longer show "-1:-0.0" for a first chord
  that starts a few ms before 0.

### W04 · Renamed Rubato · 2026-11-14
- The app is now **Rubato** (rubato: expressive, flexible tempo; the app follows real playing).
  Logo "Wave": three chord tones riding a stretched line (`scripts/make_icons.py` -> `web/icons/`:
  SVG, favicon, home-screen and maskable icons, link-preview image). The repo moved to
  `helloparthshah/rubato` and the site to https://helloparthshah.github.io/rubato/ (the old
  /chordstream/ address no longer resolves); the Python package and CLI keep the name `chordstream`. The site is installable (web manifest) and caches the
  page, engine and model in a service worker, so repeat visits load at once and work offline.
- Name check (2026-11-14): no chord app called Rubato; nearby is the sequencer "Fugue Machine
  Rubato" (iOS/macOS, 2025), so a store listing would want "Rubato Chords".

### W03 · Display decoupled from the engine · 2026-11-14
- The timeline's "now", the beat pulse and the big-chord updates were driven by engine events, so a
  slow engine (e.g. a busy GPU) stalled the animations, and sheet chords were placed relative to
  event arrival, i.e. shifted later by the engine's lag. Now an audio clock (16 kHz samples captured,
  interpolated) drives drawing once per animation frame; events only update state. The pulse is
  extrapolated from the tracker's last beat and tempo; the unanswered stretch of the timeline is a
  faint band; sheet placement uses audio-clock age. Checked with the GPU 9 s behind (timeline and
  bars keep moving) and on the CPU (engine within 0.04 s of the clock, chords match player time).

### W02 · Live server retired · 2026-11-14
- With the browser version live, the Python live server was removed at the user's request: the
  WebSocket engine endpoint, Whisper word alignment (`asr.py`, `serve --asr`), Windows "now playing"
  (`nowplaying.py`), the server-side LRCLIB proxy (`lyrics.py`), the Docker image and the LAN tools
  (`tools/`). The page is browser-only; the sheet times words by character position and looks lyrics
  up from a played file's name or the search box. Browser Whisper (transformers.js, tiny/base) and
  mic song recognition (AudD/ACRCloud, paid key + proxy) were considered and deferred.
- `chordstream serve` remains only for the `/annotate` gold-label tool. The Python engine
  (`stream.py`) remains for `transcribe`, `mic` and the browser parity reference.

### D01 · Key-aware chord spelling (display) · 2026-11-12
- Chord classes had one fixed spelling (A♭m in E major, C♯ in A♭ major). `spelling.KeySpeller`
  keeps a ~30 s decaying histogram of the chords' notes, picks the best-fitting major-scale key
  signature after 8 s (switching only when another fits 10% better) and spells roots with its
  sharps or flats. Events carry `key`; the UI shows it next to the BPM. Recognition unchanged.

---

## Queue

| ID | Experiment | Question | Notes |
|---|---|---|---|
| later | More Bollywood / Indian labelled data | Depends on E21 | **Survey (2026-11-12): no public time-aligned Bollywood chord dataset exists.** ChoCo's 18 sources are Western/Japanese pop, jazz, classical; Saraga (CompMusic) is Indian art music without chords; Chordonomicon (667k songs) is untimed UG scrapes without titles/artists (CC-BY-NC); Hooktheory/TheoryTab is time-aligned and human but has only a handful of Indian songs (e.g. *Kal Ho Naa Ho*, *Jai Ho*), gated, non-commercial. Best levers: scale the E21 pipeline (more years/playlists; ~37% of songs currently find a usable sheet), other human sheet sites (e.g. Acousterr; check terms), and a small hand-corrected gold test set |
| E21 scale-up | More Bollywood songs ([`2026-11-12_bollywood_scale.ps1`](../experiments/2026-11-12_bollywood_scale.ps1)) | Does more sheet-aligned data help further? | 2000–2024, 3 playlists × 40 songs per year, full pipeline → `data/cache_bolly3`. **Done, see E28: 213 songs, no gain over v8.** Other sheet sites: Acousterr's terms forbid automated access (§5), so not used; UG's terms allow personal, non-commercial access (§6.3), which this is (sheets stay local) |
| E21 gold | Hand-corrected test labels (`/annotate` page, `chordstream.gold`) | How accurate are models on Bollywood with exact labels? | Built 2026-11-12; the benchmark uses a song's gold labels once marked done. Needs a person to annotate (~10–15 songs). Now lists the 61 v3 test songs. The next useful step for Bollywood, since more aligned data stopped helping (E28) |


## Model directory registry

| `models/` | From | Status |
|---|---|---|
| `latest` | E01 | init for all models (keep) |
| `bb_ft`, `v2`, `v3`, `v4` | E03, E09, E12, E15 | superseded; kept for comparison |
| `v5` | E17 | previous live model, release `model-v5` |
| `v6` | E18 | previous live model, release `model-v6` |
| `v7` | E25 | previous live model, release `model-v7` (= `e25a_ext_delay3` + HMM 5 s / power 2) |
| `e25a_ext_delay3`, `e25b_ext_delay6` | E25 | extended label-delay models |
| `e26a_sheet8`, `e26b_sheet12` | E26 | rejected; safe to delete |
| `e27_rare` | E27 | rare-chord mix (150 renders); not deployed |
| `e27b_rare50` | E27b | rare-chord mix (50 renders); candidate |
| `e21a_bolly`, `e21b_bolly_ft` | E21 | Bollywood mix (scratch / fine-tune); Billboard regression, not deployed |
| `e21a_live` | E21 | `e21a_bolly` + HMM 5 s / power 2, exported; was on trial in the UI; safe to delete |
| `v8` | E21e | previous live model, release `model-v8` (= `e21e_bolly_clean` + HMM 5 s / power 2) |
| `v9` | E28b | previous live model, release `model-v9` (= `e28b_bolly3x2` + HMM 5 s / power 2) |
| `v10` | E32b | previous live model, release `model-v10` (= `e32b_eq5_bass3` + HMM 5 s / power 2) |
| `v12` | E38 | **live**, release `model-v12` (= `e38_ctx800` + HMM 5 s / power 2) |
| `v11` | E34b | previous live model, release `model-v11` (= `e34b_ctx400_eq` + HMM 5 s / power 2) |
| `e38_ctx800` (+ `_h5p2`) | E38 | 32 s context; = v12 |
| `e39_ctx1600` (+ `_h5p2`) | E39 | 64 s context; +0.5 pt, not deployed (cost) |
| `e35_tune`, `e35b_tune25` (+ `_h5p2`) | E35 | detuned training copies; not deployed |
| `e36a_early1`, `e36b_early3` (+ `_h5p2`) | E36 | early-switch penalty; not deployed |
| `e37a_bins24`, `e37b_bins36`, `e37c_bins36_micro` (+ `_h5p2`) | E37 | small 36-bin pilot; safe to delete |
| `e33a_raw_init`, `e33b_raw_scratch` | E33 | raw-waveform front end; not deployed |
| `e34_ctx400`, `e34b_ctx400_eq` (+ `_h5p2`) | E34 | 16 s context; e34b = v11 |
| `e31a_bassdrop3`, `e31b_bassdrop6`, `e32a_eq5`, `e32b_eq5_bass3` (+ `_h5p2`) | E31, E32 | bass dropout / random EQ; e32b = v10 |
| `e21c_bolly_mask3`, `e21d_bolly_mask1`, `e21e_bolly_clean`, `e21f_bolly_clean_rare` | E21 | Bollywood variants (e21e = v8) |
| `e28a_bolly3x1`, `e28b_bolly3x2` (+ `_h5p2` eval copies) | E28 | scaled Bollywood mixes; e28b = v9 |
| `e29_room` (+ `_h5p2`) | E29 | room-echo copies; not deployed |
| `e30a_stab2`, `e30b_stab5` (+ `_h5p2`) | E30 | stability loss with delay; not deployed |
| `teacher` | E17 | offline teacher for majmin distillation |
| `exp_a`, `exp_b`, `exp_w2/3/5`, `exp_g10`, `synth_g10`, `exp_lr3`, `exp_big`, `synth_big`, `v4_clean`, `v4_orig`, `student` | E08–E17 | experiment outputs; safe to delete |
| `ext_teacher` | E18 | offline teacher for extended-vocabulary distillation |
| `ext_student` | E18 | same weights as `v6` (before HMM retune); safe to delete |
| `e19_hpss`, `e20_key`, `e22a_stability` | E19, E20, E22 | experiment outputs; safe to delete |
| `e23a_delay3`, `e23b_delay6` | E23 | label-delay majmin models (e23a exported to ONNX) |
| `e24_ext_nodistill` | E24 | non-distilled extended baseline |
