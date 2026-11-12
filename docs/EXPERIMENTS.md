# Experiment log

Every experiment run on chordstream, in order: what was tried, how, what happened and what was
decided. Each entry is reproducible from the command shown. Batch scripts are in
[`experiments/`](../experiments/). Model bundles live in `models/` (git-ignored); the ones in
use are published as [GitHub Releases](https://github.com/helloparthshah/chordstream/releases).

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
| v5 | was live ([release](https://github.com/helloparthshah/chordstream/releases/tag/model-v5)) | 78.9% | 80.8% (36 clean) | 1.6× | +80 ms | E17 |
| v6 (extended vocab) | was live ([release](https://github.com/helloparthshah/chordstream/releases/tag/model-v6)) | 79.9% | 80.8% (36 clean); 70.3% full vocab | 1.6× | +80 ms | E18 |
| teacher (offline, sees ±5 s) | never | 82.0% | 82.9% (36 clean) | 1.04× | – | E17 |
| E23b (label delay 240 ms, human labels) | candidate | 83.0% | 83.9% (36 clean) | 1.44× | +240 ms display | E23 |
| E24 (extended, human labels) | baseline | 80.9% majmin | 81.0% majmin, 70.7% full (36 clean) | 1.71× | +80 ms | E24 |
| **v7** (extended, 120 ms label delay, human labels) | **live** ([release](https://github.com/helloparthshah/chordstream/releases/tag/model-v7)) | **83.1%** majmin | **83.3% majmin, 73.0% full (36 clean)** | 1.48× | 120 ms display | E25 |

Biggest wins: real data (E03, +28 pts over synthetic only), training budget (E13/E15, +2.4),
a trained-in 120 ms label delay (E23/E25, +2.5 majmin and fewer flips), more data (E08, +2),
transition loss (E09, lag 120 → 40–80 ms), and a 480 ms hold for the chord sheet (F03, changes
1.6× → 1.03×). Distillation gave +0.7 (E17) but is no longer used (policy above; E24 showed no
gain on the extended model).
Things that did **not** help: beat or key priors in the HMM, a faster frontend, a bigger model
at the old learning rate, re-timing labels (for training), data beyond ~200 songs at the old
training budget, drum suppression (E19), a stability loss (E22), draft/verify "speculative"
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
  [`model-v6`](https://github.com/helloparthshah/chordstream/releases/tag/model-v6)). Sus4 is weak and
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
  [`model-v7`](https://github.com/helloparthshah/chordstream/releases/tag/model-v7)): +2.5 pt majmin,
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
  train on the cleaned labels (×3), f also with E27b's 50 rare renders. Running.
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

### D01 · Key-aware chord spelling (display) · 2026-11-12
- Chord classes had one fixed spelling (A♭m in E major, C♯ in A♭ major). `spelling.KeySpeller`
  keeps a ~30 s decaying histogram of the chords' notes, picks the best-fitting major-scale key
  signature after 8 s (switching only when another fits 10% better) and spells roots with its
  sharps or flats. Events carry `key`; the UI shows it next to the BPM. Recognition unchanged.

---

## Queue

| ID | Experiment | Question | Notes |
|---|---|---|---|
| E21 | Bollywood (above) | Does training on sheet-aligned Bollywood songs help on Bollywood without hurting Billboard? | Training running |
| later | More Bollywood / Indian labelled data | Depends on E21 | **Survey (2026-11-12): no public time-aligned Bollywood chord dataset exists.** ChoCo's 18 sources are Western/Japanese pop, jazz, classical; Saraga (CompMusic) is Indian art music without chords; Chordonomicon (667k songs) is untimed UG scrapes without titles/artists (CC-BY-NC); Hooktheory/TheoryTab is time-aligned and human but has only a handful of Indian songs (e.g. *Kal Ho Naa Ho*, *Jai Ho*), gated, non-commercial. Best levers: scale the E21 pipeline (more years/playlists; ~37% of songs currently find a usable sheet), other human sheet sites (e.g. Acousterr; check terms), and a small hand-corrected gold test set |


## Model directory registry

| `models/` | From | Status |
|---|---|---|
| `latest` | E01 | init for all models (keep) |
| `bb_ft`, `v2`, `v3`, `v4` | E03, E09, E12, E15 | superseded; kept for comparison |
| `v5` | E17 | previous live model, release `model-v5` |
| `v6` | E18 | previous live model, release `model-v6` |
| `v7` | E25 | **live**, release `model-v7` (= `e25a_ext_delay3` + HMM 5 s / power 2) |
| `e25a_ext_delay3`, `e25b_ext_delay6` | E25 | extended label-delay models |
| `e26a_sheet8`, `e26b_sheet12` | E26 | rejected; safe to delete |
| `e27_rare` | E27 | rare-chord mix (150 renders); not deployed |
| `e27b_rare50` | E27b | rare-chord mix (50 renders); candidate |
| `e21a_bolly`, `e21b_bolly_ft` | E21 | Bollywood mix (scratch / fine-tune); Billboard regression, not deployed |
| `e21a_live` | E21 | `e21a_bolly` + HMM 5 s / power 2, exported; **on trial in the UI** (2026-11-12, user request); release stays `model-v7` |
| `teacher` | E17 | offline teacher for majmin distillation |
| `exp_a`, `exp_b`, `exp_w2/3/5`, `exp_g10`, `synth_g10`, `exp_lr3`, `exp_big`, `synth_big`, `v4_clean`, `v4_orig`, `student` | E08–E17 | experiment outputs; safe to delete |
| `ext_teacher` | E18 | offline teacher for extended-vocabulary distillation |
| `ext_student` | E18 | same weights as `v6` (before HMM retune); safe to delete |
| `e19_hpss`, `e20_key`, `e22a_stability` | E19, E20, E22 | experiment outputs; safe to delete |
| `e23a_delay3`, `e23b_delay6` | E23 | label-delay majmin models (e23a exported to ONNX) |
| `e24_ext_nodistill` | E24 | non-distilled extended baseline |
