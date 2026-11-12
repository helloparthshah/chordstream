# Experiment scripts

Batch scripts and analyses behind the entries in [`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md).
Run everything from the repository root.

**Batch training scripts** (`<date>_<topic>.ps1`), e.g. `powershell -File experiments/2026-11-12_ext_delay.ps1`.
Each step logs to `data/logs/<step>.log` and appends a start marker to `data/logs/batch.log`.
They assume the synthetic init `models/latest` and the caches built by the CLI:

| cache | built by | contents |
|---|---|---|
| `data/cache_bb` | `chordstream features data/billboard` | Billboard songs, majmin labels |
| `data/cache_bb_fix` | `chordstream audit` + `chordstream clean` | re-timed labels, drifting songs removed (E14) |
| `data/cache_ext` | `chordstream build-ext` | same songs, 109-class labels + root/bass/notes (E18) |
| `data/cache_rare` | `chordstream synth --rare` + `features --extended synthx` | rare-chord renders, training only (E27) |
| `data/cache_ext_rare`, `data/cache_ext_rare50` | experiment scripts (hard links) | `cache_ext` + 150 / 50 rare renders (E27, E27b) |
| `data/cache_bolly` | `chordstream bollywood cache` | Bollywood songs, sheet-aligned labels (E21) |
| `data/cache_bolly2` | `chordstream bollywood clean` | cleaned labels: bass-aware, capo key, least-confident 30% ignored (E21) |
| `data/cache_ext_bolly2`, `data/cache_ext_bolly2_rare` | `2026-11-12_bollywood_clean.ps1` | `cache_ext` + cleaned Bollywood ×3 (+ 50 rare renders) (E21e/f) |
| `data/cache_bolly3` | `2026-11-12_bollywood_scale.ps1` | E21 scale-up: 213 cleaned songs (61 test) (E28; v3 benchmark) |
| `data/cache_room` | `build_room_cache.py` | room-echo (+ half phone band / noise) copies of the 483 training songs, `roomx__` (E29) |
| `data/cache_ext_bolly3x2_room` | `2026-11-12_room_echo.ps1` | `cache_ext_bolly3x2` + `cache_room` (E29) |
| `data/cache_ext_bolly3x1`, `data/cache_ext_bolly3x2` | `2026-11-12_bollywood_scaled_train.ps1` | `cache_ext` + 152 Bollywood training songs ×1 / ×2 (E28) |
| `data/cache_ext_bolly` | `2026-11-12_bollywood.ps1` (hard links) | `cache_ext` + trusted Bollywood training songs ×3 (E21) |
| `data/cache_bolly_masked`, `data/cache_ext_bollym3/1` | `bollywood.mask_boundaries`, `2026-11-12_bollywood_masked.ps1` | Bollywood labels ignored ±320 ms around aligned changes; mixes at 3× / 1× (E21c/d) |

**Analyses** (`analysis_*.py`, `uv run --no-sync python experiments/<script> ...`; usage in each docstring):

| script | entry | what it measures |
|---|---|---|
| `analysis_fixed_lag.py` | E23–E26 | per-model fixed-lag decoding and HMM grid; caches posteriors in `data/analysis/` |
| `analysis_paired.py` | E23 | paired per-song significance at equal delay |
| `analysis_speculative.py` | F02 | draft + verifier display rules |
| `analysis_beat_sheet.py` | F03 | chord-sheet sources: beat pooling, beat vote, HMM path |
| `analysis_sheet_align.py`, `analysis_sheet_align_grid.py` | E21 | accuracy of sheet-to-audio alignment on Billboard |
| `analysis_bollywood.py` | E21 | Bollywood test songs: time in sheet, sequence coverage, changes, aligned accuracy (`--v3`: the 61 scaled test songs) |
| `analysis_flicker.py` | F04 | live-readout flicker: changes/min, short segments, A→B→A blips and their chord pairs |
| `analysis_flicker_fixes.py`, `analysis_evidence_grid.py` | F05 | display rules (steady / evidence hold) and pitch-shift averaging, scored at display time |
| `analysis_degraded.py` | F06 | Billboard test songs with room echo / phone band / noise: accuracy and flicker |
| `analysis_bass.py` | F07 | bass dependence: high-passed audio, low-bass frames, inverted chords |
| `hmm55.py` | E28 | copy a model to `<dir>_h5p2` with the deployed HMM (5 s / power 2) for evaluation |
| `analysis_per_type.py` | E18, E27 | per-chord-type recall with a common HMM |
| `analysis_label_cleaning.py` | E21 | aligner variants on Billboard: bass evidence, beat snapping, confidence mask |
