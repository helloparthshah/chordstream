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
| `data/cache_bolly` | `chordstream bollywood cache` | Bollywood songs, sheet-aligned labels (E21) |

**Analyses** (`analysis_*.py`, `uv run --no-sync python experiments/<script> ...`; usage in each docstring):

| script | entry | what it measures |
|---|---|---|
| `analysis_fixed_lag.py` | E23–E26 | per-model fixed-lag decoding and HMM grid; caches posteriors in `data/analysis/` |
| `analysis_paired.py` | E23 | paired per-song significance at equal delay |
| `analysis_speculative.py` | F02 | draft + verifier display rules |
| `analysis_beat_sheet.py` | F03 | chord-sheet sources: beat pooling, beat vote, HMM path |
| `analysis_sheet_align.py`, `analysis_sheet_align_grid.py` | E21 | accuracy of sheet-to-audio alignment on Billboard |
