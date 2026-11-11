# Experiment scripts

Batch scripts behind the entries in [`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md), named
`<date>_<topic>.ps1`. Run them from the repository root, e.g.
`powershell -File experiments/2026-11-11_v4.ps1`. Each step logs to `data/logs/<step>.log`
and appends a start marker to `data/logs/batch.log`.

They assume the caches built by `chordstream features` / `audit` / `build_extended_cache`
(`data/cache_bb`, `data/cache_bb_fix`, `data/cache_ext`) and the synthetic init `models/latest`.
