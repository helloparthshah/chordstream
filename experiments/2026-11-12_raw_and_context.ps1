# E33a/b: learned raw-waveform front end instead of the CQT (experiments/train_raw.py; v9 data and
# recipe). a: ChordNet initialised from models/latest, front end random. b: everything from scratch.
# E34: v9 recipe with 400 frames (16 s, ~8 bars at 120 BPM) of attention context instead of 128 (5 s);
# batch 16 to fit the longer crops in 4 GB.
$ErrorActionPreference = 'Continue'
function Log($m) { "=== $m $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log }
Log 'e33a_raw_init'
uv run --no-sync python experiments/train_raw.py models/e33a_raw_init --init models/latest 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath data/logs/e33a_raw_init.log
Log 'e33b_raw_scratch'
uv run --no-sync python experiments/train_raw.py models/e33b_raw_scratch 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath data/logs/e33b_raw_scratch.log
Log 'e34_ctx400'
& uv run --no-sync chordstream train --init models/latest --lr 3e-4 --epochs 80 --crops 2000 --workers 0 --batch-size 16 `
    --transition-weight 3 --eval-every 10 --vocab extended --label-delay 3 --cache data/cache_ext_bolly3x2 `
    --context 400 --out models/e34_ctx400 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath data/logs/e34_ctx400.log
Log 'e33/e34 done'
