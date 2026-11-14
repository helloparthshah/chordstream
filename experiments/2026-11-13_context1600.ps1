# E39: 64 s attention context. E34 (5 -> 16 s, +1.9 pt overall) and E38 (16 -> 32 s, +1.2 pt, v12) both
# gained; does 64 s? v12 recipe, --context 1600, batch 4 (same frames per step).
$ErrorActionPreference = 'Continue'
function Log($m) { "=== $m $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log }
$name = 'e39_ctx1600'
Log $name
& uv run --no-sync chordstream train --init models/latest --lr 3e-4 --epochs 80 --crops 2000 --workers 0 --batch-size 4 `
    --transition-weight 3 --eval-every 10 --vocab extended --label-delay 3 --cache data/cache_ext_bolly3x2 `
    --context 1600 --random-eq 0.5 --bass-dropout 0.3 --out "models/$name" 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath "data/logs/$name.log"
Log "$name eval"
uv run --no-sync python experiments/hmm55.py "models/$name"
uv run --no-sync python experiments/analysis_bollywood.py --v3 "models/${name}_h5p2" 2>&1 | Where-Object { $_ -match 'aligned' } | Tee-Object -FilePath "data/logs/${name}_eval.log"
uv run --no-sync python experiments/analysis_bass.py "models/${name}_h5p2" 2>&1 | Where-Object { $_ -match 'Billboard|Bollywood' } | Tee-Object -Append -FilePath "data/logs/${name}_eval.log"
uv run --no-sync python experiments/analysis_raw_flicker.py "models/${name}_h5p2" 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -Append -FilePath "data/logs/${name}_eval.log"
uv run --no-sync python experiments/analysis_detune.py "models/${name}_h5p2" 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath "data/logs/f11_$name.log"
Log 'e39 done'
