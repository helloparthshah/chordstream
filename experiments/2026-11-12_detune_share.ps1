# E35b: E35 drew ~60% of its crops from detuned tunex__ copies (sampling by duration) and lost 13 pt on
# in-tune Bollywood (81.0 -> 67.8%) while fixing the Billboard flat cliff. Same recipe, detuned copies
# capped at 25% of crops (--tune-share 0.25).
$ErrorActionPreference = 'Continue'
function Log($m) { "=== $m $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log }
$name = 'e35b_tune25'
Log $name
& uv run --no-sync chordstream train --init models/latest --lr 3e-4 --epochs 80 --crops 2000 --workers 0 --batch-size 16 `
    --transition-weight 3 --eval-every 10 --vocab extended --label-delay 3 --cache data/cache_ext_bolly3x2_tune `
    --context 400 --random-eq 0.5 --bass-dropout 0.3 --tune-share 0.25 --out "models/$name" 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath "data/logs/$name.log"
Log "$name eval"
uv run --no-sync python experiments/hmm55.py "models/$name"
uv run --no-sync python experiments/analysis_detune.py "models/${name}_h5p2" 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath "data/logs/f11_$name.log"
uv run --no-sync python experiments/analysis_bollywood.py --v3 "models/${name}_h5p2" 2>&1 | Where-Object { $_ -match 'aligned' } | Tee-Object -FilePath "data/logs/${name}_eval.log"
uv run --no-sync python experiments/analysis_bass.py "models/${name}_h5p2" 2>&1 | Where-Object { $_ -match 'Billboard|Bollywood' } | Tee-Object -Append -FilePath "data/logs/${name}_eval.log"
uv run --no-sync python experiments/analysis_raw_flicker.py "models/${name}_h5p2" 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -Append -FilePath "data/logs/${name}_eval.log"
Log 'e35b done'
