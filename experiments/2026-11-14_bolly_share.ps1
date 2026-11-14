# E40b: E40's data (v12 set + 444 ChoCo / recovered Billboard songs) with Bollywood's share of training
# crops pinned to its v12 level (52%; it fell to 32% in E40, and Bollywood dropped 2.4 pt).
$ErrorActionPreference = 'Continue'
function Log($m) { "=== $m $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log }
$name = 'e40b_bolly52'
Log $name
& uv run --no-sync chordstream train --init models/latest --lr 3e-4 --epochs 80 --crops 2000 --workers 0 --batch-size 8 `
    --transition-weight 3 --eval-every 10 --vocab extended --label-delay 3 --cache data/cache_ext_bolly3x2_choco `
    --context 800 --random-eq 0.5 --bass-dropout 0.3 --share bollytrain=0.52 --out "models/$name" 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath "data/logs/$name.log"
Log "$name eval"
uv run --no-sync python experiments/hmm55.py "models/$name"
uv run --no-sync python experiments/analysis_bollywood.py --v3 "models/${name}_h5p2" 2>&1 | Where-Object { $_ -match 'aligned' } | Tee-Object -FilePath "data/logs/${name}_eval.log"
uv run --no-sync python experiments/analysis_bass.py "models/${name}_h5p2" 2>&1 | Where-Object { $_ -match 'Billboard|Bollywood' } | Tee-Object -Append -FilePath "data/logs/${name}_eval.log"
Log 'e40b done'
