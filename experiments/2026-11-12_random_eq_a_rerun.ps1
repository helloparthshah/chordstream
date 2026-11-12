# E32a rerun (the first run died with CUDA out of memory while an evaluation shared the GPU).
$ErrorActionPreference = 'Continue'
"=== e32a_eq5 rerun $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
& uv run --no-sync chordstream train --init models/latest --lr 3e-4 --epochs 80 --crops 2000 --workers 0 `
    --transition-weight 3 --eval-every 10 --vocab extended --label-delay 3 --cache data/cache_ext_bolly3x2 `
    --random-eq 0.5 --out models/e32a_eq5 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath data/logs/e32a_eq5.log
"=== e32a rerun done $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
