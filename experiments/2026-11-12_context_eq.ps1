# E34b: E34 (16 s attention context) + v10's robustness augmentation (random EQ 0.5, bass dropout 0.3).
# E34 gained ~+2 pt clean over v9 but lost v10's no-bass robustness, since it trained without them.
$ErrorActionPreference = 'Continue'
"=== e34b_ctx400_eq $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
& uv run --no-sync chordstream train --init models/latest --lr 3e-4 --epochs 80 --crops 2000 --workers 0 --batch-size 16 `
    --transition-weight 3 --eval-every 10 --vocab extended --label-delay 3 --cache data/cache_ext_bolly3x2 `
    --context 400 --random-eq 0.5 --bass-dropout 0.3 --out models/e34b_ctx400_eq 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath data/logs/e34b_ctx400_eq.log
"=== e34b done $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
