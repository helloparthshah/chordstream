# E27b: lighter rare-chord mix (50 renders, ~12% of songs) after E27 (150) cost 0.5 pt overall.
$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force data/logs, data/cache_ext_rare50 | Out-Null
Get-ChildItem data/cache_ext/*.npz | ForEach-Object { $d = "data/cache_ext_rare50/$($_.Name)"; if (-not (Test-Path $d)) { New-Item -ItemType HardLink -Path $d -Target $_.FullName | Out-Null } }
Get-ChildItem data/cache_rare/*.npz | Sort-Object Name | Select-Object -First 50 | ForEach-Object { $d = "data/cache_ext_rare50/$($_.Name)"; if (-not (Test-Path $d)) { New-Item -ItemType HardLink -Path $d -Target $_.FullName | Out-Null } }
"=== e27b_rare50 $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
& uv run --no-sync chordstream train --init models/latest --lr 3e-4 --epochs 80 --crops 2000 --workers 0 --transition-weight 3 `
    --eval-every 10 --cache data/cache_ext_rare50 --vocab extended --label-delay 3 --out models/e27b_rare50 2>&1 |
    Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath data/logs/e27b_rare50.log
"=== e27b done $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
