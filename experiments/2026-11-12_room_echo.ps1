# E29: v9's recipe plus one room-echo copy of every training song (data/cache_room, roomx__ prefix,
# train-only). Question: does it recover F06's 4.8 pt loss under room echo without costing clean audio?
$ErrorActionPreference = 'Continue'
"=== e29 room cache $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
uv run --no-sync python experiments/build_room_cache.py data/cache_room 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath data/logs/e29_room_cache.log
$dir = 'data/cache_ext_bolly3x2_room'
New-Item -ItemType Directory -Force $dir | Out-Null
foreach ($src in @('data/cache_ext_bolly3x2', 'data/cache_room')) {
    Get-ChildItem "$src/*.npz" | ForEach-Object { $d = "$dir/$($_.Name)"; if (-not (Test-Path $d)) { New-Item -ItemType HardLink -Path $d -Target $_.FullName | Out-Null } }
}
"=== e29_room $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
& uv run --no-sync chordstream train --init models/latest --lr 3e-4 --epochs 80 --crops 2000 --workers 0 `
    --transition-weight 3 --eval-every 10 --vocab extended --label-delay 3 `
    --cache $dir --out models/e29_room 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath data/logs/e29_room.log
"=== e29 done $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
