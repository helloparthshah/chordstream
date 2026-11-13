# F11 + E35: tuning robustness. v11 collapses when a recording is > ~25 cents sharp. F11 measures accuracy
# on the test songs detuned by -45..+45 cents; E35 = v11's recipe plus two detuned copies (15-45 cents
# flat / sharp, tunex__ prefix, train-only) of every training song.
$ErrorActionPreference = 'Continue'
function Log($m) { "=== $m $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log }
if (-not (Test-Path data/logs/f11_v11.log)) {
Log 'f11 v11 detune'
uv run --no-sync python experiments/analysis_detune.py models/v11 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath data/logs/f11_v11.log
}
Log 'e35 tune cache'
uv run --no-sync python experiments/build_tune_cache.py data/cache_tune 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath data/logs/e35_tune_cache.log
$dir = 'data/cache_ext_bolly3x2_tune'
New-Item -ItemType Directory -Force $dir | Out-Null
foreach ($src in @('data/cache_ext_bolly3x2', 'data/cache_tune')) {
    Get-ChildItem "$src/*.npz" | ForEach-Object { $d = "$dir/$($_.Name)"; if (-not (Test-Path $d)) { New-Item -ItemType HardLink -Path $d -Target $_.FullName | Out-Null } }
}
Log 'e35_tune'
& uv run --no-sync chordstream train --init models/latest --lr 3e-4 --epochs 80 --crops 2000 --workers 0 --batch-size 16 `
    --transition-weight 3 --eval-every 10 --vocab extended --label-delay 3 --cache $dir `
    --context 400 --random-eq 0.5 --bass-dropout 0.3 --out models/e35_tune 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath data/logs/e35_tune.log
Log 'e35 eval'
uv run --no-sync python experiments/hmm55.py models/e35_tune
uv run --no-sync python experiments/analysis_detune.py models/e35_tune_h5p2 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath data/logs/f11_e35.log
uv run --no-sync python experiments/analysis_bollywood.py --v3 models/e35_tune_h5p2 2>&1 | Where-Object { $_ -match 'aligned' } | Tee-Object -FilePath data/logs/e35_eval.log
uv run --no-sync python experiments/analysis_bass.py models/e35_tune_h5p2 2>&1 | Where-Object { $_ -match 'Billboard|Bollywood' } | Tee-Object -Append -FilePath data/logs/e35_eval.log
Log 'e35 done'
