# E42: beat phase as a Transformer input (user idea). The causal beat tracker's phase (sin/cos, scaled
# by confidence, + confidence) is projected to d_model and added after the CNN. v12's data and recipe,
# so beat input is the only change. a: delay 120 ms; b: no delay (can rhythm replace lookahead?);
# c: control, no beat and no delay. Waits for E40b and the beat cache.
$ErrorActionPreference = 'Continue'
function Log($m) { "=== $m $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log }
while (-not (Select-String -Path data/logs/batch.log -Pattern 'e40b done' -Quiet)) { Start-Sleep 120 }
while ((Get-ChildItem data/beat_cache/*.npy -ErrorAction SilentlyContinue).Count -lt 1100) { Start-Sleep 60 }
foreach ($run in @(@('e42a_beat_d3', '--beat-input', 3), @('e42b_beat_d0', '--beat-input', 0), @('e42c_nobeat_d0', '', 0))) {
    $name, $flag, $delay = $run
    Log $name
    $extra = @(); if ($flag) { $extra += $flag }
    & uv run --no-sync chordstream train --init models/latest --lr 3e-4 --epochs 80 --crops 2000 --workers 0 --batch-size 8 `
        --transition-weight 3 --eval-every 10 --vocab extended --label-delay $delay --cache data/cache_ext_bolly3x2 `
        --context 800 --random-eq 0.5 --bass-dropout 0.3 @extra --out "models/$name" 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath "data/logs/$name.log"
    Log "$name eval"
    uv run --no-sync python experiments/hmm55.py "models/$name"
    uv run --no-sync python experiments/analysis_bollywood.py --v3 "models/${name}_h5p2" 2>&1 | Where-Object { $_ -match 'aligned' } | Tee-Object -FilePath "data/logs/${name}_eval.log"
    uv run --no-sync python experiments/analysis_bass.py "models/${name}_h5p2" 2>&1 | Where-Object { $_ -match 'Billboard|Bollywood' } | Tee-Object -Append -FilePath "data/logs/${name}_eval.log"
    uv run --no-sync python experiments/analysis_raw_flicker.py "models/${name}_h5p2" 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -Append -FilePath "data/logs/${name}_eval.log"
}
Log 'e42 done'
