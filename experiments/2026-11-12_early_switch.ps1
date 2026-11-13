# E36 (user idea): penalise switching early. v11's recipe (E34b) plus -log(1 - p(next chord)) on the
# 0.5 s before every chord change, weight 1 (a) / 3 (b). The existing transition weight already
# up-weights the 0.5 s *after* a change (switch on time); this adds the other side (don't switch early).
$ErrorActionPreference = 'Continue'
function Log($m) { "=== $m $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log }
foreach ($r in @(@('e36a_early1', '1'), @('e36b_early3', '3'))) {
    $name = $r[0]
    Log $name
    & uv run --no-sync chordstream train --init models/latest --lr 3e-4 --epochs 80 --crops 2000 --workers 0 --batch-size 16 `
        --transition-weight 3 --eval-every 10 --vocab extended --label-delay 3 --cache data/cache_ext_bolly3x2 `
        --context 400 --random-eq 0.5 --bass-dropout 0.3 --early-weight $r[1] --out "models/$name" 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath "data/logs/$name.log"
    uv run --no-sync python experiments/hmm55.py "models/$name"
    uv run --no-sync python experiments/analysis_raw_flicker.py "models/${name}_h5p2" 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath "data/logs/${name}_eval.log"
    uv run --no-sync python experiments/analysis_bollywood.py --v3 "models/${name}_h5p2" 2>&1 | Where-Object { $_ -match 'aligned' } | Tee-Object -Append -FilePath "data/logs/${name}_eval.log"
}
Log 'e36 done'
