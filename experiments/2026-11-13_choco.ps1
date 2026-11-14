# E40: 547 more human-labelled songs from ChoCo (Isophonics 214, USPop2002 183, Robbie Williams 61,
# JAAH 89; Billboard duplicates and Bohemian Rhapsody removed), audio from YouTube via fetch-choco
# (auto-aligned and verified like Billboard). Train-only ("choco__"), so the test sets are unchanged.
# Rejected downloads are re-checked with tuning-aware alignment first (align_tuned: the verification
# chroma failed recordings 15-45 cents flat), which also recovers flat Billboard songs ("bbrec__").
# E40 = v12's recipe (32 s context) on cache_ext_bolly3x2 + the new songs.
$ErrorActionPreference = 'Continue'
function Log($m) { "=== $m $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log }
Log 'e40 features'
uv run --no-sync chordstream features data/choco_audio --out data/cache_choco --extended choco 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath data/logs/e40_features.log
uv run --no-sync chordstream features data/billboard_rec --out data/cache_choco --extended bbrec 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -Append -FilePath data/logs/e40_features.log
$dir = 'data/cache_ext_bolly3x2_choco'
New-Item -ItemType Directory -Force $dir | Out-Null
foreach ($src in @('data/cache_ext_bolly3x2', 'data/cache_choco')) {
    Get-ChildItem "$src/*.npz" | ForEach-Object { $d = "$dir/$($_.Name)"; if (-not (Test-Path $d)) { New-Item -ItemType HardLink -Path $d -Target $_.FullName | Out-Null } }
}
$name = 'e40_choco'
Log $name
& uv run --no-sync chordstream train --init models/latest --lr 3e-4 --epochs 80 --crops 2000 --workers 0 --batch-size 8 `
    --transition-weight 3 --eval-every 10 --vocab extended --label-delay 3 --cache $dir `
    --context 800 --random-eq 0.5 --bass-dropout 0.3 --out "models/$name" 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath "data/logs/$name.log"
Log "$name eval"
uv run --no-sync python experiments/hmm55.py "models/$name"
uv run --no-sync python experiments/analysis_bollywood.py --v3 "models/${name}_h5p2" 2>&1 | Where-Object { $_ -match 'aligned' } | Tee-Object -FilePath "data/logs/${name}_eval.log"
uv run --no-sync python experiments/analysis_bass.py "models/${name}_h5p2" 2>&1 | Where-Object { $_ -match 'Billboard|Bollywood' } | Tee-Object -Append -FilePath "data/logs/${name}_eval.log"
uv run --no-sync python experiments/analysis_raw_flicker.py "models/${name}_h5p2" 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -Append -FilePath "data/logs/${name}_eval.log"
uv run --no-sync python experiments/analysis_detune.py "models/${name}_h5p2" 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath "data/logs/f11_$name.log"
Log 'e40 done'
