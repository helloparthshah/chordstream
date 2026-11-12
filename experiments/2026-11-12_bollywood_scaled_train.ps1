# E28a/b: train on the scaled Bollywood set (data/cache_bolly3, 152 cleaned training songs vs 47 in
# E21e). a = each song once (Bollywood share about E21e's), b = twice. v7/v8 recipe from the
# synthetic init. Test songs (30% by hash) never linked.
$ErrorActionPreference = 'Continue'
$train = uv run --no-sync python -c "import zlib; from pathlib import Path; from chordstream.data.bollywood import trusted_ids_v2; ok = trusted_ids_v2('align_cache_bolly3.jsonl'); print('\n'.join(f.name for f in sorted(Path('data/cache_bolly3').glob('*.npz')) if f.stem.split('__')[-1] in ok and zlib.crc32(f.stem.encode()) % 1000 >= 300))"
"bollywood v3 training songs: $($train.Count)" | Tee-Object -Append data/logs/batch.log
foreach ($cfg in @(@{dir='data/cache_ext_bolly3x1'; reps=1}, @{dir='data/cache_ext_bolly3x2'; reps=2})) {
    New-Item -ItemType Directory -Force $cfg.dir | Out-Null
    Get-ChildItem data/cache_ext/*.npz | ForEach-Object { $d = "$($cfg.dir)/$($_.Name)"; if (-not (Test-Path $d)) { New-Item -ItemType HardLink -Path $d -Target $_.FullName | Out-Null } }
    foreach ($n in $train) { foreach ($k in 1..$cfg.reps) { $d = "$($cfg.dir)/bollytrain__$($n.Replace('bolly__','').Replace('.npz',''))__$k.npz"; if (-not (Test-Path $d)) { New-Item -ItemType HardLink -Path $d -Target (Resolve-Path "data/cache_bolly3/$n") | Out-Null } } }
}
function Run($name, [string[]]$cmd) {
    "=== $name $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
    & uv run --no-sync chordstream @cmd 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath "data/logs/$name.log"
}
$rec = @('train', '--init', 'models/latest', '--lr', '3e-4', '--epochs', '80', '--crops', '2000', '--workers', '0',
         '--transition-weight', '3', '--eval-every', '10', '--vocab', 'extended', '--label-delay', '3')
Run 'e28a_bolly3x1' ($rec + @('--cache', 'data/cache_ext_bolly3x1', '--out', 'models/e28a_bolly3x1'))
Run 'e28b_bolly3x2' ($rec + @('--cache', 'data/cache_ext_bolly3x2', '--out', 'models/e28b_bolly3x2'))
"=== e28 done $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
