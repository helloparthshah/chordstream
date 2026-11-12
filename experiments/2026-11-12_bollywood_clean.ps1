# E21e/f: Bollywood with cleaned labels (data/cache_bolly2: bass-aware alignment at the capo
# transposition, least-confident 30% of frames ignored), training songs x3. f also adds E27b's 50
# rare-chord renders. v7 recipe from the synthetic init. Test songs (30% by hash) never linked.
$ErrorActionPreference = 'Continue'
$train = uv run --no-sync python -c "import zlib; from pathlib import Path; from chordstream.data.bollywood import trusted_ids_v2; ok = trusted_ids_v2(); print('\n'.join(f.name for f in sorted(Path('data/cache_bolly2').glob('*.npz')) if f.stem.split('__')[-1] in ok and zlib.crc32(f.stem.encode()) % 1000 >= 300))"
"bollywood v2 training songs: $($train.Count)" | Tee-Object -Append data/logs/batch.log
foreach ($cfg in @(@{dir='data/cache_ext_bolly2'; rare=0}, @{dir='data/cache_ext_bolly2_rare'; rare=50})) {
    New-Item -ItemType Directory -Force $cfg.dir | Out-Null
    Get-ChildItem data/cache_ext/*.npz | ForEach-Object { $d = "$($cfg.dir)/$($_.Name)"; if (-not (Test-Path $d)) { New-Item -ItemType HardLink -Path $d -Target $_.FullName | Out-Null } }
    foreach ($n in $train) { foreach ($k in 1..3) { $d = "$($cfg.dir)/bollytrain__$($n.Replace('bolly__','').Replace('.npz',''))__$k.npz"; if (-not (Test-Path $d)) { New-Item -ItemType HardLink -Path $d -Target (Resolve-Path "data/cache_bolly2/$n") | Out-Null } } }
    if ($cfg.rare) { Get-ChildItem data/cache_rare/*.npz | Sort-Object Name | Select-Object -First $cfg.rare | ForEach-Object { $d = "$($cfg.dir)/$($_.Name)"; if (-not (Test-Path $d)) { New-Item -ItemType HardLink -Path $d -Target $_.FullName | Out-Null } } }
}
function Run($name, [string[]]$cmd) {
    "=== $name $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
    & uv run --no-sync chordstream @cmd 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath "data/logs/$name.log"
}
$rec = @('train', '--init', 'models/latest', '--lr', '3e-4', '--epochs', '80', '--crops', '2000', '--workers', '0',
         '--transition-weight', '3', '--eval-every', '10', '--vocab', 'extended', '--label-delay', '3')
Run 'e21e_bolly_clean' ($rec + @('--cache', 'data/cache_ext_bolly2', '--out', 'models/e21e_bolly_clean'))
Run 'e21f_bolly_clean_rare' ($rec + @('--cache', 'data/cache_ext_bolly2_rare', '--out', 'models/e21f_bolly_clean_rare'))
"=== e21ef done $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
