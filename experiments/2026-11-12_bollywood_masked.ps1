# E21c/d: as E21a, but Bollywood labels ignored within +-320 ms of each aligned chord change
# (data/cache_bolly_masked); c = 3x weight, d = 1x weight. Same 41 training songs, same test songs.
$ErrorActionPreference = 'Continue'
$train = uv run --no-sync python -c "import zlib; from pathlib import Path; from chordstream.data.bollywood import trusted_ids; ok = trusted_ids(); print('\n'.join(f.name for f in sorted(Path('data/cache_bolly').glob('*.npz')) if f.stem.split('__')[-1] in ok and zlib.crc32(f.stem.encode()) % 1000 >= 300))"
foreach ($cfg in @(@{dir='data/cache_ext_bollym3'; k=3}, @{dir='data/cache_ext_bollym1'; k=1})) {
    New-Item -ItemType Directory -Force $cfg.dir | Out-Null
    Get-ChildItem data/cache_ext/*.npz | ForEach-Object { $d = "$($cfg.dir)/$($_.Name)"; if (-not (Test-Path $d)) { New-Item -ItemType HardLink -Path $d -Target $_.FullName | Out-Null } }
    foreach ($n in $train) { foreach ($k in 1..$cfg.k) { $d = "$($cfg.dir)/bollytrain__$($n.Replace('bolly__','').Replace('.npz',''))__$k.npz"; if (-not (Test-Path $d)) { New-Item -ItemType HardLink -Path $d -Target (Resolve-Path "data/cache_bolly_masked/$n") | Out-Null } } }
}
function Run($name, [string[]]$cmd) {
    "=== $name $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
    & uv run --no-sync chordstream @cmd 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath "data/logs/$name.log"
}
$rec = @('train', '--init', 'models/latest', '--lr', '3e-4', '--epochs', '80', '--crops', '2000', '--workers', '0',
         '--transition-weight', '3', '--eval-every', '10', '--vocab', 'extended', '--label-delay', '3')
Run 'e21c_bolly_mask3' ($rec + @('--cache', 'data/cache_ext_bollym3', '--out', 'models/e21c_bolly_mask3'))
Run 'e21d_bolly_mask1' ($rec + @('--cache', 'data/cache_ext_bollym1', '--out', 'models/e21d_bolly_mask1'))
"=== e21cd done $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
