# E21: Billboard (data/cache_ext) + trusted Bollywood training songs (sheet-aligned labels), human labels only.
# Bollywood test songs (30% by hash, see experiments/analysis_bollywood.py) are never linked in.
# Each Bollywood training song is linked 3x (prefix bollytrain__: training only) so ~41 songs carry ~25% weight.
$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force data/logs, data/cache_ext_bolly | Out-Null
Get-ChildItem data/cache_ext/*.npz | ForEach-Object { $d = "data/cache_ext_bolly/$($_.Name)"; if (-not (Test-Path $d)) { New-Item -ItemType HardLink -Path $d -Target $_.FullName | Out-Null } }
$train = uv run --no-sync python -c "import zlib; from pathlib import Path; from chordstream.data.bollywood import trusted_ids; ok = trusted_ids(); print('\n'.join(f.name for f in sorted(Path('data/cache_bolly').glob('*.npz')) if f.stem.split('__')[-1] in ok and zlib.crc32(f.stem.encode()) % 1000 >= 300))"
foreach ($n in $train) { foreach ($k in 1..3) { $d = "data/cache_ext_bolly/bollytrain__$($n.Replace('bolly__','').Replace('.npz',''))__$k.npz"; if (-not (Test-Path $d)) { New-Item -ItemType HardLink -Path $d -Target (Resolve-Path "data/cache_bolly/$n") | Out-Null } } }
"bollywood training songs: $($train.Count)" | Tee-Object -Append data/logs/batch.log
function Run($name, [string[]]$cmd) {
    "=== $name $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
    & uv run --no-sync chordstream @cmd 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath "data/logs/$name.log"
}
$rec = @('--crops', '2000', '--workers', '0', '--transition-weight', '3', '--eval-every', '10', '--cache', 'data/cache_ext_bolly',
         '--vocab', 'extended', '--label-delay', '3')
Run 'e21a_bolly_scratch' (@('train', '--init', 'models/latest', '--lr', '3e-4', '--epochs', '80', '--out', 'models/e21a_bolly') + $rec)
Run 'e21b_bolly_ft' (@('train', '--init', 'models/v7', '--lr', '1e-4', '--epochs', '20', '--out', 'models/e21b_bolly_ft') + $rec)
"=== e21 done $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
