# E27: rare chord types from synthetic renders, mixed into v7's recipe (human labels + exact synthetic labels).
# data/cache_ext_rare = data/cache_ext + 150 files of data/cache_rare (training only: prefix synthx), hard-linked.
$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force data/logs, data/cache_ext_rare | Out-Null
Get-ChildItem data/cache_ext/*.npz | ForEach-Object { $d = "data/cache_ext_rare/$($_.Name)"; if (-not (Test-Path $d)) { New-Item -ItemType HardLink -Path $d -Target $_.FullName | Out-Null } }
Get-ChildItem data/cache_rare/*.npz | Sort-Object Name | Select-Object -First 150 | ForEach-Object { $d = "data/cache_ext_rare/$($_.Name)"; if (-not (Test-Path $d)) { New-Item -ItemType HardLink -Path $d -Target $_.FullName | Out-Null } }
function Run($name, [string[]]$cmd) {
    "=== $name $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
    & uv run --no-sync chordstream @cmd 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath "data/logs/$name.log"
}
Run 'e27_rare' @('train', '--init', 'models/latest', '--lr', '3e-4', '--epochs', '80', '--crops', '2000', '--workers', '0',
                 '--transition-weight', '3', '--eval-every', '10', '--cache', 'data/cache_ext_rare', '--vocab', 'extended',
                 '--label-delay', '3', '--out', 'models/e27_rare')
"=== e27 done $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
