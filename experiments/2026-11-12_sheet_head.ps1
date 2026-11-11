# E26: two delay heads in one network (extended vocabulary, human labels only): label delay 3
# (live readout, 120 ms) + a sheet head with delay 8 / 12 (320 / 480 ms) for the per-beat chords.
$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force data/logs | Out-Null
function Run($name, [string[]]$cmd) {
    "=== $name $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
    & uv run --no-sync chordstream @cmd 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath "data/logs/$name.log"
}
$common = @('train', '--init', 'models/latest', '--lr', '3e-4', '--epochs', '80', '--crops', '2000', '--workers', '0',
            '--transition-weight', '3', '--eval-every', '10', '--cache', 'data/cache_ext', '--vocab', 'extended',
            '--label-delay', '3')
Run 'e26a_sheet8'  ($common + @('--out', 'models/e26a_sheet8', '--sheet-delay', '8'))
Run 'e26b_sheet12' ($common + @('--out', 'models/e26b_sheet12', '--sheet-delay', '12'))
"=== e26 done $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
