# v4: longer / higher-LR training, on cleaned labels vs original labels (control).
$ErrorActionPreference = 'Continue'
function Run($name, [string[]]$cmd) {
    "=== $name $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
    & uv run --no-sync chordstream @cmd 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath "data/logs/$name.log"
}
$common = @('--init', 'models/latest', '--lr', '3e-4', '--epochs', '80', '--crops', '2000', '--workers', '0',
            '--transition-weight', '3', '--eval-every', '10')
Run 'v4_clean' (@('train', '--cache', 'data/cache_bb_fix', '--out', 'models/v4_clean') + $common)
Run 'v4_orig'  (@('train', '--cache', 'data/cache_bb', '--out', 'models/v4_orig') + $common)
"=== v4 done $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
