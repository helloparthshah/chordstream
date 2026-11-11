# E19 (causal HPSS features) and E20 (oracle key-signature input).
# Same recipe as E15/v4_clean (lr 3e-4, 80 epochs, transition weight 3, cleaned labels), so each
# result compares directly with models/v4_clean on the 36 clean held-out songs.
$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force data/logs | Out-Null
function Run($name, [string[]]$cmd) {
    "=== $name $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
    & uv run --no-sync chordstream @cmd 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath "data/logs/$name.log"
}
"=== variants $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
& uv run --no-sync python -c "from chordstream.data.variants import build_variant as b; print('hpss', b('data/cache_bb_fix', 'data/cache_hpss', 'hpss')); print('key', b('data/cache_bb_fix', 'data/cache_key', 'key'))" 2>&1 | Tee-Object -FilePath data/logs/variants.log
$common = @('--init', 'models/latest', '--lr', '3e-4', '--epochs', '80', '--crops', '2000', '--workers', '0',
            '--transition-weight', '3', '--eval-every', '10')
Run 'e19_hpss' (@('train', '--cache', 'data/cache_hpss', '--out', 'models/e19_hpss') + $common)
Run 'e20_key'  (@('train', '--cache', 'data/cache_key', '--out', 'models/e20_key', '--key-input') + $common)
"=== e19/e20 done $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
