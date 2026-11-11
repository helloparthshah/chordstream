# E25: extended vocabulary + label delay (E23's small lookahead on E24's model). Human labels only.
# Compared with models/e24_ext_nodistill (zero delay) and with fixed-lag decoding of it at equal delay.
$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force data/logs | Out-Null
function Run($name, [string[]]$cmd) {
    "=== $name $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
    & uv run --no-sync chordstream @cmd 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath "data/logs/$name.log"
}
$common = @('train', '--init', 'models/latest', '--lr', '3e-4', '--epochs', '80', '--crops', '2000', '--workers', '0',
            '--transition-weight', '3', '--eval-every', '10', '--cache', 'data/cache_ext', '--vocab', 'extended')
Run 'e25a_ext_delay3' ($common + @('--out', 'models/e25a_ext_delay3', '--label-delay', '3'))
Run 'e25b_ext_delay6' ($common + @('--out', 'models/e25b_ext_delay6', '--label-delay', '6'))
"=== e25 done $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
