# E22 (zero-lookahead anti-flutter training), E23 (small lookahead via label delay) and E24
# (extended vocabulary without distillation). Human labels only, no teacher soft labels: the
# project avoids distillation so results reflect the method itself.
# E22/E23 use E15's recipe and data (data/cache_bb_fix, lr 3e-4, 80 epochs, transition weight 3)
# and compare with models/v4_clean. E24 uses data/cache_ext and compares with models/v6.
$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force data/logs | Out-Null
function Run($name, [string[]]$cmd) {
    "=== $name $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
    & uv run --no-sync chordstream @cmd 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath "data/logs/$name.log"
}
$common = @('train', '--init', 'models/latest', '--lr', '3e-4', '--epochs', '80', '--crops', '2000', '--workers', '0',
            '--transition-weight', '3', '--eval-every', '10')
Run 'e22a_stability' ($common + @('--cache', 'data/cache_bb_fix', '--out', 'models/e22a_stability', '--stability-weight', '2.0'))
Run 'e23a_delay3'    ($common + @('--cache', 'data/cache_bb_fix', '--out', 'models/e23a_delay3', '--label-delay', '3'))
Run 'e23b_delay6'    ($common + @('--cache', 'data/cache_bb_fix', '--out', 'models/e23b_delay6', '--label-delay', '6'))
Run 'e24_ext_nodistill' ($common + @('--cache', 'data/cache_ext', '--out', 'models/e24_ext_nodistill', '--vocab', 'extended'))
"=== e22/e23/e24 done $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
