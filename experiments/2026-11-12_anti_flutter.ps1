# E22 (zero-lookahead anti-flutter training) and E23 (small lookahead via label delay).
# Same recipe and data as E17's student / v5 (teacher soft labels in data/cache_distill,
# transition weight 3, lr 3e-4, 80 epochs), so each compares directly with models/student.
$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force data/logs | Out-Null
function Run($name, [string[]]$cmd) {
    "=== $name $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
    & uv run --no-sync chordstream @cmd 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath "data/logs/$name.log"
}
$common = @('train', '--cache', 'data/cache_distill', '--init', 'models/latest', '--lr', '3e-4', '--epochs', '80',
            '--crops', '2000', '--workers', '0', '--transition-weight', '3', '--eval-every', '10')
Run 'e22a_stability' ($common + @('--out', 'models/e22a_stability', '--distill-weight', '1.0', '--stability-weight', '2.0'))
Run 'e22b_distill3'  ($common + @('--out', 'models/e22b_distill3', '--distill-weight', '3.0'))
Run 'e23a_delay3'    ($common + @('--out', 'models/e23a_delay3', '--distill-weight', '1.0', '--label-delay', '3'))
Run 'e23b_delay6'    ($common + @('--out', 'models/e23b_delay6', '--distill-weight', '1.0', '--label-delay', '6'))
"=== e22/e23 done $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
