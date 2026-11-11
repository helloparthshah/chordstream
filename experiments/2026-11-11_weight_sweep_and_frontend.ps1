# Experiment batch: transition-weight sweep + faster frontend (gamma 10). Logs in data/logs/.
$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force data/logs | Out-Null
function Run($name, [string[]]$cmd) {
    "=== $name $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
    & uv run --no-sync chordstream @cmd 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath "data/logs/$name.log"
}
# 1. Snapshot the same song set for both frontends.
Run 'cache_bb'      @('features', 'data/billboard', '--out', 'data/cache_bb')
Run 'cache_bb_g10'  @('features', 'data/billboard', '--out', 'data/cache_bb_g10', '--gamma', '10')
Run 'cache_syn_g10' @('features', 'data/synth', 'data/synth_val', '--out', 'data/cache_g10', '--gamma', '10')
# 2. Transition-weight sweep (gamma 3), identical data and seed.
foreach ($w in 2, 3, 5) {
    Run "exp_w$w" @('train', '--cache', 'data/cache_bb', '--init', 'models/latest', '--out', "models/exp_w$w",
                    '--lr', '1e-4', '--epochs', '40', '--crops', '2000', '--workers', '0', '--transition-weight', "$w")
}
# 3. Faster frontend: re-pretrain on synthetic audio, then fine-tune with weight 3.
Run 'synth_g10' @('train', '--cache', 'data/cache_g10', '--out', 'models/synth_g10', '--gamma', '10', '--epochs', '12',
                  '--crops', '2000', '--holdout-prefix', 'synth_val__', '--workers', '0')
Run 'exp_g10' @('train', '--cache', 'data/cache_bb_g10', '--init', 'models/synth_g10', '--out', 'models/exp_g10',
                '--lr', '1e-4', '--epochs', '40', '--crops', '2000', '--workers', '0', '--transition-weight', '3')
"=== done $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
