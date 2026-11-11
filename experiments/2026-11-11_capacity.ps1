# Capacity experiments: is v3 (79% on its own training songs) under-trained or too small?
$ErrorActionPreference = 'Continue'
function Run($name, [string[]]$cmd) {
    "=== $name $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
    & uv run --no-sync chordstream @cmd 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath "data/logs/$name.log"
}
Run 'exp_lr3' @('train', '--cache', 'data/cache_bb', '--init', 'models/latest', '--out', 'models/exp_lr3', '--lr', '3e-4',
                '--epochs', '40', '--crops', '2000', '--workers', '0', '--transition-weight', '3', '--eval-every', '5')
Run 'synth_big' @('train', '--cache', 'data/cache', '--out', 'models/synth_big', '--d-model', '256', '--layers', '6',
                  '--epochs', '12', '--crops', '2000', '--holdout-prefix', 'synth_val__', '--workers', '0')
Run 'exp_big' @('train', '--cache', 'data/cache_bb', '--init', 'models/synth_big', '--out', 'models/exp_big', '--lr', '1e-4',
                '--epochs', '40', '--crops', '2000', '--workers', '0', '--transition-weight', '3', '--eval-every', '5')
"=== capacity done $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
