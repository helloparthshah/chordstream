# E30a/b: E22's stability loss (squared change of the predicted distribution between frames where the
# true chord is constant) on v9's recipe and data. E22 failed at zero delay; with the 120 ms label
# delay the model can see a change coming, so penalizing flips may now cut flicker in the model.
$ErrorActionPreference = 'Continue'
function Run($name, [string[]]$cmd) {
    "=== $name $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
    & uv run --no-sync chordstream @cmd 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath "data/logs/$name.log"
}
$rec = @('train', '--init', 'models/latest', '--lr', '3e-4', '--epochs', '80', '--crops', '2000', '--workers', '0',
         '--transition-weight', '3', '--eval-every', '10', '--vocab', 'extended', '--label-delay', '3', '--cache', 'data/cache_ext_bolly3x2')
Run 'e30a_stab2' ($rec + @('--stability-weight', '2', '--out', 'models/e30a_stab2'))
Run 'e30b_stab5' ($rec + @('--stability-weight', '5', '--out', 'models/e30b_stab5'))
"=== e30 done $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
