# E31a/b: v9's recipe and data with bass dropout: in a share of training crops (0.3 / 0.6) everything
# below a random cutoff (92-370 Hz) is removed or attenuated, so chords must be read from the upper
# harmony. F07: v9 falls from 80.0% to 46.8% on Bollywood when the audio below 300 Hz is removed, and
# gets inverted chords right only 46.6% of the time.
$ErrorActionPreference = 'Continue'
function Run($name, [string[]]$cmd) {
    "=== $name $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
    & uv run --no-sync chordstream @cmd 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath "data/logs/$name.log"
}
$rec = @('train', '--init', 'models/latest', '--lr', '3e-4', '--epochs', '80', '--crops', '2000', '--workers', '0',
         '--transition-weight', '3', '--eval-every', '10', '--vocab', 'extended', '--label-delay', '3', '--cache', 'data/cache_ext_bolly3x2')
Run 'e31a_bassdrop3' ($rec + @('--bass-dropout', '0.3', '--out', 'models/e31a_bassdrop3'))
Run 'e31b_bassdrop6' ($rec + @('--bass-dropout', '0.6', '--out', 'models/e31b_bassdrop6'))
"=== e31 done $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
