# E32a/b (user idea): random EQ instead of only removing the low end. In half of the training crops the
# CQT magnitudes pass through a random EQ curve: low shelf (mostly cuts up to 40 dB, corner 70-400 Hz),
# high shelf (-24..+6 dB, 1-3.5 kHz), tilt +-6 dB, 0-3 peaks/notches (-15..+10 dB). Applied exactly on
# the log1p features. b also keeps E31's bass dropout (0.3).
$ErrorActionPreference = 'Continue'
function Run($name, [string[]]$cmd) {
    "=== $name $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
    & uv run --no-sync chordstream @cmd 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath "data/logs/$name.log"
}
$rec = @('train', '--init', 'models/latest', '--lr', '3e-4', '--epochs', '80', '--crops', '2000', '--workers', '0',
         '--transition-weight', '3', '--eval-every', '10', '--vocab', 'extended', '--label-delay', '3', '--cache', 'data/cache_ext_bolly3x2')
Run 'e32a_eq5' ($rec + @('--random-eq', '0.5', '--out', 'models/e32a_eq5'))
Run 'e32b_eq5_bass3' ($rec + @('--random-eq', '0.5', '--bass-dropout', '0.3', '--out', 'models/e32b_eq5_bass3'))
"=== e32 done $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
