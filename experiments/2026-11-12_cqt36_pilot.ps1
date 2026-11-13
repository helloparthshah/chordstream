# E37 pilot (runs alongside E35): does a 36 bin/octave CQT (3 per semitone) fix the tuning cliff?
# Same small setup for all three, from scratch (models/latest is 24-bin), Billboard only (data/cache_ext),
# 5 s context, 40 epochs: A 24 bins (control), B 36 bins, C 36 bins + random +-1 sub-semitone bin
# shift (+-33 cents, labels kept). Scored by analysis_detune.py on the 36 Billboard test songs.
$ErrorActionPreference = 'Continue'
function Log($m) { "=== $m $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log }
Log 'e37 cqt36 cache'
uv run --no-sync python experiments/build_cqt36_cache.py data/cache_ext36 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath data/logs/e37_cache.log
$common = @('train', '--lr', '1e-3', '--epochs', '40', '--crops', '2000', '--workers', '0', '--batch-size', '32',
            '--transition-weight', '3', '--eval-every', '10', '--vocab', 'extended', '--label-delay', '3')
$runs = @(
    @('e37a_bins24', @('--cache', 'data/cache_ext')),
    @('e37b_bins36', @('--cache', 'data/cache_ext36', '--bins-per-octave', '36')),
    @('e37c_bins36_micro', @('--cache', 'data/cache_ext36', '--bins-per-octave', '36', '--micro-shift'))
)
foreach ($r in $runs) {
    $name = $r[0]
    Log $name
    & uv run --no-sync chordstream @($common + $r[1] + @('--out', "models/$name")) 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath "data/logs/$name.log"
    uv run --no-sync python experiments/hmm55.py "models/$name"
}
Log 'e37 detune'
uv run --no-sync python experiments/analysis_detune.py --billboard-only models/e37a_bins24_h5p2 models/e37b_bins36_h5p2 models/e37c_bins36_micro_h5p2 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath data/logs/e37_detune.log
Log 'e37 done'
