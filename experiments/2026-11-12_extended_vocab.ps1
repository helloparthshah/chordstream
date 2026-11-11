# Extended vocabulary: teacher -> pseudo-labels -> causal student (same recipe as v5).
$ErrorActionPreference = 'Continue'
function Run($name, [string[]]$cmd) {
    "=== $name $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
    & uv run --no-sync chordstream @cmd 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath "data/logs/$name.log"
}
Run 'ext_teacher' @('train', '--cache', 'data/cache_ext', '--init', 'models/latest', '--out', 'models/ext_teacher',
                    '--vocab', 'extended', '--bidirectional', '--lr', '3e-4', '--epochs', '80', '--crops', '2000',
                    '--workers', '0', '--eval-every', '10')
Run 'ext_pseudo' @('pseudo-label', '--teacher', 'models/ext_teacher', '--labelled', 'data/cache_ext',
                   '--out', 'data/cache_distill_ext', '--unlabelled-audio', 'data/billboard/_rejected',
                   '--unlabelled-cache', 'data/cache_bb')
Run 'ext_student' @('train', '--cache', 'data/cache_distill_ext', '--init', 'models/latest', '--out', 'models/ext_student',
                    '--vocab', 'extended', '--lr', '3e-4', '--epochs', '80', '--crops', '2000', '--workers', '0',
                    '--transition-weight', '3', '--distill-weight', '1.0', '--eval-every', '10')
"=== ext done $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
