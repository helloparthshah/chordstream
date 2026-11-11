# Teacher -> pseudo-labels -> causal student (E17).
$ErrorActionPreference = 'Continue'
function Run($name, [string[]]$cmd) {
    "=== $name $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
    & uv run --no-sync chordstream @cmd 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -FilePath "data/logs/$name.log"
}
# 1. Offline teacher: sees +-5 s, trained on the cleaned labels with the improved recipe.
Run 'teacher' @('train', '--cache', 'data/cache_bb_fix', '--init', 'models/latest', '--out', 'models/teacher', '--bidirectional',
                '--lr', '3e-4', '--epochs', '80', '--crops', '2000', '--workers', '0', '--eval-every', '10')
# 2. Soft targets for the clean songs, the drifting songs (labels dropped) and the rejected downloads.
Run 'pseudo' @('pseudo-label', '--teacher', 'models/teacher', '--labelled', 'data/cache_bb_fix', '--out', 'data/cache_distill',
               '--unlabelled-audio', 'data/billboard/_rejected', '--unlabelled-cache', 'data/cache_bb')
# 3. Causal student: human labels + teacher posteriors, same recipe as v4.
Run 'student' @('train', '--cache', 'data/cache_distill', '--init', 'models/latest', '--out', 'models/student',
                '--lr', '3e-4', '--epochs', '80', '--crops', '2000', '--workers', '0', '--transition-weight', '3',
                '--distill-weight', '1.0', '--eval-every', '10')
"=== distill done $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
