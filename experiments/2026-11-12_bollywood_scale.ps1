# E21 scale-up: more years and playlists, then the full pipeline (resumable stages).
$ErrorActionPreference = 'Continue'
function Stage($name, [string[]]$cmd) {
    "=== bolly $name $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
    & uv run --no-sync chordstream bollywood @cmd 2>&1 | Where-Object { $_ -notmatch 'Warning' } | Tee-Object -Append -FilePath "data/logs/bolly_scale.log"
}
Stage songs @('songs', '--year-from', '2000', '--year-to', '2024', '--per-year', '40', '--playlists-per-year', '3')
Stage sheets @('sheets')
Stage validate @('validate')
Stage audio @('audio')
Stage cache @('cache')
"=== bolly clean $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log
uv run --no-sync python -c "from chordstream.data.bollywood import stage_cache_clean; print(stage_cache_clean('data/cache_bolly3'))" 2>&1 | Tee-Object -Append -FilePath data/logs/bolly_scale.log
"=== bolly scale done $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append data/logs/batch.log

