# Lets other devices on your PRIVATE network reach the chordstream server (TCP 8000).
# Run as Administrator. Shows what it changes and asks before applying.
$ErrorActionPreference = 'Stop'
$blocks = Get-NetFirewallRule -Direction Inbound | Where-Object { $_.DisplayName -eq 'python.exe' -and $_.Action -eq 'Block' }

Write-Host "Will remove these inbound BLOCK rules (they stop the server receiving connections):"
$blocks | ForEach-Object {
    "  {0}  {1}  {2}" -f $_.DisplayName, $_.Profile, ($_ | Get-NetFirewallApplicationFilter).Program
}
Write-Host "Will add: allow inbound TCP 8000 on Private networks only (rule 'chordstream (TCP 8000)')."
if ((Read-Host "Apply? (y/n)") -ne 'y') { Write-Host "Nothing changed."; exit }

$blocks | Remove-NetFirewallRule
if (-not (Get-NetFirewallRule -DisplayName 'chordstream (TCP 8000)' -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName 'chordstream (TCP 8000)' -Direction Inbound -Protocol TCP -LocalPort 8000 `
        -Action Allow -Profile Private | Out-Null
}
Write-Host "Done. From your phone open https://<this PC's LAN IP>:8000"
Read-Host "Press Enter to close"
