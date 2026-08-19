# Commits current changes and pushes. Uses the credentials git already stored.
$ErrorActionPreference = "Continue"
$p = "$env:USERPROFILE\influencer-platform"
$g = "$env:USERPROFILE\anaconda3\envs\influencer\Library\bin\git.exe"
Set-Location $p

$msg = if ($args.Count -gt 0) { $args -join " " } else { "Update" }

Get-ChildItem -Path $p -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

& $g add -A 2>&1 | Out-Null
& $g -c core.pager=cat commit -m $msg 2>&1 | Select-Object -Last 2
& $g push origin main 2>&1 | Select-Object -Last 5

if ($LASTEXITCODE -eq 0) {
    Write-Output "=== PUSHED ==="
    & $g -c core.pager=cat log --oneline -1
} else {
    Write-Output "=== PUSH FAILED (exit $LASTEXITCODE) ==="
}
