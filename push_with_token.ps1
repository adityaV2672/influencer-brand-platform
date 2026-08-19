# Pushes to GitHub using a Personal Access Token. No dialog boxes involved.
#
# Run it, paste your token when asked, press Enter.
# The token is read as a SecureString so it is not echoed to the screen and is
# not written into your PowerShell history.
$ErrorActionPreference = "Continue"

$Username = "adityaV2672"
$RepoName = "influencer-brand-platform"
$p = "$env:USERPROFILE\influencer-platform"
$g = "$env:USERPROFILE\anaconda3\envs\influencer\Library\bin\git.exe"
Set-Location $p

Write-Host ""
Write-Host "=== GitHub push ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "You need a Personal Access Token. If you don't have one yet:" -ForegroundColor Yellow
Write-Host "  1. Open  https://github.com/settings/tokens"
Write-Host "  2. 'Generate new token'  ->  'Generate new token (classic)'"
Write-Host "  3. Note: anything, e.g. 'streamlit deploy'.  Expiration: 30 days is fine."
Write-Host "  4. Tick the top-level  repo  checkbox"
Write-Host "  5. Scroll down, 'Generate token', then COPY it (starts with ghp_)"
Write-Host ""
Write-Host "Paste the token below and press Enter. It will not appear on screen." -ForegroundColor Yellow
Write-Host ""

$secure = Read-Host -Prompt "Token" -AsSecureString
$token  = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
    [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))

if ([string]::IsNullOrWhiteSpace($token)) {
    Write-Host "No token entered. Nothing was changed." -ForegroundColor Red
    Read-Host "Press Enter to close"
    exit 1
}

# Push with the token supplied inline, then immediately reset the stored remote
# to the clean URL so the token is never left sitting in .git/config.
$authUrl  = "https://${Username}:${token}@github.com/$Username/$RepoName.git"
$cleanUrl = "https://github.com/$Username/$RepoName.git"

& $g remote remove origin 2>$null
& $g remote add origin $cleanUrl

Write-Host ""
Write-Host "pushing..." -ForegroundColor Cyan
& $g push $authUrl main:main 2>&1 | ForEach-Object { $_ -replace [regex]::Escape($token), "***" }
$code = $LASTEXITCODE

$token = $null
[System.GC]::Collect()

if ($code -eq 0) {
    & $g branch --set-upstream-to=origin/main main 2>$null
    Write-Host ""
    Write-Host "=== PUSHED SUCCESSFULLY ===" -ForegroundColor Green
    Write-Host "https://github.com/$Username/$RepoName"
    Write-Host ""
    Write-Host "Next: https://share.streamlit.io  ->  Create app  ->  Deploy from GitHub"
    Write-Host "  Repository:     $Username/$RepoName"
    Write-Host "  Branch:         main"
    Write-Host "  Main file path: app/Home.py"
} else {
    Write-Host ""
    Write-Host "=== PUSH FAILED (exit $code) ===" -ForegroundColor Red
    Write-Host "If it said 'Authentication failed', the token was wrong or lacked the 'repo' scope."
    Write-Host "If it said 'Updates were rejected', run this then try again:"
    Write-Host "  & '$g' pull --rebase origin main"
}

Write-Host ""
Read-Host "Press Enter to close this window"
