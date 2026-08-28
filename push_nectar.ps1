<#
    The audited Nectar build is already installed and committed in this folder.
    This script does the one step that needs your GitHub credentials: the push.

    Run it: right-click, "Run with PowerShell".
#>

$ErrorActionPreference = "Continue"
Set-Location "C:\Users\adity\influencer-platform"

Write-Host ""
Write-Host "  Pushing the audited build to GitHub" -ForegroundColor Cyan
Write-Host "  -----------------------------------" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Commits waiting to go up:" -ForegroundColor Yellow
git log origin/main..HEAD --oneline
Write-Host ""

git push origin HEAD

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  Push failed — this is almost always Git asking for credentials." -ForegroundColor Red
    Write-Host "  Run push_with_token.ps1 instead; it is already in this folder."
    Read-Host "`n  Press Enter to close"
    exit 1
}

Write-Host ""
Write-Host "  Pushed. Streamlit Cloud redeploys in about a minute." -ForegroundColor Green
Write-Host "  https://influencer-brand-platform-cbeykxfdvrypcursfund6t.streamlit.app/" -ForegroundColor Green
Write-Host ""
Write-Host "  To check it worked, open the app and go to Model & Methods -> Model." -ForegroundColor Green
Write-Host "  The top row should read: R2 0.638, structural baseline 0.473." -ForegroundColor Green
Write-Host ""
Read-Host "  Press Enter to close"
