# Pushes the project to GitHub.
#
# Usage:  .\push_to_github.ps1 YOUR-GITHUB-USERNAME
#
# Before running this, create an EMPTY public repository at
# https://github.com/new  named  influencer-brand-platform
# Do NOT tick "Add a README", ".gitignore" or "license" - the project already
# has them and ticking those causes a conflict on the first push.
param(
    [Parameter(Mandatory = $true)]
    [string]$Username,
    [string]$RepoName = "influencer-brand-platform"
)

$ErrorActionPreference = "Continue"
$p = "$env:USERPROFILE\influencer-platform"
$g = "$env:USERPROFILE\anaconda3\envs\influencer\Library\bin\git.exe"
Set-Location $p

$url = "https://github.com/$Username/$RepoName.git"
Write-Output "pushing to $url"

# Replace any existing remote so re-running is safe.
& $g remote remove origin 2>$null
& $g remote add origin $url

Write-Output ""
Write-Output "A browser window may open asking you to sign in to GitHub."
Write-Output "If a terminal PASSWORD prompt appears instead, GitHub no longer accepts"
Write-Output "account passwords there - create a token at"
Write-Output "  https://github.com/settings/tokens  (Generate new token (classic), tick 'repo')"
Write-Output "and paste the token as the password."
Write-Output ""

& $g push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Output ""
    Write-Output "=== PUSHED SUCCESSFULLY ==="
    Write-Output "Repository: https://github.com/$Username/$RepoName"
    Write-Output ""
    Write-Output "Next: go to https://share.streamlit.io"
    Write-Output "  1. Sign in with GitHub"
    Write-Output "  2. Create app -> Deploy a public app from GitHub"
    Write-Output "  3. Repository:     $Username/$RepoName"
    Write-Output "     Branch:         main"
    Write-Output "     Main file path: app/Home.py     <- this one matters"
    Write-Output "  4. Deploy. First build takes 3-6 minutes."
} else {
    Write-Output ""
    Write-Output "=== PUSH FAILED ==="
    Write-Output "Most likely causes:"
    Write-Output "  - the repo name or username is wrong -> check https://github.com/$Username/$RepoName exists"
    Write-Output "  - you ticked 'Add a README' when creating it. Fix with:"
    Write-Output "      & '$g' pull --rebase origin main"
    Write-Output "      & '$g' push -u origin main"
}
