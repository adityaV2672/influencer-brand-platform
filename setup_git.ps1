# Prepares the local git repository and makes the first commit.
# You still have to create the empty repo on GitHub and push - that needs your login.
$ErrorActionPreference = "Continue"

$p = "$env:USERPROFILE\influencer-platform"
$g = "$env:USERPROFILE\anaconda3\envs\influencer\Library\bin\git.exe"
Set-Location $p

# Compiled caches should never be committed.
Get-ChildItem -Path $p -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

if (-not (Test-Path "$p\.git")) {
    & $g @("init", "-b", "main") | Out-Null
    Write-Output "created repository"
} else {
    Write-Output "repository already exists"
}

& $g config user.name  "Aditya Verma"
& $g config user.email "adityavrm26@gmail.com"
& $g config core.autocrlf true

& $g add -A 2>&1 | Out-Null
& $g -c core.pager=cat commit -m "Influencer-brand collaboration platform: ML scoring, NLP benchmarking, freemium dashboard" 2>&1 |
    Select-Object -Last 3

Write-Output "--- tracked files ---"
$tracked = & $g ls-files
Write-Output $tracked.Count

Write-Output "--- app_data files tracked (must be > 0) ---"
$appdata = & $g ls-files "app_data"
Write-Output $appdata.Count

Write-Output "--- largest tracked files ---"
$tracked | ForEach-Object {
    $f = Join-Path $p $_
    if (Test-Path $f) { [PSCustomObject]@{ KB = [math]::Round((Get-Item $f).Length / 1KB); File = $_ } }
} | Sort-Object KB -Descending | Select-Object -First 8 | Format-Table -AutoSize

Write-Output "--- total repo size ---"
"{0:N1} MB" -f ((Get-ChildItem "$p\.git" -Recurse -File | Measure-Object Length -Sum).Sum / 1MB)
