# ==========================================================================
# Installs Ollama (per-user, no admin) and pulls a small instruct model.
# Powers the LLM sarcasm-detection method.
#
# The first attempt failed with a mid-download connection reset, so this uses
# curl.exe (shipped with Windows 10+) with resume and retry rather than
# Invoke-WebRequest, which restarts from zero on any interruption.
# ==========================================================================
$ErrorActionPreference = "Continue"
$ProgressPreference    = "SilentlyContinue"

$log = "$env:USERPROFILE\influencer-platform\ollama_log.txt"
function Log($m) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m
    Write-Output $line
    Add-Content -Path $log -Value $line
}
Set-Content -Path $log -Value "=== ollama setup $(Get-Date) ==="

$exe = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"

if (-not (Test-Path $exe)) {
    $installer = "$env:TEMP\OllamaSetup.exe"
    Log "downloading OllamaSetup.exe with resume support"

    # -C -  resume from wherever a partial file left off
    # --retry 8 --retry-all-errors  survive transient resets
    & curl.exe -L --fail --retry 8 --retry-delay 5 --retry-all-errors `
        -C - --connect-timeout 30 --max-time 3600 `
        -o $installer "https://ollama.com/download/OllamaSetup.exe" 2>&1 | Out-Null

    if (-not (Test-Path $installer) -or (Get-Item $installer).Length -lt 50MB) {
        Log "FATAL: download failed or file is too small to be valid"
        Log "  Fix manually: download https://ollama.com/download/windows and run it,"
        Log "  then re-run this script to pull the model."
        exit 1
    }
    Log "downloaded $([math]::Round((Get-Item $installer).Length/1MB)) MB"

    Log "installing (silent, per-user)"
    Start-Process -FilePath $installer -ArgumentList "/VERYSILENT","/NORESTART","/SUPPRESSMSGBOXES" -Wait
    Start-Sleep -Seconds 12
}

if (-not (Test-Path $exe)) { Log "FATAL: ollama.exe not found after install"; exit 1 }
Log "ollama installed at $exe"

$listening = $false
try { $null = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 4 -UseBasicParsing; $listening = $true } catch {}
if (-not $listening) {
    Log "starting the ollama server"
    Start-Process -FilePath $exe -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 15
}

# qwen2.5:7b-instruct quantised is ~4.7 GB and fits comfortably in 15 GB RAM.
Log "pulling qwen2.5:7b-instruct (~4.7 GB)"
& $exe pull qwen2.5:7b-instruct 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Log "7b pull failed - falling back to qwen2.5:3b-instruct"
    & $exe pull qwen2.5:3b-instruct 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { Log "FATAL: no model could be pulled"; exit 1 }
}

Log "models available:"
(& $exe list 2>&1) | ForEach-Object { Log "    $_" }
Log "=== OLLAMA READY ==="
