# ==========================================================
# One-time environment setup for the Influencer Platform.
# Creates an isolated conda environment so nothing touches
# your existing Anaconda base install.
# ==========================================================
$ErrorActionPreference = "Continue"
$ProgressPreference    = "SilentlyContinue"

$conda   = "$env:USERPROFILE\anaconda3\Scripts\conda.exe"
$envName = "influencer"
$envPath = "$env:USERPROFILE\anaconda3\envs\$envName"
$py      = "$envPath\python.exe"
$log     = "$env:USERPROFILE\influencer-platform\setup_log.txt"

function Log($m) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m
    Write-Output $line
    Add-Content -Path $log -Value $line
}

Set-Content -Path $log -Value "=== setup started $(Get-Date) ==="

Log "STEP 1/4  creating conda environment '$envName' (python 3.11)"
& $conda create -y -n $envName python=3.11 2>&1 | Out-Null
if (-not (Test-Path $py)) { Log "FATAL: env creation failed"; exit 1 }
Log "STEP 1/4  done"

Log "STEP 2/4  installing git into the environment"
& $conda install -y -n $envName -c conda-forge git 2>&1 | Out-Null
Log "STEP 2/4  done"

Log "STEP 3/4  upgrading pip"
& $py -m pip install --upgrade pip --quiet 2>&1 | Out-Null
Log "STEP 3/4  done"

Log "STEP 4/4  installing python packages (this is the long one, ~10-20 min)"
$pkgs = @(
    "numpy", "pandas", "pyarrow", "scipy", "scikit-learn",
    "lightgbm", "shap", "networkx",
    "vaderSentiment", "nrclex", "textblob",
    "torch", "transformers", "sentence-transformers", "datasets",
    "umap-learn", "hdbscan", "bertopic", "gensim",
    "streamlit", "plotly", "altair", "pydeck",
    "python-docx", "python-pptx", "openpyxl", "XlsxWriter",
    "matplotlib", "seaborn", "tqdm", "pyyaml", "requests", "joblib"
)
foreach ($p in $pkgs) {
    & $py -m pip install $p --quiet 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) { Log "    ok   $p" } else { Log "    FAIL $p" }
}
Log "STEP 4/4  done"
Log "=== SETUP COMPLETE ==="
& $py -c "import sys; print('python', sys.version)" 2>&1 | Tee-Object -Append -FilePath $log
