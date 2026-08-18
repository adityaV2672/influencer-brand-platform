# ==========================================================================
# Rebuild every artifact. Safe to re-run at any time: completed stages are
# skipped or resumed rather than recomputed.
#
# The first production run of this died overnight when the laptop slept, so
# this version asks Windows to stay awake for the duration and the expensive
# NLP stage checkpoints to disk as it goes.
# ==========================================================================
$ErrorActionPreference = "Continue"
$ProgressPreference    = "SilentlyContinue"

$proj = "$env:USERPROFILE\influencer-platform"
$py   = "$env:USERPROFILE\anaconda3\envs\influencer\python.exe"
$log  = "$proj\pipeline_log.txt"

Set-Location $proj
$env:PYTHONPATH                   = $proj
$env:PYTHONIOENCODING             = "utf-8"
$env:PYTHONUTF8                   = "1"
$env:TOKENIZERS_PARALLELISM       = "false"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:OMP_NUM_THREADS              = "12"

# --- keep the machine awake for the duration (no admin rights needed) -----
$sig = @'
[DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
try {
    $power = Add-Type -MemberDefinition $sig -Name "Power" -Namespace "Win32" -PassThru
    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
    [void]$power::SetThreadExecutionState([uint32]"0x80000040" -bor [uint32]"0x00000001")
    "sleep prevention enabled" | Out-File -Append -Encoding utf8 $log
} catch {
    "WARNING: could not disable sleep - $($_.Exception.Message)" | Out-File -Append -Encoding utf8 $log
}

"=== pipeline started $(Get-Date) ===" | Out-File -Append -Encoding utf8 $log

# -u = unbuffered, so the log updates live rather than in bursts.
& $py -u run_pipeline.py --continue-on-error 2>&1 |
    ForEach-Object { $_ | Out-File -Append -Encoding utf8 $log }

"=== pipeline finished $(Get-Date) ===" | Out-File -Append -Encoding utf8 $log

# Release the sleep block (ES_CONTINUOUS only).
try { [void]$power::SetThreadExecutionState([uint32]"0x80000000") } catch {}
