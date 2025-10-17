param(
    [switch]$Stop,
    [switch]$Status
)

$workDir = "C:\Users\ML\Desktop\Donut-Finetuning"
$pidFile = "$workDir\train_scripts\training.pid"
$logOut = "$workDir\train_scripts\train_output.log"
$logErr = "$workDir\train_scripts\train_error.log"

# ==============================
# 상태 확인
# ==============================
if ($Status) {
    if (Test-Path $pidFile) {
        $trainingPid = Get-Content $pidFile
        if (Get-Process -Id $trainingPid -ErrorAction SilentlyContinue) {
            Write-Host "Training is running (PID: $trainingPid)"
            Write-Host "Check progress at: https://wandb.ai/vingle/donut-finetuning"
        } else {
            Write-Host "Training is not running"
            Remove-Item $pidFile -Force
        }
    } else {
        Write-Host "Training is not running"
    }
    exit 0
}

# ==============================
# 프로세스 중지
# ==============================
if ($Stop) {
    if (Test-Path $pidFile) {
        $trainingPid = Get-Content $pidFile
        Stop-Process -Id $trainingPid -Force -ErrorAction SilentlyContinue
        Remove-Item $pidFile -Force
        Write-Host "Training stopped"

        # GPU 정리
        if (Test-Path "$workDir\train_scripts\clean_gpu.ps1") {
            Write-Host "Running GPU cleanup..."
            & "$workDir\train_scripts\clean_gpu.ps1"
        }
    } else {
        Write-Host "No training process found"
    }
    exit 0
}

# ==============================
# 이미 실행 중인지 확인
# ==============================
if (Test-Path $pidFile) {
    $trainingPid = Get-Content $pidFile
    if (Get-Process -Id $trainingPid -ErrorAction SilentlyContinue) {
        Write-Host "Training already running (PID: $trainingPid)"
        Write-Host "Use -Stop to stop it first"
        exit 1
    } else {
        Remove-Item $pidFile -Force
    }
}

Write-Host "Starting training in background..."
Write-Host "Logs: $logOut, $logErr"
Write-Host "Monitor at: https://wandb.ai/vingle/donut-finetuning"

# ==============================
# 실행할 스크립트 블록 (PowerShell용 Conda 환경)
# ==============================
$scriptBlock = @"
Set-Location '$workDir'
`$env:PYTHONPATH = '$workDir'

try {
    Write-Host "=== Starting training at `$(Get-Date) ==="
    Write-Host "Working directory: '$workDir'"

    # PowerShell용 Conda 초기화
    & "C:\Users\ML\miniconda3\shell\condabin\conda-hook.ps1"
    conda activate donut

    # GPU 상태 로깅
    Write-Host "=== GPU Status ==="
    try {
        & nvidia-smi
    } catch {
        Write-Host "nvidia-smi not available"
    }

    # Python 학습 실행
    python "$workDir\src\train.py" --config "$workDir\config.json"

    Write-Host "=== Training completed at `$(Get-Date) ==="
}
catch {
    Write-Host "ERROR: `$(`$_.Exception.Message)"
}
finally {
    if (Test-Path '$pidFile') {
        Remove-Item '$pidFile' -Force
    }
}
"@

# ==============================
# 백그라운드 프로세스 실행 (로그 저장 포함)
# ==============================
$process = Start-Process powershell -ArgumentList @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-Command", $scriptBlock
) -RedirectStandardOutput $logOut `
  -RedirectStandardError $logErr `
  -WindowStyle Hidden `
  -PassThru

# PID 저장
$process.Id | Out-File $pidFile -Encoding UTF8

Write-Host "Training started (PID: $($process.Id))"
Write-Host "Status: .\train_scripts\run_background.ps1 -Status"
Write-Host "Stop:   .\train_scripts\run_background.ps1 -Stop"
Write-Host "Logs:   $logOut"
