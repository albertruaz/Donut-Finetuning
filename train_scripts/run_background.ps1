param(
    [switch]$Stop,
    [switch]$Status
)

$workDir = "c:\Users\ML\Desktop\AI\Donut-Finetuning"
$pidFile = "$workDir\train_scripts\training.pid"

# 상태 확인
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

# 프로세스 중지
if ($Stop) {
    if (Test-Path $pidFile) {
        $trainingPid = Get-Content $pidFile
        Stop-Process -Id $trainingPid -Force -ErrorAction SilentlyContinue
        Remove-Item $pidFile -Force
        Write-Host "Training stopped"
        
        # GPU 정리 실행
        Write-Host "Running GPU cleanup..."
        & "$workDir\train_scripts\clean_gpu.ps1"
    } else {
        Write-Host "No training process found"
    }
    exit 0
}

# 이미 실행 중인지 확인
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
Write-Host "All logs will be sent to WandB: https://wandb.ai/vingle/donut-finetuning"

# 실행할 명령어를 스크립트 블록으로 생성
$scriptBlock = @"
Set-Location '$workDir'
`$env:PYTHONPATH = '$workDir'
try {
    Write-Host "=== Starting training at `$(Get-Date) ==="
    Write-Host "Working directory: '$workDir'"
    Write-Host "Logs will be available at WandB dashboard"
    Write-Host ""
    
    & C:\Users\ML\Miniconda3\Scripts\activate.bat donut
    & C:\Users\ML\Miniconda3\envs\donut\python.exe '$workDir\src\train.py' --config '$workDir\config.json'
    
    Write-Host ""
    Write-Host "=== Training completed at `$(Get-Date) ==="
} catch {
    Write-Host "ERROR: `$(`$_.Exception.Message)"
} finally {
    if (Test-Path '$pidFile') {
        Remove-Item '$pidFile' -Force
    }
}
"@

# 터미널과 독립적으로 실행되는 백그라운드 프로세스 시작
$process = Start-Process powershell -ArgumentList @(
    "-NoProfile", 
    "-WindowStyle", "Hidden",
    "-Command", $scriptBlock
) -PassThru

# PID 저장
$process.Id | Out-File $pidFile -Encoding UTF8

Write-Host "Training started (PID: $($process.Id))"
Write-Host "Monitor at: https://wandb.ai/vingle/donut-finetuning"
Write-Host "Status:     .\train_scripts\run_background.ps1 -Status"
Write-Host "Stop:       .\train_scripts\run_background.ps1 -Stop"