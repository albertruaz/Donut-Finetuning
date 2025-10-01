# GPU 메모리 정리 스크립트
Write-Host "=== GPU Memory Cleanup Started ==="

# 현재 GPU 상태 확인
Write-Host "Current GPU status:"
nvidia-smi

Write-Host ""
Write-Host "Cleaning up GPU memory..."

# Python 프로세스 중 GPU를 사용하는 것들 찾기 및 종료
$gpuProcesses = nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits
if ($gpuProcesses) {
    $gpuProcesses | ForEach-Object {
        $processPid = $_.Trim()
        if ($processPid -and $processPid -ne "No running processes found") {
            try {
                $process = Get-Process -Id $processPid -ErrorAction SilentlyContinue
                if ($process -and $process.ProcessName -eq "python") {
                    Write-Host "Terminating Python process with PID: $processPid"
                    Stop-Process -Id $processPid -Force -ErrorAction SilentlyContinue
                    Start-Sleep -Seconds 2
                }
            } catch {
                Write-Host "Could not terminate process $processPid"
            }
        }
    }
}

# PyTorch 캐시 정리를 위한 Python 스크립트 실행
$cleanupScript = @"
import torch
import gc

try:
    if torch.cuda.is_available():
        print('GPU available, clearing cache...')
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        print('GPU cache cleared successfully')
        
        # 메모리 상태 출력
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        print(f'GPU Memory - Allocated: {allocated / 1024**3:.2f}GB, Reserved: {reserved / 1024**3:.2f}GB')
    else:
        print('No GPU available')
except Exception as e:
    print(f'Error during cleanup: {e}')
"@

Write-Host "Running PyTorch cache cleanup..."
$cleanupScript | python

Write-Host ""
Write-Host "Final GPU status:"
nvidia-smi

Write-Host "=== GPU Memory Cleanup Completed ==="