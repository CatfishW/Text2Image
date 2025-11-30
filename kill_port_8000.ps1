# Kill processes running on port 8000 (PowerShell)
# Usage: .\kill_port_8000.ps1

Write-Host "Killing processes on port 8000..." -ForegroundColor Cyan

try {
    # Get processes using port 8000
    $connections = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
    
    if ($connections) {
        $pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique
        
        foreach ($pid in $pids) {
            $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
            if ($process) {
                Write-Host "Found process: $($process.ProcessName) (PID: $pid)" -ForegroundColor Yellow
                try {
                    Stop-Process -Id $pid -Force
                    Write-Host "✓ Successfully killed process $pid" -ForegroundColor Green
                } catch {
                    Write-Host "✗ Failed to kill process $pid: $_" -ForegroundColor Red
                    Write-Host "  Try running as administrator" -ForegroundColor Yellow
                }
            }
        }
    } else {
        Write-Host "No processes found on port 8000" -ForegroundColor Green
    }
    
    # Verify
    Start-Sleep -Seconds 1
    $remaining = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
    
    if ($remaining) {
        Write-Host "`n⚠ Warning: Port 8000 may still be in use" -ForegroundColor Yellow
        Write-Host "Remaining processes:" -ForegroundColor Yellow
        $remaining | Select-Object OwningProcess | ForEach-Object {
            $proc = Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue
            if ($proc) {
                Write-Host "  - $($proc.ProcessName) (PID: $($proc.Id))" -ForegroundColor Yellow
            }
        }
    } else {
        Write-Host "`n✓ Port 8000 is now free" -ForegroundColor Green
    }
    
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host "Make sure you're running PowerShell as administrator" -ForegroundColor Yellow
}

