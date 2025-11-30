# Reverse SSH Tunnel Setup Script for Windows PowerShell
# Run this on the GPU machine (backend) to create a reverse SSH tunnel to the public server

# Configuration - UPDATE THESE VALUES
$PUBLIC_SERVER_USER = "your-username"
$PUBLIC_SERVER_HOST = "your-public-server.com"
$PUBLIC_SERVER_PORT = 22
$BACKEND_PORT = 8000
$TUNNEL_PORT = 8000  # Port on public server that will forward to backend

Write-Host "Setting up reverse SSH tunnel..." -ForegroundColor Cyan
Write-Host "Backend (GPU machine) -> Public Server"
Write-Host "Local port $BACKEND_PORT will be accessible on public server at localhost:$TUNNEL_PORT"
Write-Host ""

# Check if SSH is available
if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    Write-Host "✗ SSH not found. Please install OpenSSH or use WSL." -ForegroundColor Red
    exit 1
}

# Create reverse SSH tunnel
# -R: Reverse tunnel (public_server:port -> local:port)
# -N: Don't execute remote commands
# -f: Background (not available in Windows SSH, so we'll use Start-Process)
# -o ExitOnForwardFailure=yes: Exit if port forwarding fails
# -o ServerAliveInterval=60: Keep connection alive

$sshArgs = @(
    "-R", "${TUNNEL_PORT}:localhost:${BACKEND_PORT}",
    "-N",
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ServerAliveInterval=60",
    "-o", "ServerAliveCountMax=3",
    "${PUBLIC_SERVER_USER}@${PUBLIC_SERVER_HOST}",
    "-p", "${PUBLIC_SERVER_PORT}"
)

Write-Host "Starting SSH tunnel in background..." -ForegroundColor Yellow
$process = Start-Process -FilePath "ssh" -ArgumentList $sshArgs -PassThru -WindowStyle Hidden

Start-Sleep -Seconds 2

if ($process -and -not $process.HasExited) {
    Write-Host "✓ Reverse SSH tunnel established successfully!" -ForegroundColor Green
    Write-Host "Backend API is now accessible on public server at localhost:$TUNNEL_PORT"
    Write-Host ""
    Write-Host "Process ID: $($process.Id)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To check if tunnel is running:" -ForegroundColor Yellow
    Write-Host "  ssh $PUBLIC_SERVER_USER@$PUBLIC_SERVER_HOST 'netstat -tlnp | grep $TUNNEL_PORT'"
    Write-Host ""
    Write-Host "To kill the tunnel:" -ForegroundColor Yellow
    Write-Host "  Stop-Process -Id $($process.Id)"
} else {
    Write-Host "✗ Failed to establish reverse SSH tunnel" -ForegroundColor Red
    Write-Host "Please check:" -ForegroundColor Yellow
    Write-Host "  1. SSH key authentication is set up"
    Write-Host "  2. Public server allows remote port forwarding (GatewayPorts in sshd_config)"
    Write-Host "  3. Port $TUNNEL_PORT is not already in use on public server"
    exit 1
}

