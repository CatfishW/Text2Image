# Startup script for the backend API Gateway (Windows PowerShell)
# This script starts the FastAPI gateway server

Write-Host "Starting Text-to-Image API Gateway..." -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & .\venv\Scripts\Activate.ps1
}

# Check if dependencies are installed
try {
    python -c "import fastapi" 2>$null
} catch {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "⚠ Warning: .env file not found" -ForegroundColor Yellow
    Write-Host "  Using default configuration: TEXT2IMAGE_SERVER_URL=http://localhost:8001" -ForegroundColor Yellow
    Write-Host "  Create .env file to customize settings" -ForegroundColor Yellow
}

# Check connection to text-to-image server
$text2imageUrl = if ($env:TEXT2IMAGE_SERVER_URL) { $env:TEXT2IMAGE_SERVER_URL } else { "http://localhost:8001" }
Write-Host "Checking connection to text-to-image server at $text2imageUrl..." -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri "$text2imageUrl/health" -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
    Write-Host "✓ Text-to-image server is reachable" -ForegroundColor Green
} catch {
    Write-Host "⚠ Warning: Cannot reach text-to-image server" -ForegroundColor Yellow
    Write-Host "  Make sure text2image_server.py is running on $text2imageUrl" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Starting API Gateway on http://0.0.0.0:8000" -ForegroundColor Green
Write-Host "API docs will be available at http://localhost:8000/docs" -ForegroundColor Green
Write-Host ""

# Start the server
python main.py

