@echo off
REM Deployment script for Text2Image Web Application (Windows)
REM This script deploys both frontend and backend services

setlocal enabledelayedexpansion

REM Configuration
set FRONTEND_DIR=frontend
set BACKEND_DIR=backend
set BACKEND_PORT=21115
set FRONTEND_PORT=21116
set TEXT2IMAGE_SERVER_URL=http://localhost:8010

echo ========================================
echo Text2Image Web Application Deployment
echo ========================================
echo.

REM Check prerequisites
echo Checking prerequisites...

where node >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js is not installed. Please install Node.js 18+ first.
    exit /b 1
)

where npm >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] npm is not installed. Please install npm first.
    exit /b 1
)

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python 3 is not installed. Please install Python 3.8+ first.
    exit /b 1
)

echo [OK] All prerequisites met
echo.

REM Deploy Frontend
echo ========================================
echo Deploying Frontend (Next.js)
echo ========================================

cd %FRONTEND_DIR%
if %errorlevel% neq 0 (
    echo [ERROR] Failed to change to frontend directory
    exit /b 1
)

REM Check if node_modules exists
if not exist "node_modules" (
    echo Installing frontend dependencies...
    call npm install
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to install frontend dependencies
        cd ..
        exit /b 1
    )
) else (
    echo Updating frontend dependencies...
    call npm install
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to update frontend dependencies
        cd ..
        exit /b 1
    )
)

REM Set API URL environment variable
set NEXT_PUBLIC_API_URL=http://localhost:%BACKEND_PORT%

REM Build frontend
echo Building frontend for production...
call npm run build
if !errorlevel! neq 0 (
    echo [ERROR] Failed to build frontend
    cd ..
    exit /b 1
)

echo [OK] Frontend built successfully
echo.

REM Return to root directory
cd ..

REM Deploy Backend
echo ========================================
echo Deploying Backend (API Gateway)
echo ========================================

cd %BACKEND_DIR%
if %errorlevel% neq 0 (
    echo [ERROR] Failed to change to backend directory
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment
        cd ..
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if !errorlevel! neq 0 (
    echo [ERROR] Failed to activate virtual environment
    cd ..
    exit /b 1
)

REM Install/update dependencies
echo Installing backend dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
if !errorlevel! neq 0 (
    echo [ERROR] Failed to install backend dependencies
    cd ..
    exit /b 1
)

REM Create .env file if it doesn't exist
if not exist ".env" (
    echo Creating .env file with default configuration...
    (
        echo TEXT2IMAGE_SERVER_URL=%TEXT2IMAGE_SERVER_URL%
        echo REQUEST_TIMEOUT=300
        echo MAX_CONCURRENT_REQUESTS=10
    ) > .env
    echo [OK] Created .env file
) else (
    echo [OK] .env file already exists
)

echo [OK] Backend dependencies installed
echo.

REM Return to root directory
cd ..

REM Summary
echo ========================================
echo Deployment Summary
echo ========================================
echo.
echo [OK] Frontend:
echo   - Built and ready in: %FRONTEND_DIR%
echo   - Production build: %FRONTEND_DIR%\.next
echo   - API URL: http://localhost:%BACKEND_PORT%
echo.
echo [OK] Backend:
echo   - Dependencies installed in: %BACKEND_DIR%\venv
echo   - Configuration: %BACKEND_DIR%\.env
echo   - Port: %BACKEND_PORT%
echo.
echo Next Steps:
echo.
echo 1. Start the Text-to-Image Server (if not running):
echo    python text2image_server.py --port 8001
echo.
echo 2. Start the Backend API Gateway:
echo    cd backend
echo    venv\Scripts\activate
echo    python main.py
echo    Or: cd backend ^&^& start.bat
echo.
echo 3. Start the Frontend:
echo    cd frontend
echo    npm start
echo    Frontend will be available at: http://localhost:%FRONTEND_PORT%
echo.
echo ========================================
echo Deployment completed successfully!
echo ========================================

endlocal

