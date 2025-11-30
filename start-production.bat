@echo off
REM Production start script for Text2Image Web Application (Windows)
REM Starts both backend and frontend services

setlocal enabledelayedexpansion

REM Configuration
set BACKEND_DIR=backend
set FRONTEND_DIR=frontend
set BACKEND_PORT=21115
set FRONTEND_PORT=3000

echo ========================================
echo Starting Text2Image Web Application
echo ========================================
echo.

REM Start Backend
echo Starting Backend API Gateway...
cd %BACKEND_DIR%
if %errorlevel% neq 0 (
    echo [ERROR] Failed to change to backend directory
    exit /b 1
)

if not exist "venv" (
    echo [ERROR] Virtual environment not found. Run deploy.bat first.
    exit /b 1
)

call venv\Scripts\activate.bat

REM Set environment variables
set NEXT_PUBLIC_API_URL=http://localhost:%BACKEND_PORT%

REM Start backend in new window
start "Text2Image Backend" cmd /k "python main.py"

cd ..

REM Wait a bit for backend to start
timeout /t 3 /nobreak >nul

echo [OK] Backend started on port %BACKEND_PORT%
echo.

REM Start Frontend
echo Starting Frontend...
cd %FRONTEND_DIR%
if %errorlevel% neq 0 (
    echo [ERROR] Failed to change to frontend directory
    exit /b 1
)

if not exist ".next" (
    echo [ERROR] Frontend not built. Run deploy.bat first.
    exit /b 1
)

REM Set API URL
set NEXT_PUBLIC_API_URL=http://localhost:%BACKEND_PORT%

REM Start frontend in new window
start "Text2Image Frontend" cmd /k "npm start"

cd ..

echo [OK] Frontend started on port %FRONTEND_PORT%
echo.

REM Summary
echo ========================================
echo Services Running
echo ========================================
echo.
echo [OK] Backend API Gateway:
echo    http://localhost:%BACKEND_PORT%
echo    API Docs: http://localhost:%BACKEND_PORT%/docs
echo.
echo [OK] Frontend:
echo    http://localhost:%FRONTEND_PORT%
echo.
echo Services are running in separate windows.
echo Close those windows to stop the services.
echo.

pause

endlocal

