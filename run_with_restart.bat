@echo off
REM Auto-restart wrapper for text2image_server.py (Windows batch script)
REM Automatically restarts the server if it crashes

setlocal enabledelayedexpansion

set MAX_RESTARTS=1000
set RESTART_DELAY=5
set RESTART_COUNT=0
set LOG_FILE=server_crash.log

:start
set /a RESTART_COUNT+=1

if %MAX_RESTARTS% GTR 0 (
    if !RESTART_COUNT! GTR %MAX_RESTARTS% (
        echo [%date% %time%] ERROR: Maximum restart attempts (%MAX_RESTARTS%) reached. Stopping. >> %LOG_FILE%
        echo ERROR: Maximum restart attempts reached. Stopping.
        exit /b 1
    )
)

echo [%date% %time%] Starting server - Restart attempt #!RESTART_COUNT! >> %LOG_FILE%
echo Starting server - Restart attempt #!RESTART_COUNT!

python text2image_server.py %*

set EXIT_CODE=%ERRORLEVEL%

if %EXIT_CODE% EQU 0 (
    echo [%date% %time%] Server exited normally. Stopping auto-restart. >> %LOG_FILE%
    echo Server exited normally. Stopping auto-restart.
    exit /b 0
)

echo [%date% %time%] Server crashed with exit code %EXIT_CODE% (Restart #!RESTART_COUNT!) >> %LOG_FILE%
echo Server crashed with exit code %EXIT_CODE% (Restart #!RESTART_COUNT!)
echo Waiting %RESTART_DELAY% seconds before restarting...

timeout /t %RESTART_DELAY% /nobreak >nul

goto start

