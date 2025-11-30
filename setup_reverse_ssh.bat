@echo off
REM Reverse SSH Tunnel Setup Script for Windows
REM Run this on the GPU machine (backend) to create a reverse SSH tunnel to the public server

REM Configuration
set PUBLIC_SERVER_USER=login
set PUBLIC_SERVER_HOST=vpn.agaii.org
set PUBLIC_SERVER_PORT=22
set PASSWORD=Clb1997521

REM Local ports to tunnel
set BACKEND_PORT=21115
set TEXT2IMAGE_PORT=8010

REM Remote ports on public server (where the tunnels will be accessible)
set TUNNEL_BACKEND_PORT=21115
set TUNNEL_TEXT2IMAGE_PORT=8010

echo ============================================================
echo Setting up reverse SSH tunnels...
echo Backend (GPU machine) -^> Public Server
echo ============================================================
echo.
echo Backend Gateway (port %BACKEND_PORT%) will be accessible on public server at localhost:%TUNNEL_BACKEND_PORT%
echo Text2Image Server (port %TEXT2IMAGE_PORT%) will be accessible on public server at localhost:%TUNNEL_TEXT2IMAGE_PORT%
echo.

REM Check if plink.exe is available (PuTTY's command-line tool)
where plink.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Using plink.exe for SSH tunneling...
    echo.
    
    REM Start reverse tunnel for backend gateway
    echo Starting reverse tunnel for Backend Gateway (port %BACKEND_PORT%)...
    start "SSH Tunnel - Backend Gateway" /MIN plink.exe -ssh -R %TUNNEL_BACKEND_PORT%:localhost:%BACKEND_PORT% -N -pw %PASSWORD% -o ExitOnForwardFailure=yes -o ServerAliveInterval=60 -o ServerAliveCountMax=3 %PUBLIC_SERVER_USER%@%PUBLIC_SERVER_HOST% -P %PUBLIC_SERVER_PORT%
    
    REM Wait a moment
    timeout /t 2 /nobreak >nul
    
    REM Start reverse tunnel for text2image server
    echo Starting reverse tunnel for Text2Image Server (port %TEXT2IMAGE_PORT%)...
    start "SSH Tunnel - Text2Image Server" /MIN plink.exe -ssh -R %TUNNEL_TEXT2IMAGE_PORT%:localhost:%TEXT2IMAGE_PORT% -N -pw %PASSWORD% -o ExitOnForwardFailure=yes -o ServerAliveInterval=60 -o ServerAliveCountMax=3 %PUBLIC_SERVER_USER%@%PUBLIC_SERVER_HOST% -P %PUBLIC_SERVER_PORT%
    
    echo.
    echo ============================================================
    echo Reverse SSH tunnels established!
    echo ============================================================
    echo.
    echo Backend Gateway: http://localhost:%TUNNEL_BACKEND_PORT% on public server
    echo Text2Image Server: http://localhost:%TUNNEL_TEXT2IMAGE_PORT% on public server
    echo.
    echo Tunnels are running in background windows (minimized).
    echo To stop the tunnels, close the minimized windows or use Task Manager.
    echo.
    pause
    exit /b 0
)

REM Fallback: Try using ssh (will prompt for password interactively)
where ssh.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo plink.exe not found. Using ssh.exe (will prompt for password)...
    echo.
    echo NOTE: Standard ssh.exe does not support password in command line.
    echo You will be prompted to enter the password: %PASSWORD%
    echo.
    pause
    
    REM Start reverse tunnel for backend gateway
    echo Starting reverse tunnel for Backend Gateway (port %BACKEND_PORT%)...
    start "SSH Tunnel - Backend Gateway" /MIN ssh.exe -R %TUNNEL_BACKEND_PORT%:localhost:%BACKEND_PORT% -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=60 -o ServerAliveCountMax=3 %PUBLIC_SERVER_USER%@%PUBLIC_SERVER_HOST% -p %PUBLIC_SERVER_PORT%
    
    REM Wait a moment
    timeout /t 2 /nobreak >nul
    
    REM Start reverse tunnel for text2image server
    echo Starting reverse tunnel for Text2Image Server (port %TEXT2IMAGE_PORT%)...
    start "SSH Tunnel - Text2Image Server" /MIN ssh.exe -R %TUNNEL_TEXT2IMAGE_PORT%:localhost:%TEXT2IMAGE_PORT% -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=60 -o ServerAliveCountMax=3 %PUBLIC_SERVER_USER%@%PUBLIC_SERVER_HOST% -p %PUBLIC_SERVER_PORT%
    
    echo.
    echo ============================================================
    echo Reverse SSH tunnels established!
    echo ============================================================
    echo.
    echo Backend Gateway: http://localhost:%TUNNEL_BACKEND_PORT% on public server
    echo Text2Image Server: http://localhost:%TUNNEL_TEXT2IMAGE_PORT% on public server
    echo.
    echo Tunnels are running in background windows (minimized).
    echo To stop the tunnels, close the minimized windows or use Task Manager.
    echo.
    pause
    exit /b 0
)

REM If neither plink nor ssh is found
echo ERROR: Neither plink.exe nor ssh.exe found!
echo.
echo Please install one of the following:
echo   1. PuTTY (includes plink.exe) - Recommended for password authentication
echo      Download from: https://www.putty.org/
echo   2. OpenSSH for Windows (includes ssh.exe)
echo      Usually comes with Windows 10/11, or install via: Settings -^> Apps -^> Optional Features
echo.
pause
exit /b 1

