# Auto-Restart Guide

The Text2Image server now includes auto-restart functionality to automatically recover from crashes.

## Quick Start

### Windows
```batch
run_with_restart.bat
```

### Linux/Mac/Cross-platform
```bash
python run_with_restart.py
```

### With Custom Arguments
```bash
# Python script (passes through all arguments)
python run_with_restart.py --host 0.0.0.0 --port 8010

# Windows batch
run_with_restart.bat --host 0.0.0.0 --port 8010
```

## Features

- **Automatic Restart**: Server automatically restarts after crashes
- **Crash Logging**: All crashes are logged to `server_crash.log`
- **Server Logging**: Normal server logs are written to `server.log`
- **Smart Restart Delays**: Fast restart for initial crashes, then standard delays
- **Graceful Shutdown**: Properly handles Ctrl+C and shutdown signals
- **Exit Code Handling**: Only restarts on crashes, stops on normal exit

## Configuration

Edit `run_with_restart.py` to customize:
- `MAX_RESTART_ATTEMPTS`: Maximum restarts before giving up (0 = unlimited)
- `RESTART_DELAY`: Seconds to wait before restarting (default: 5)
- `RESTART_DELAY_FAST`: Fast restart delay for initial crashes (default: 2)

## Log Files

- `server_crash.log`: Crash reports and error traces
- `server.log`: Standard server logs and operations

## Normal Operation

When you want to stop the server:
1. Press `Ctrl+C` once
2. Wait for graceful shutdown (server completes current requests)
3. Server will NOT restart if stopped normally

## Monitoring

The wrapper script shows:
- Restart attempt number
- Server process ID (PID)
- Exit codes
- Crash timestamps

All information is also logged to `server_crash.log` for later review.

