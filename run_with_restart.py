#!/usr/bin/env python3
"""
Auto-restart wrapper for text2image_server.py
Automatically restarts the server if it crashes or exits unexpectedly.
"""

import subprocess
import sys
import time
import signal
import os
from datetime import datetime
from pathlib import Path

# Configuration
MAX_RESTART_ATTEMPTS = 1000  # Maximum restart attempts (0 = unlimited)
RESTART_DELAY = 5  # Seconds to wait before restarting after crash
RESTART_DELAY_FAST = 2  # Seconds for fast restart (first few restarts)
FAST_RESTART_COUNT = 3  # Number of fast restarts before using normal delay

# Log file for crash records
LOG_FILE = Path("server_crash.log")

def log_message(message: str):
    """Log a message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    print(log_entry.strip())
    
    # Append to log file
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Warning: Could not write to log file: {e}")

def run_server(args):
    """Run the server and return the process"""
    script_path = Path(__file__).parent / "text2image_server.py"
    
    if not script_path.exists():
        log_message(f"ERROR: Server script not found: {script_path}")
        sys.exit(1)
    
    # Build command
    cmd = [sys.executable, str(script_path)] + args
    
    log_message(f"Starting server: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        return process
    except Exception as e:
        log_message(f"ERROR: Failed to start server: {e}")
        raise

def main():
    """Main loop with auto-restart"""
    # Parse command line arguments (pass through to server)
    server_args = sys.argv[1:]
    
    restart_count = 0
    consecutive_crashes = 0
    last_exit_code = None
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        log_message("Received shutdown signal, stopping server...")
        if 'process' in locals():
            try:
                process.terminate()
                process.wait(timeout=10)
            except:
                try:
                    process.kill()
                except:
                    pass
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    log_message("=" * 60)
    log_message("Text2Image Server Auto-Restart Wrapper Started")
    log_message(f"Maximum restart attempts: {MAX_RESTART_ATTEMPTS if MAX_RESTART_ATTEMPTS > 0 else 'Unlimited'}")
    log_message("=" * 60)
    
    while True:
        # Check restart limit
        if MAX_RESTART_ATTEMPTS > 0 and restart_count >= MAX_RESTART_ATTEMPTS:
            log_message(f"ERROR: Maximum restart attempts ({MAX_RESTART_ATTEMPTS}) reached. Stopping.")
            sys.exit(1)
        
        restart_count += 1
        
        try:
            # Start the server
            process = run_server(server_args)
            
            # Monitor output and wait for process to exit
            log_message(f"Server started (PID: {process.pid}) - Restart attempt #{restart_count}")
            
            # Stream output in real-time
            try:
                for line in process.stdout:
                    print(line, end='')
            except Exception as e:
                log_message(f"Warning: Error reading server output: {e}")
            
            # Wait for process to complete
            exit_code = process.wait()
            last_exit_code = exit_code
            
            # Check exit code
            if exit_code == 0:
                log_message("Server exited normally (exit code 0). Stopping auto-restart.")
                sys.exit(0)
            elif exit_code == 130:  # SIGINT (Ctrl+C)
                log_message("Server received interrupt signal. Stopping auto-restart.")
                sys.exit(0)
            else:
                consecutive_crashes += 1
                log_message(f"⚠️  Server crashed with exit code {exit_code} (Restart #{restart_count}, Consecutive crashes: {consecutive_crashes})")
                
                # Determine restart delay
                if consecutive_crashes <= FAST_RESTART_COUNT:
                    delay = RESTART_DELAY_FAST
                    log_message(f"Fast restart: Waiting {delay} seconds before restarting...")
                else:
                    delay = RESTART_DELAY
                    log_message(f"Standard restart: Waiting {delay} seconds before restarting...")
                
                # Wait before restarting
                time.sleep(delay)
                
        except KeyboardInterrupt:
            log_message("Received keyboard interrupt. Stopping server...")
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                try:
                    process.kill()
                except:
                    pass
            sys.exit(0)
            
        except Exception as e:
            consecutive_crashes += 1
            log_message(f"⚠️  Exception while running server: {e} (Restart #{restart_count}, Consecutive crashes: {consecutive_crashes})")
            
            # Exponential backoff for repeated exceptions
            delay = min(RESTART_DELAY * (2 ** min(consecutive_crashes - FAST_RESTART_COUNT, 4)), 60)
            log_message(f"Waiting {delay} seconds before restarting...")
            time.sleep(delay)

if __name__ == "__main__":
    main()

