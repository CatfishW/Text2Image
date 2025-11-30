#!/usr/bin/env python3
"""
Production server runner with Gunicorn
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker for GPU model sharing
        log_level="info",
    )

