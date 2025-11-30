#!/bin/bash
# Startup script for the backend API Gateway
# This script starts the FastAPI gateway server

echo "Starting Text-to-Image API Gateway..."
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠ Warning: .env file not found"
    echo "  Using default configuration: TEXT2IMAGE_SERVER_URL=http://localhost:8001"
    echo "  Create .env file to customize settings"
fi

# Check connection to text-to-image server
TEXT2IMAGE_URL=${TEXT2IMAGE_SERVER_URL:-http://localhost:8001}
echo "Checking connection to text-to-image server at $TEXT2IMAGE_URL..."
if command -v curl &> /dev/null; then
    if curl -s --connect-timeout 2 "$TEXT2IMAGE_URL/health" > /dev/null 2>&1; then
        echo "✓ Text-to-image server is reachable"
    else
        echo "⚠ Warning: Cannot reach text-to-image server"
        echo "  Make sure text2image_server.py is running on $TEXT2IMAGE_URL"
    fi
fi

echo ""
echo "Starting API Gateway on http://0.0.0.0:8000"
echo "API docs will be available at http://localhost:8000/docs"
echo ""

# Start the server
python main.py

