#!/bin/bash
# Production start script for Text2Image Web Application
# Starts both backend and frontend services

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
BACKEND_DIR="backend"
FRONTEND_DIR="frontend"
BACKEND_PORT=${BACKEND_PORT:-21115}
FRONTEND_PORT=${FRONTEND_PORT:-3000}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Text2Image Web Application${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down services...${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start Backend
echo -e "${YELLOW}Starting Backend API Gateway...${NC}"
cd "$BACKEND_DIR" || exit 1

if [ ! -d "venv" ]; then
    echo -e "${RED}Error: Virtual environment not found. Run deploy.sh first.${NC}"
    exit 1
fi

source venv/bin/activate

# Set environment variables
export NEXT_PUBLIC_API_URL="http://localhost:${BACKEND_PORT}"

# Start backend in background
python main.py &
BACKEND_PID=$!

cd ..

# Wait for backend to be ready
echo -e "${YELLOW}Waiting for backend to start...${NC}"
sleep 3

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}Error: Backend failed to start${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Backend started on port ${BACKEND_PORT}${NC}"
echo ""

# Start Frontend
echo -e "${YELLOW}Starting Frontend...${NC}"
cd "$FRONTEND_DIR" || exit 1

if [ ! -d ".next" ]; then
    echo -e "${RED}Error: Frontend not built. Run deploy.sh first.${NC}"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Set API URL
export NEXT_PUBLIC_API_URL="http://localhost:${BACKEND_PORT}"

# Start frontend in background
npm start &
FRONTEND_PID=$!

cd ..

echo -e "${GREEN}✓ Frontend started on port ${FRONTEND_PORT}${NC}"
echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Services Running${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}✓ Backend API Gateway:${NC}"
echo "   http://localhost:${BACKEND_PORT}"
echo "   API Docs: http://localhost:${BACKEND_PORT}/docs"
echo ""
echo -e "${GREEN}✓ Frontend:${NC}"
echo "   http://localhost:${FRONTEND_PORT}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# Wait for processes
wait

