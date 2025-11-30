# Deployment Guide

This guide explains how to deploy the Text2Image web application using the provided deployment scripts.

## Prerequisites

Before deploying, ensure you have:

- **Node.js 18+** and npm installed
- **Python 3.8+** installed
- **Text-to-Image Server** running (text2image_server.py)

## Quick Start

### Linux/Mac

1. **Deploy the application:**
   ```bash
   ./deploy.sh
   ```

2. **Start production services:**
   ```bash
   ./start-production.sh
   ```

### Windows

1. **Deploy the application:**
   ```cmd
   deploy.bat
   ```

2. **Start production services:**
   ```cmd
   start-production.bat
   ```

## What the Deployment Scripts Do

### `deploy.sh` / `deploy.bat`

These scripts:

1. ✅ Check prerequisites (Node.js, Python, npm)
2. ✅ Install frontend dependencies (`npm install`)
3. ✅ Build frontend for production (`npm run build`)
4. ✅ Create Python virtual environment for backend
5. ✅ Install backend dependencies
6. ✅ Create `.env` file with default configuration
7. ✅ Verify all components are ready

### `start-production.sh` / `start-production.bat`

These scripts:

1. ✅ Start the backend API gateway
2. ✅ Start the frontend Next.js server
3. ✅ Display service URLs

## Manual Deployment

If you prefer to deploy manually:

### Frontend

```bash
cd frontend
npm install
export NEXT_PUBLIC_API_URL=http://localhost:21115
npm run build
npm start
```

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Create .env file
echo "TEXT2IMAGE_SERVER_URL=http://localhost:8001" > .env
echo "REQUEST_TIMEOUT=300" >> .env
echo "MAX_CONCURRENT_REQUESTS=10" >> .env

python main.py
```

## Configuration

### Backend Port

The backend API gateway runs on port **21115** by default.

To change it, modify:
- `backend/main.py` (line 295)
- `frontend/lib/api.ts` (default API URL)
- Environment variable `NEXT_PUBLIC_API_URL` when building frontend

### Frontend Port

The frontend runs on port **3000** by default (Next.js default).

To change it:
```bash
cd frontend
PORT=3001 npm start
```

### Text-to-Image Server URL

Configure the text-to-image server URL in `backend/.env`:

```env
TEXT2IMAGE_SERVER_URL=http://localhost:8001
REQUEST_TIMEOUT=300
MAX_CONCURRENT_REQUESTS=10
```

## Service URLs

After deployment:

- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:21115
- **API Documentation:** http://localhost:21115/docs
- **Health Check:** http://localhost:21115/health

## Troubleshooting

### Frontend can't connect to backend

1. Ensure backend is running on port 21115
2. Check `NEXT_PUBLIC_API_URL` environment variable
3. Verify CORS settings in backend

### Backend can't connect to text-to-image server

1. Ensure text2image_server.py is running
2. Check `TEXT2IMAGE_SERVER_URL` in `backend/.env`
3. Verify the server is accessible:
   ```bash
   curl http://localhost:8001/health
   ```

### Port already in use

If a port is already in use:

**Linux/Mac:**
```bash
# Find process using port
lsof -i :21115
# Kill process
kill -9 <PID>
```

**Windows:**
```cmd
# Find process using port
netstat -ano | findstr :21115
# Kill process
taskkill /PID <PID> /F
```

## Production Deployment

For production deployment:

1. **Use a process manager** (PM2, systemd, etc.)
2. **Set up reverse proxy** (Nginx, Apache)
3. **Configure SSL/TLS** certificates
4. **Set proper environment variables**
5. **Use production build** (already done by deploy scripts)

### Example with PM2

```bash
# Install PM2
npm install -g pm2

# Start backend
cd backend
source venv/bin/activate
pm2 start main.py --name text2image-backend --interpreter python

# Start frontend
cd ../frontend
pm2 start npm --name text2image-frontend -- start
```

## Environment Variables

### Frontend

- `NEXT_PUBLIC_API_URL` - Backend API URL (default: http://localhost:21115)

### Backend

- `TEXT2IMAGE_SERVER_URL` - Text-to-image server URL (default: http://localhost:8001)
- `REQUEST_TIMEOUT` - Request timeout in seconds (default: 300)
- `MAX_CONCURRENT_REQUESTS` - Max concurrent requests (default: 10)

## Architecture

```
┌─────────────────┐
│   Frontend      │  Port 3000
│   (Next.js)     │
└────────┬────────┘
         │
         │ HTTP
         ▼
┌─────────────────┐
│   Backend       │  Port 21115
│   (API Gateway) │
└────────┬────────┘
         │
         │ HTTP
         ▼
┌─────────────────┐
│ Text-to-Image   │  Port 8001
│ Server          │
└─────────────────┘
```

## Support

For issues or questions:
1. Check the logs in the terminal
2. Verify all services are running
3. Check network connectivity
4. Review configuration files

