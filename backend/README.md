# Backend API Gateway

FastAPI backend that acts as an API gateway, forwarding requests to a remote text-to-image server (text2image_server.py).

## Architecture

This backend does **NOT** load the deep learning model locally. Instead, it:
- Forwards requests to a remote text-to-image server
- Acts as a proxy/gateway layer
- Maintains the same API interface for the frontend
- Provides connection pooling and request management

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Note: This backend no longer requires PyTorch, diffusers, or CUDA dependencies.

2. **Configure the text-to-image server URL:**
   ```bash
   cp .env.example .env
   # Edit .env and set TEXT2IMAGE_SERVER_URL
   ```
   
   Example `.env`:
   ```env
   TEXT2IMAGE_SERVER_URL=http://localhost:8001
   REQUEST_TIMEOUT=300
   MAX_CONCURRENT_REQUESTS=10
   ```

3. **Make sure text2image_server.py is running** on the configured URL.

4. **Run the gateway:**
   ```bash
   python main.py
   # Or
   uvicorn main:app --host 0.0.0.0 --port 8000
   # Or for production
   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 --timeout 300
   ```

## API Endpoints

- `GET /` - API gateway information
- `GET /health` - Health check (also checks upstream server)
- `POST /generate` - Generate image (forwards to text-to-image server)
- `GET /metrics` - Get metrics from upstream server

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `TEXT2IMAGE_SERVER_URL` | `http://localhost:8001` | URL of the text-to-image server |
| `REQUEST_TIMEOUT` | `300` | Request timeout in seconds |
| `MAX_CONCURRENT_REQUESTS` | `10` | Max concurrent requests to upstream server |

## Architecture Diagram

```
Frontend (Next.js)
    ↓
Backend API Gateway (this service) - Port 8000
    ↓ HTTP Forward
Text-to-Image Server (text2image_server.py) - Port 8001
    ↓
GPU Machine with Model
```

## Benefits

- ✅ No GPU/VRAM requirements on gateway machine
- ✅ Can run multiple gateway instances for load balancing
- ✅ Separates concerns: gateway handles routing, server handles generation
- ✅ Easy to scale horizontally
- ✅ Frontend doesn't need to change

