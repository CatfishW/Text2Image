# API Gateway Setup Guide

This backend now acts as an **API Gateway** that forwards requests to a remote text-to-image server instead of loading the model locally.

## Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│   Frontend      │ ──────> │  API Gateway     │ ──────> │ Text-to-Image   │
│   (Next.js)     │         │  (backend/main.py)│         │ Server          │
│   Port 3000     │         │  Port 8000       │         │ (text2image_    │
│                 │         │                  │         │  server.py)      │
│                 │         │  - No GPU        │         │  Port 8001      │
│                 │         │  - No Model      │         │                 │
│                 │         │  - Proxy Layer  │         │  - GPU Required │
│                 │         │                  │         │  - Model Loaded  │
└─────────────────┘         └──────────────────┘         └─────────────────┘
```

## Benefits

- ✅ **No GPU required** on gateway machine
- ✅ **No model loading** - saves memory and startup time
- ✅ **Easy scaling** - run multiple gateway instances
- ✅ **Separation of concerns** - gateway handles routing, server handles generation
- ✅ **Frontend unchanged** - same API interface

## Setup Steps

### 1. Start Text-to-Image Server

On the GPU machine, start the text-to-image server:

```bash
# On GPU machine
python text2image_server.py --port 8001
```

Or use the default port 8000 if you prefer.

### 2. Configure API Gateway

Create `.env` file in `backend/` directory:

```env
TEXT2IMAGE_SERVER_URL=http://localhost:8001
REQUEST_TIMEOUT=300
MAX_CONCURRENT_REQUESTS=10
```

**For remote server:**
```env
TEXT2IMAGE_SERVER_URL=http://192.168.1.100:8001
# Or
TEXT2IMAGE_SERVER_URL=http://gpu-server.example.com:8001
```

### 3. Install Gateway Dependencies

```bash
cd backend
pip install -r requirements.txt
```

**Note:** The gateway no longer needs:
- PyTorch
- diffusers
- transformers
- CUDA

Only FastAPI, httpx, and related web dependencies are needed.

### 4. Start API Gateway

```bash
# Development
python main.py

# Production
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 --timeout 300
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|---------|---------|-------------|
| `TEXT2IMAGE_SERVER_URL` | `http://localhost:8001` | URL of text-to-image server |
| `REQUEST_TIMEOUT` | `300` | Request timeout in seconds |
| `MAX_CONCURRENT_REQUESTS` | `10` | Max concurrent requests to upstream |

### Port Configuration

**Option 1: Different ports (recommended)**
- Gateway: Port 8000
- Text-to-Image Server: Port 8001

**Option 2: Same machine, different ports**
- Gateway: Port 8000
- Text-to-Image Server: Port 8001

**Option 3: Remote server**
- Gateway: Port 8000 (on frontend server)
- Text-to-Image Server: Port 8001 (on GPU machine, via reverse SSH)

## Testing

### 1. Check Gateway Health

```bash
curl http://localhost:8000/health
```

Should return:
```json
{
  "status": "healthy",
  "gateway": "running",
  "text2image_server": {
    "status": "healthy",
    "model_loaded": true,
    "cuda_available": true
  }
}
```

### 2. Test Generation

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful landscape",
    "height": 1024,
    "width": 1024
  }'
```

### 3. Check Metrics

```bash
curl http://localhost:8000/metrics
```

## Troubleshooting

### Gateway can't connect to server

**Error:** `Cannot connect to text-to-image server`

**Solutions:**
1. Verify text-to-image server is running:
   ```bash
   curl http://localhost:8001/health
   ```

2. Check `TEXT2IMAGE_SERVER_URL` in `.env` file

3. Check firewall/network connectivity

4. For remote server, ensure reverse SSH tunnel is set up (if needed)

### Timeout errors

**Error:** `Request to text-to-image server timed out`

**Solutions:**
1. Increase `REQUEST_TIMEOUT` in `.env`
2. Check if text-to-image server is overloaded
3. Check network latency

### Gateway returns 503

**Error:** `Text-to-image server unavailable`

**Solutions:**
1. Check text-to-image server health: `curl http://localhost:8001/health`
2. Check if server queue is full
3. Check server logs

## Production Deployment

### Gateway (Multiple Instances)

```bash
# Use multiple workers for load balancing
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### Text-to-Image Server (Single Instance)

```bash
# Single worker (GPU sharing)
python text2image_server.py --port 8001
```

### Load Balancer Setup

```
                    ┌─────────────┐
                    │   Nginx     │
                    │ Load Balancer│
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼────┐       ┌────▼────┐       ┌────▼────┐
   │ Gateway │       │ Gateway │       │ Gateway │
   │   :8000 │       │   :8000 │       │   :8000 │
   └────┬────┘       └────┬────┘       └────┬────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                    ┌──────▼──────┐
                    │ Text-to-    │
                    │ Image Server│
                    │   :8001     │
                    └─────────────┘
```

## Migration from Local Model

If you were previously using the backend with local model loading:

1. ✅ **No frontend changes needed** - API interface is the same
2. ✅ **Update dependencies** - Remove PyTorch/diffusers, add httpx
3. ✅ **Configure server URL** - Set `TEXT2IMAGE_SERVER_URL` in `.env`
4. ✅ **Start text-to-image server** - Run `text2image_server.py` separately

The gateway maintains the same API contract, so your frontend will work without modifications!

