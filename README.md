# Text-to-Image Web Application

A production-ready text-to-image generation application built with FastAPI (backend) and Next.js 14+ (frontend). The backend uses the **Tongyi-MAI/Z-Image-Turbo** model for high-performance image generation.

## Architecture

- **Backend**: FastAPI running on GPU machine with model loaded once at startup
- **Frontend**: Next.js 14+ with App Router, TypeScript, Tailwind CSS, and shadcn/ui
- **Model**: Tongyi-MAI/Z-Image-Turbo (bfloat16, CUDA)
- **Connection**: Reverse SSH tunnel from GPU machine to public server (frontend)

**Note**: The backend and frontend run on different machines. Use reverse SSH tunneling to connect them. See [REVERSE_SSH_SETUP.md](REVERSE_SSH_SETUP.md) for detailed setup instructions.

## Features

### Backend
- ✅ Single model load at startup (reused for all requests)
- ✅ Concurrent request limiting (max 2 simultaneous generations)
- ✅ CUDA cache clearing after each generation
- ✅ Health check endpoint
- ✅ CORS support
- ✅ Optional image saving to disk
- ✅ Production-ready with Gunicorn + Uvicorn

### Frontend
- ✅ Beautiful, modern UI with dark/light mode
- ✅ Large prompt textarea with character counter
- ✅ Collapsible negative prompt section
- ✅ Resolution presets (1024×1024, 768×1152, 1152×768, etc.)
- ✅ Custom width/height inputs
- ✅ Seed input with random seed generator
- ✅ Steps slider (6-12, default 9)
- ✅ Real-time image display from base64
- ✅ Gallery of last 12 generated images
- ✅ Copy prompt/seed, download image buttons
- ✅ Toast notifications
- ✅ Fully mobile-responsive
- ✅ API health monitoring

## Prerequisites

### Backend (GPU Machine)
- Python 3.11+
- CUDA-capable GPU with sufficient VRAM (recommended: 24GB+)
- NVIDIA drivers and CUDA toolkit
- PyTorch with CUDA support

### Frontend
- Node.js 18+ and npm

## Quick Start

### Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the server:**
   ```bash
   # Development
   python main.py
   
   # Or with uvicorn directly
   uvicorn main:app --host 0.0.0.0 --port 8000
   
   # Production (with Gunicorn)
   gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 --timeout 300
   ```

The backend will be available at `http://localhost:8000`

**Important**: The backend is wrapped as a pure API - the model loading happens in `main.py` via the lifespan context manager, not in `run.py`. The `run.py` file is just a server runner.

### Reverse SSH Tunnel Setup

Since the backend runs on a GPU machine (possibly behind a firewall) and the frontend runs on a public server, you need to set up a reverse SSH tunnel.

**Quick Setup:**

1. **On GPU machine (backend):**
   ```bash
   cd backend
   # Edit setup_reverse_ssh.sh with your public server details
   chmod +x setup_reverse_ssh.sh
   ./setup_reverse_ssh.sh
   ```

2. **On public server (frontend):**
   ```bash
   # Verify tunnel is working
   curl http://localhost:8000/health
   ```

See [REVERSE_SSH_SETUP.md](REVERSE_SSH_SETUP.md) for complete instructions.

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Create `.env.local` file:**
   ```bash
   cp .env.example .env.local
   ```

4. **Update `.env.local` with your backend URL:**
   ```env
   # If using reverse SSH tunnel (frontend on same server as tunnel endpoint):
   NEXT_PUBLIC_API_URL=http://localhost:8000
   
   # Or if using a public domain with Nginx reverse proxy:
   # NEXT_PUBLIC_API_URL=http://api.yourdomain.com
   ```

5. **Run development server:**
   ```bash
   npm run dev
   ```

The frontend will be available at `http://localhost:3000`

## Docker Deployment

### Backend with Docker

1. **Build and run with docker-compose:**
   ```bash
   docker-compose up -d
   ```

   Or build manually:
   ```bash
   cd backend
   docker build -t text2image-backend .
   docker run --gpus all -p 8000:8000 text2image-backend
   ```

**Note**: Docker deployment requires NVIDIA Container Toolkit for GPU access.

### Frontend Deployment

The frontend can be deployed to:
- **Vercel** (recommended for Next.js)
- **Netlify**
- **Any static hosting service**

#### Deploy to Vercel

1. Push your code to GitHub
2. Import project in Vercel
3. Set environment variable: `NEXT_PUBLIC_API_URL=https://your-backend-url.com`
4. Deploy

#### Deploy to Netlify

1. Build the project:
   ```bash
   cd frontend
   npm run build
   ```

2. Deploy the `.next` folder or connect to Git repository
3. Set environment variable: `NEXT_PUBLIC_API_URL=https://your-backend-url.com`

## API Documentation

### Endpoints

#### `GET /`
Root endpoint with API information.

#### `GET /health`
Health check endpoint. Returns:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "cuda_available": true,
  "cuda_device": "NVIDIA GeForce RTX 4090"
}
```

#### `POST /generate`
Generate an image from a text prompt.

**Request Body:**
```json
{
  "prompt": "A beautiful landscape with mountains",
  "negative_prompt": "blurry, low quality",
  "height": 1024,
  "width": 1024,
  "seed": -1,
  "num_inference_steps": 9,
  "guidance_scale": 0.0
}
```

**Response:**
```json
{
  "image_base64": "data:image/png;base64,...",
  "seed": 12345,
  "generation_time_ms": 842,
  "width": 1024,
  "height": 1024,
  "image_id": "uuid-here"
}
```

## Configuration

### Backend Environment Variables

Create a `.env` file in the `backend` directory:

```env
SAVE_IMAGES=false
# Set to true to save generated images to disk

# CORS Origins (comma-separated)
# CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
```

### Frontend Environment Variables

Create a `.env.local` file in the `frontend` directory:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Production Deployment

### Backend on GPU Server

1. **SSH into your GPU server**
2. **Clone the repository**
3. **Set up Python environment and install dependencies**
4. **Run with Gunicorn:**
   ```bash
   gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 --timeout 300
   ```

5. **Use a process manager (recommended):**
   - **systemd** (Linux)
   - **PM2** (Node.js process manager)
   - **Supervisor**

#### Example systemd Service

Create `/etc/systemd/system/text2image.service`:

```ini
[Unit]
Description=Text-to-Image API
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/Text2Image/backend
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 --timeout 300
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable text2image
sudo systemctl start text2image
```

### Nginx Reverse Proxy (Optional)

Example nginx configuration for backend:

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
    }
}
```

## Performance Optimization

### Backend
- Model is loaded once at startup (shared across requests)
- Concurrent generations limited to 2 (prevents OOM)
- CUDA cache cleared after each generation
- Uses `torch.inference_mode()` and `torch.no_grad()` for efficiency

### Optional Optimizations
Uncomment in `backend/main.py`:
```python
# pipe.transformer.compile()  # Faster after first run
# pipe.enable_model_cpu_offload()  # For <24GB VRAM
```

## Troubleshooting

### Backend Issues

**CUDA out of memory:**
- Reduce `MAX_CONCURRENT_GENERATIONS` in `main.py`
- Enable CPU offloading: `pipe.enable_model_cpu_offload()`
- Reduce image resolution

**Model loading fails:**
- Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
- Verify GPU drivers are installed
- Check available VRAM

### Frontend Issues

**API connection errors:**
- Verify `NEXT_PUBLIC_API_URL` is correct
- Check CORS settings on backend
- Ensure backend is running and accessible

**Build errors:**
- Clear `.next` folder: `rm -rf .next`
- Reinstall dependencies: `rm -rf node_modules && npm install`

## License

This project is provided as-is for educational and development purposes.

## Acknowledgments

- Model: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- Built with [FastAPI](https://fastapi.tiangolo.com/) and [Next.js](https://nextjs.org/)

