import platform
from pathlib import Path
import torch
from diffusers import ZImagePipeline

# ============================================================================
# System Detection
# ============================================================================

IS_WINDOWS = platform.system() == "Windows"

# ============================================================================
# CUDA Optimizations (apply at import time for best performance)
# ============================================================================

# Enable cuDNN benchmark for better performance on fixed input sizes
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed

# Enable TensorFloat-32 for faster computation on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ============================================================================
# 1. Load the pipeline with speed optimizations
# ============================================================================

# Try to use local model if available
MODEL_NAME = "Tongyi-MAI/Z-Image-Turbo"
local_model_path = None

# Check current directory for model folders
current_dir = Path(".")
potential_folders = [
    "models--Tongyi-MAI--Z-Image-Turbo",
    "models--Tongyi-AI--Z-Image-Turbo",
    "Tongyi-MAI-Z-Image-Turbo",
]

for folder_name in potential_folders:
    folder_path = current_dir / folder_name
    if folder_path.exists() and folder_path.is_dir():
        # Check for config.json or model files
        if (folder_path / "config.json").exists() or list(folder_path.glob("*.safetensors")):
            # Check snapshots subdirectory (HuggingFace cache format)
            snapshots_dir = folder_path / "snapshots"
            if snapshots_dir.exists():
                for snapshot in snapshots_dir.iterdir():
                    if snapshot.is_dir() and (snapshot / "config.json").exists():
                        local_model_path = str(snapshot.resolve())
                        print(f"✓ Using local model from: {local_model_path}")
                        break
                if local_model_path:
                    break
            else:
                local_model_path = str(folder_path.resolve())
                print(f"✓ Using local model from: {local_model_path}")
                break

if not local_model_path:
    print(f"Local model not found, will download from HuggingFace: {MODEL_NAME}")

# Use bfloat16 for optimal performance on supported GPUs
pipe = ZImagePipeline.from_pretrained(
    local_model_path if local_model_path else MODEL_NAME,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,  # Faster loading if you have enough RAM
    local_files_only=(local_model_path is not None),
)
pipe.to("cuda")

# ============================================================================
# Speed Optimizations (enable all for maximum performance)
# ============================================================================

# Enable Flash Attention 3 (fastest) or Flash Attention 2 (fallback)
try:
    pipe.transformer.set_attention_backend("_flash_3")
    print("✓ Flash Attention 3 enabled (fastest)")
except:
    try:
        pipe.transformer.set_attention_backend("flash")
        print("✓ Flash Attention 2 enabled")
    except Exception as e:
        print(f"⚠ Flash Attention not available: {e}")

# Enable VAE tiling for efficient large image processing (better than slicing)
try:
    pipe.enable_vae_tiling()
    print("✓ VAE tiling enabled")
except Exception as e:
    print(f"⚠ VAE tiling not available: {e}")

# Modern PyTorch 2.0+ compilation (MUCH faster than transformer.compile())
# Skip compilation on Windows (compatibility issues)
if IS_WINDOWS:
    print("⚠ Skipping model compilation (not supported on Windows)")
    print("  Performance will still be excellent with Flash Attention enabled")
elif hasattr(torch, 'compile'):
    print("Compiling transformer with torch.compile (first run will be slower)...")
    pipe.transformer = torch.compile(
        pipe.transformer,
        mode="reduce-overhead",  # Best balance of speed and compilation time
        fullgraph=False,
        dynamic=False,
    )
    print("✓ torch.compile() enabled - EXPECT SIGNIFICANT SPEEDUP!")
else:
    # Fallback to legacy compilation
    try:
        pipe.transformer.compile()
        print("✓ Legacy compilation enabled")
    except Exception as e:
        print(f"⚠ Compilation not available: {e}")

# DO NOT enable CPU offloading, VAE slicing, or attention slicing - they slow things down
# Only enable if you run out of VRAM

prompt = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."

# 2. Generate Image
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=9,  # This actually results in 8 DiT forwards
    guidance_scale=0.0,     # Guidance should be 0 for the Turbo models
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("example.png")
