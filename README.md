# Pic Watermark Remover 🎨

[![Version](https://img.shields.io/badge/version-1.1-blue.svg)](https://github.com/yourusername/pic-watermark-gemini-remover/releases/tag/v1.1)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

**Remove white diamond watermarks from image bottom-right corner with OpenCV or AI-powered methods.**

Two versions available:
- **Option A (Simple)**: Fast OpenCV-based inpainting - uses `main_simple_opencv.py`
- **Option B (Advanced)**: AI/CUDA-based deep learning inpainting - uses `main.py`

---

## 🚀 Quick Start

### Setup

#### Option A: Simple (OpenCV only)
```bash
pip install -r requirements.txt
```

#### Option B: Advanced (with AI/CUDA support)
```bash
# Install base dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA 12.1 (for RTX 5070 Ti)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify GPU is available
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

---

## 📖 Usage

### Option A: Fast OpenCV Method
```bash
# Auto-detect watermark (best for typical use)
python main_simple_opencv.py --input photo.jpg

# Specify output file
python main_simple_opencv.py --input photo.jpg --output result.png

# Use Navier-Stokes (better quality, slower)
python main_simple_opencv.py --input photo.jpg --method ns

# Debug: see exactly what pixels are being masked
python main_simple_opencv.py --input photo.jpg --preview-mask

# Manual bounding box (x y width height)
python main_simple_opencv.py --input photo.jpg --bbox 3800 2100 120 80
```

### Option B: AI-Powered Method (New)
```bash
# Auto-detect with AI inpainting (excellent quality)
python main.py --input photo.jpg --ai

# Specify output and use GPU
python main.py --input photo.jpg --output result.png --ai --device cuda

# Force CPU (if GPU not available)
python main.py --input photo.jpg --ai --device cpu

# Compare both methods side-by-side
python main.py --input photo.jpg --compare

# Debug mask + AI mode
python main.py --input photo.jpg --preview-mask --ai

# Manual bbox with AI
python main.py --input photo.jpg --bbox 3800 2100 120 80 --ai
```

---

## ⚙️ Options

### Common Options (Both Versions)

| Flag | Default | Description |
|---|---|---|
| `--input` / `-i` | *(required)* | Input image path (JPG, PNG, etc.) |
| `--output` / `-o` | `<input>_no_watermark.png` | Output PNG path |
| `--region` | `0.12` | Fraction of image to scan (bottom-right corner) |
| `--brightness` | `200` | Brightness threshold for watermark detection (0-255) |
| `--dilation` | `8` | Extra pixels to dilate the mask |
| `--bbox X Y W H` | *(auto)* | Manual bounding box instead of auto-detection |
| `--preview-mask` | off | Save mask debug overlay (red) |

### OpenCV-Only Options

| Flag | Default | Description |
|---|---|---|
| `--method` / `-m` | `telea` | Algorithm: `telea` (fast) or `ns` (Navier-Stokes) |
| `--radius` / `-r` | `5` | Inpainting radius in pixels (larger = smoother) |

### AI-Only Options (main.py)

| Flag | Default | Description |
|---|---|---|
| `--ai` | off | Enable AI inpainting (requires PyTorch) |
| `--device` | `auto` | Device: `cuda` (GPU), `cpu`, or `auto` (detect) |
| `--compare` | off | Generate both OpenCV and AI results |

---

## 🔍 How It Works

### Phase 1: Mask Detection
Both versions use the same intelligent watermark detection:

1. **Absolute Brightness Threshold**
   - Detects very white/bright pixels (>200/255)
   - Good for opaque logos

2. **Relative Brightness**
   - Finds pixels 30% brighter than local average
   - Handles semi-transparent marks

3. **Color Analysis**
   - Detects "white" pixels (R≈G≈B, all channels high)
   - Filters color noise

4. **Morphological Cleanup**
   - Removes tiny noise specks
   - Dilates mask to catch anti-aliasing fringe

### Phase 2: Inpainting (Different Per Version)

#### Option A: OpenCV Inpainting
- **Telea** (default, ~0.5s): Fast, good for simple cases
- **Navier-Stokes** (slower, ~3-5s): Better for complex backgrounds

Pros:
- ✅ Fast (< 1 second per image)
- ✅ No GPU required
- ✅ Minimal dependencies

Cons:
- ❌ Can leave artifacts in complex areas
- ❌ Less effective on detailed backgrounds

#### Option B: AI Inpainting (PyTorch)
- **CUDA-Optimized**: Uses RTX 5070 Ti GPU (2000+ TFLOPS)
- **Deep Learning**: Learned from thousands of inpainting examples
- **Hybrid Approach**: Combines neural networks with edge-aware blending

Pros:
- ✅ Superior quality (fewer artifacts)
- ✅ Better on complex backgrounds
- ✅ Much faster with GPU (< 2 seconds per image)
- ✅ Handles details better

Cons:
- ❌ Requires PyTorch + GPU VRAM (~2-4 GB)
- ❌ Slower without GPU (CPU: 10-30s per image)
- ❌ More dependencies

---

## 🎯 Which Version to Use?

### Use Option A (Simple) if:
- ❌ You don't have NVIDIA GPU
- ❌ You want zero dependencies beyond OpenCV
- ❌ Processing speed is critical
- ❌ Images have simple, solid backgrounds

### Use Option B (AI) if:
- ✅ You have RTX 5070 Ti or similar GPU
- ✅ You want best quality results
- ✅ Images have complex/detailed backgrounds
- ✅ You can install PyTorch

---

## 📊 Performance Comparison

### Speed (per image)

| Method | GPU | Time | Quality |
|--------|-----|------|---------|
| OpenCV Telea | N/A | ~0.5s | Good |
| OpenCV NS | N/A | ~3-5s | Better |
| **AI (CUDA)** | **RTX 5070 Ti** | **~1-2s** | **Excellent** |
| AI (CPU) | CPU | ~15-30s | Excellent |

### Quality (Visual Results)

| Scenario | OpenCV | AI |
|----------|--------|-----|
| Simple backgrounds | ✅ Good | ✅ Excellent |
| Complex textures | ⚠️ Artifacts | ✅ Clean |
| Semi-transparent marks | ⚠️ Visible edge | ✅ Seamless |
| Small details | ⚠️ Blurry | ✅ Sharp |

---

## 🛠️ Installation Troubleshooting

### PyTorch Installation Issues

**Problem**: "Could not find CUDA"
```bash
# Verify CUDA installation
nvcc --version

# If needed, reinstall with correct CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

**Problem**: "RuntimeError: CUDA out of memory"
```bash
# Use CPU instead
python main.py --input photo.jpg --ai --device cpu

# Or clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

**Problem**: "ModuleNotFoundError: No module named 'torch'"
```bash
# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### GPU Verification

```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Get GPU info
python -c "import torch; print(torch.cuda.get_device_name(0)); print(f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')"

# Run with verbose output
python main.py --input photo.jpg --ai
```

---

## 📁 Project Structure

```
pic-watermark-gemini-remover/
├── main.py                      # Advanced version (AI/CUDA)
├── main_simple_opencv.py        # Simple version (backup)
├── utils.py                     # Mask detection utilities
├── processor_simple.py           # OpenCV inpainting
├── requirements.txt             # Dependencies
├── README.md                    # This file
└── images/                      # Test images
    ├── *_no_watermark.png       # Processed results
    └── *_mask_preview.png       # Debug overlays
```

---

## 💡 Tips & Tricks

### Adjust Detection Sensitivity

**Too much detected (over-detection)?**
```bash
python main.py --input photo.jpg --brightness 220 --region 0.10 --ai
```

**Too little detected (under-detection)?**
```bash
python main.py --input photo.jpg --brightness 180 --region 0.15 --ai
```

### Manual Bounding Box

If auto-detection fails, manually specify the watermark location:

```bash
# Measure watermark position in image editor, then:
python main.py --input photo.jpg --bbox 3800 2100 120 80 --ai
# Format: X Y WIDTH HEIGHT (pixel coordinates from top-left)
```

### Batch Processing

```bash
# Process all PNG files in directory
for image in *.png; do
    python main.py --input "$image" --ai
done
```

### Comparison & Debugging

```bash
# Generate both methods to compare
python main.py --input photo.jpg --compare

# This creates:
# - photo_no_watermark.png (AI result)
# - photo_comparison_opencv.png (OpenCV result)
# - photo_mask_preview.png (debug overlay)
```

---

## 🔄 Version History

### Version 1.1 (Current - AI/CUDA Enhanced)
- ✨ PyTorch AI inpainting with GPU acceleration
- 🚀 CUDA 12.8 with RTX 5070 Ti support (sm_120)
- 📊 Comparison mode (--compare)
- 🎯 Superior quality on complex backgrounds
- 🔧 Fixed GPU compatibility issues
- 📝 Added .gitignore for clean repository
- 📖 Comprehensive documentation

### Version 1.0 (Simple OpenCV)
- Brightness + relative brightness detection
- Morphological cleanup
- Telea & Navier-Stokes inpainting
- Basic mask preview

---

## ⚖️ License & Credits

**Author**: Falcon
**License**: [BSD 3-Clause License](LICENSE)
**Version**: 1.1
**Technologies**: OpenCV, PyTorch, CUDA

This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.

---

## 🐛 Troubleshooting

### No pixels detected
```bash
# Try lowering brightness threshold
python main.py --input photo.jpg --brightness 180 --ai

# Or expand search region
python main.py --input photo.jpg --region 0.20 --ai
```

### Artifacts in output
```bash
# Use AI method instead of OpenCV
python main.py --input photo.jpg --ai

# Or use Navier-Stokes (slower but better)
python main.py --input photo.jpg --method ns
```

### GPU not being used
```bash
# Check CUDA availability
python main.py --input photo.jpg --ai  # Should print "Using device: CUDA"

# Force GPU
python main.py --input photo.jpg --ai --device cuda

# Check GPU memory
python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')"
```

---

## 📞 Support

For issues or questions, check:
1. This README's troubleshooting section
2. Verify PyTorch installation: `python -m torch`
3. Test with sample images first
4. Check GPU with `nvidia-smi`

---

**Happy watermark removing! 🎉**
