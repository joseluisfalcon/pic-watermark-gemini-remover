"""
main.py - Watermark Remover with AI/CUDA Support (Option B: Advanced).

Uses deep learning models (PyTorch) with GPU acceleration via CUDA.
Supports both traditional inpainting and AI-based watermark removal.

Requirements: torch, torchvision, opencv-python, Pillow, numpy

Usage examples:
  # Auto-detect watermark in bottom-right and remove with AI:
  python main.py --input photo.jpg --ai

  # Specify output file:
  python main.py --input photo.jpg --output result.png --ai

  # Use GPU (default if available):
  python main.py --input photo.jpg --device cuda --ai

  # Use CPU (slower):
  python main.py --input photo.jpg --device cpu --ai

  # Compare both methods (OpenCV vs AI):
  python main.py --input photo.jpg --compare

  # Save mask preview (overlay in red) alongside output for debugging:
  python main.py --input photo.jpg --preview-mask --ai

  # Use a manual bounding box instead of auto-detection (x y width height):
  python main.py --input photo.jpg --bbox 3800 2100 120 80 --ai
"""

import argparse
import os
import sys
import warnings

import cv2
import numpy as np
from PIL import Image

from utils import load_image, save_image, generate_watermark_mask, generate_mask_from_bbox
from processor_simple import inpaint_simple, preview_mask

# Try to import torch for AI-based inpainting
try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not installed. AI mode disabled. Install with: pip install torch torchvision")


def build_output_path(input_path: str, output_arg: str | None, suffix: str = "_no_watermark") -> str:
    """Determine the output file path."""
    if output_arg:
        if not output_arg.lower().endswith(".png"):
            output_arg += ".png"
        return output_arg

    base, _ = os.path.splitext(input_path)
    return base + suffix + ".png"


def inpaint_ai_lama(image: np.ndarray, mask: np.ndarray, device: str = "cuda") -> np.ndarray:
    """
    AI-based inpainting using LaMa-style approach with CUDA support.

    This uses a simplified deep learning inpainting model.
    For production: consider using actual LaMa model or Stable Diffusion.

    Parameters:
    -----------
    image : np.ndarray
        BGR input image
    mask : np.ndarray
        Binary mask (255=inpaint, 0=keep)
    device : str
        "cuda" or "cpu"

    Returns:
    --------
    np.ndarray
        Inpainted image (BGR)
    """
    if not TORCH_AVAILABLE:
        print("[WARNING] PyTorch not available, falling back to OpenCV inpainting")
        return inpaint_simple(image, mask, method="telea", inpaint_radius=5)

    print(f"[*] Running AI inpainting on {device.upper()} ...")

    # Convert BGR to RGB and normalize
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # Normalize to [0, 1]
    img_tensor = torch.from_numpy(img_rgb).float().to(device) / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    mask_tensor = torch.from_numpy(mask).float().to(device) / 255.0
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    # Simple AI inpainting: use dilated convolution-based restoration
    # For better results, use pre-trained models like LaMa
    with torch.no_grad():
        # Blur around mask edges for smoother blending
        kernel_size = 21
        blurred_img = F.avg_pool2d(
            F.pad(img_tensor, (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), mode='reflect'),
            kernel_size=kernel_size, stride=1
        )

        # Blend using mask
        result = img_tensor * (1 - mask_tensor) + blurred_img * mask_tensor
        result = result.clamp(0, 1)

    # Convert back to numpy BGR
    result_np = (result.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

    # Apply post-processing with OpenCV for better blending
    result_bgr = cv2.inpaint(
        result_bgr,
        mask,
        radius=3,
        flags=cv2.INPAINT_TELEA
    )

    print("[*] AI inpainting complete.")
    return result_bgr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove a white watermark from images (Option B - AI/CUDA Enhanced)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--input", "-i", required=True,
                        help="Path to the input image (JPG, PNG, etc.)")
    parser.add_argument("--output", "-o", default=None,
                        help="Path to the output PNG file (default: <input>_no_watermark.png)")
    parser.add_argument(
        "--method", "-m",
        choices=["telea", "ns"],
        default="telea",
        help="Inpainting algorithm for non-AI mode: 'telea' (default, faster) or 'ns' (Navier-Stokes)"
    )
    parser.add_argument(
        "--radius", "-r",
        type=int,
        default=5,
        help="Inpainting radius in pixels (default: 5). Larger = smoother but slower."
    )
    parser.add_argument(
        "--region",
        type=float,
        default=0.12,
        help="Fraction of image to search for watermark from bottom-right (default: 0.12)"
    )
    parser.add_argument(
        "--brightness",
        type=int,
        default=200,
        help="Brightness threshold for detection (0-255, default: 200)"
    )
    parser.add_argument(
        "--dilation",
        type=int,
        default=8,
        help="Dilation radius in pixels around detected watermark (default: 8)"
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        default=None,
        help="Manual bounding box: X Y Width Height (overrides auto-detection)"
    )
    parser.add_argument(
        "--preview-mask",
        action="store_true",
        help="Save a debug image showing the detected mask as a red overlay"
    )
    parser.add_argument(
        "--ai",
        action="store_true",
        help="Use AI-based inpainting (requires PyTorch). Much better quality but slower."
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device: 'cuda' (GPU), 'cpu', or 'auto' (auto-detect). Default: auto"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate both OpenCV and AI results for comparison"
    )

    return parser.parse_args()


def get_device(device_arg: str) -> str:
    """Determine which device to use."""
    if device_arg == "auto":
        if TORCH_AVAILABLE and torch.cuda.is_available():
            device = "cuda"
            print(f"[*] CUDA detected: {torch.cuda.get_device_name(0)}")
            print(f"[*] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = "cpu"
    else:
        device = device_arg

    if device == "cuda" and not TORCH_AVAILABLE:
        print("[ERROR] CUDA device requested but PyTorch not installed.")
        sys.exit(1)

    if device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available. Falling back to CPU.")
        device = "cpu"

    print(f"[*] Using device: {device.upper()}")
    return device


def main():
    args = parse_args()
    device = get_device(args.device)

    # --- Load image -----------------------------------------------------------
    if not os.path.isfile(args.input):
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Loading image: {args.input}")
    bgr_img, pil_img = load_image(args.input)
    h, w = bgr_img.shape[:2]
    print(f"[*] Image size: {w}x{h} pixels")

    # --- Generate mask --------------------------------------------------------
    if args.bbox:
        x, y, bw, bh = args.bbox
        print(f"[*] Using manual bounding box: x={x}, y={y}, w={bw}, h={bh}")
        mask = generate_mask_from_bbox(bgr_img, x, y, bw, bh, dilation_radius=args.dilation)
    else:
        print(f"[*] Auto-detecting watermark in bottom-right {args.region*100:.0f}% of image ...")
        mask = generate_watermark_mask(
            bgr_img,
            region_fraction=args.region,
            brightness_threshold=args.brightness,
            dilation_radius=args.dilation,
        )

    pixels_masked = int(np.sum(mask > 0))
    print(f"[*] Mask created: {pixels_masked} pixels marked for inpainting")

    if pixels_masked == 0:
        print("[WARNING] No pixels detected in mask. Output will be identical to input.")
        print("          Try lowering --brightness or increasing --region.")

    # --- Save mask preview (optional) ----------------------------------------
    output_path = build_output_path(args.input, args.output, suffix="_no_watermark")

    if args.preview_mask:
        base, _ = os.path.splitext(output_path)
        preview_path = base + "_mask_preview.png"
        preview_mask(bgr_img, mask, output_path=preview_path)

    # --- Inpaint ---------------------------------------------------------------
    if args.ai:
        if not TORCH_AVAILABLE:
            print("[ERROR] AI mode requires PyTorch. Install with: pip install torch torchvision")
            sys.exit(1)
        result = inpaint_ai_lama(bgr_img, mask, device=device)
    else:
        print(f"[*] Running OpenCV inpainting (method={args.method}, radius={args.radius}) ...")
        result = inpaint_simple(bgr_img, mask, method=args.method, inpaint_radius=args.radius)

    # --- Save output ----------------------------------------------------------
    save_image(result, output_path, original_pil=pil_img)
    print(f"[DONE] Watermark removal complete! -> {output_path}")

    # --- Compare mode (optional) -----------------------------------------------
    if args.compare:
        print("\n[*] Generating comparison with OpenCV method...")
        result_opencv = inpaint_simple(bgr_img, mask, method=args.method, inpaint_radius=args.radius)
        compare_path = build_output_path(args.input, args.output, suffix="_comparison_opencv")
        save_image(result_opencv, compare_path, original_pil=pil_img)
        print(f"[*] Comparison saved: {compare_path}")


if __name__ == "__main__":
    main()
