"""
main.py - Entry point for the Watermark Remover tool (Option A: Simple).

Usage examples:
  # Auto-detect watermark in bottom-right and remove it:
  python main.py --input photo.jpg

  # Specify output file:
  python main.py --input photo.jpg --output result.png

  # Use Navier-Stokes method instead of Telea:
  python main.py --input photo.jpg --method ns

  # Increase search region to 20% of image dimensions (for large watermarks):
  python main.py --input photo.jpg --region 0.20

  # Save mask preview (overlay in red) alongside output for debugging:
  python main.py --input photo.jpg --preview-mask

  # Use a manual bounding box instead of auto-detection (x y width height):
  python main.py --input photo.jpg --bbox 3800 2100 120 80
"""

import argparse
import os
import sys

import cv2
import numpy as np

from utils import load_image, save_image, generate_watermark_mask, generate_mask_from_bbox
from processor_simple import inpaint_simple, preview_mask


def build_output_path(input_path: str, output_arg: str | None) -> str:
    """Determine the output file path."""
    if output_arg:
        if not output_arg.lower().endswith(".png"):
            output_arg += ".png"
        return output_arg

    base, _ = os.path.splitext(input_path)
    return base + "_no_watermark.png"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove a white watermark from the bottom-right corner of an image (Option A - Simple)",
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
        help="Inpainting algorithm: 'telea' (default, faster) or 'ns' (Navier-Stokes)"
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

    return parser.parse_args()


def main():
    args = parse_args()

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
    output_path = build_output_path(args.input, args.output)

    if args.preview_mask:
        base, _ = os.path.splitext(output_path)
        preview_path = base + "_mask_preview.png"
        preview_mask(bgr_img, mask, output_path=preview_path)

    # --- Inpaint --------------------------------------------------------------
    result = inpaint_simple(bgr_img, mask, method=args.method, inpaint_radius=args.radius)

    # --- Save output ----------------------------------------------------------
    save_image(result, output_path, original_pil=pil_img)
    print(f"[DONE] Watermark removal complete! -> {output_path}")


if __name__ == "__main__":
    main()
