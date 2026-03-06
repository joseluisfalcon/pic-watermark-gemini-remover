"""
utils.py - Utility functions for watermark detection and masking.

The watermark is assumed to be a white/bright diamond-like symbol
located in the bottom-right corner of the image.
"""

import cv2
import numpy as np
from PIL import Image


def load_image(path: str):
    """
    Load an image preserving full quality.
    Returns a BGR numpy array (OpenCV format) and the original PIL image.
    """
    pil_img = Image.open(path)
    # Convert to RGB to handle RGBA / palette images
    pil_rgb = pil_img.convert("RGB")
    # OpenCV expects BGR
    bgr = cv2.cvtColor(np.array(pil_rgb), cv2.COLOR_RGB2BGR)
    return bgr, pil_img


def save_image(bgr_img: np.ndarray, output_path: str, original_pil: Image.Image = None):
    """
    Save a BGR numpy array as a PNG file, preserving original DPI metadata.
    """
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    out_pil = Image.fromarray(rgb)

    # Preserve DPI if original has it
    dpi = None
    if original_pil is not None and hasattr(original_pil, 'info'):
        dpi = original_pil.info.get('dpi')

    save_kwargs = {}
    if dpi:
        save_kwargs['dpi'] = dpi

    # Always save as high-quality PNG
    out_pil.save(output_path, format='PNG', **save_kwargs)
    print(f"[OK] Saved output to: {output_path}")


def generate_watermark_mask(
    image: np.ndarray,
    region_fraction: float = 0.12,
    brightness_threshold: int = 200,
    dilation_radius: int = 8,
    use_relative: bool = True,
    relative_boost: float = 1.3,
    debug: bool = False
) -> np.ndarray:
    """
    Automatically detect a white/semi-transparent watermark in the bottom-right corner.

    Uses a combined strategy:
    1. Absolute brightness threshold (good for opaque white logos).
    2. Relative brightness: pixels brighter than `relative_boost` × local mean
       (good for semi-transparent logos over dark or mid-tone backgrounds).
    3. Whiteness check: pixels where R≈G≈B and all channels are above a minimum.

    Parameters
    ----------
    image : np.ndarray
        BGR input image.
    region_fraction : float
        Fraction of the image dimensions to search in the corner (default 12%).
    brightness_threshold : int
        Absolute brightness threshold (0-255, default 200).
        Lower values catch dimmer / more transparent watermarks.
    dilation_radius : int
        Extra pixels to dilate the mask around detected regions.
    use_relative : bool
        If True, also detect pixels that are significantly brighter than
        the local corner average (helps with semi-transparent marks).
    relative_boost : float
        Factor above local mean to consider a pixel part of the watermark
        (default 1.3 = 30% brighter than the corner average).
    debug : bool
        If True, also returns the corner-only mask for inspection.

    Returns
    -------
    np.ndarray
        Binary mask (uint8, 0=keep, 255=inpaint) the same size as the image.
    """
    h, w = image.shape[:2]

    # --- Define the search region (bottom-right corner) -----------------------
    rh = int(h * region_fraction)
    rw = int(w * region_fraction)
    corner_y0 = h - rh
    corner_x0 = w - rw

    corner_bgr = image[corner_y0:h, corner_x0:w]

    # --- Convert corner to grayscale ------------------------------------------
    corner_gray = cv2.cvtColor(corner_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # --- Strategy 1: absolute brightness threshold ----------------------------
    _, mask_abs = cv2.threshold(
        corner_gray.astype(np.uint8), brightness_threshold, 255, cv2.THRESH_BINARY
    )

    # --- Strategy 2: relative brightness (pixel vs local mean) ----------------
    mask_rel = np.zeros_like(mask_abs)
    if use_relative:
        local_mean = float(np.mean(corner_gray))
        if local_mean > 0:
            relative_mask = (corner_gray > local_mean * relative_boost).astype(np.uint8) * 255
            mask_rel = relative_mask

    # --- Strategy 3: whiteness (R≈G≈B and moderately bright) -----------------
    b, g, r = cv2.split(corner_bgr.astype(np.float32))
    # Max channel spread: how "grey/white" the pixel is (low spread = white-ish)
    channel_min = np.minimum(np.minimum(r, g), b)
    channel_max = np.maximum(np.maximum(r, g), b)
    spread = channel_max - channel_min
    # Bright AND neutral color (spread < 30 to allow slight tinting)
    whiteness_min = 140  # minimum brightness for all channels
    mask_white = (
        (spread < 30) &
        (channel_min > whiteness_min)
    ).astype(np.uint8) * 255

    # --- Combine all strategies -----------------------------------------------
    corner_mask = cv2.bitwise_or(mask_abs, mask_rel)
    corner_mask = cv2.bitwise_or(corner_mask, mask_white)

    # --- Morphological cleanup ------------------------------------------------
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    corner_mask = cv2.morphologyEx(corner_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # Remove tiny specks (keep only regions with enough contiguous pixels)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(corner_mask, connectivity=8)
    min_area = 20  # ignore blobs smaller than 20px
    filtered = np.zeros_like(corner_mask)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == lbl] = 255
    corner_mask = filtered

    # --- Dilate to cover anti-aliasing fringe around watermark ----------------
    if dilation_radius > 0:
        kernel_dilate = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * dilation_radius + 1, 2 * dilation_radius + 1)
        )
        corner_mask = cv2.dilate(corner_mask, kernel_dilate)

    # --- Place the corner mask back into a full-image mask --------------------
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[corner_y0:h, corner_x0:w] = corner_mask

    if debug:
        return full_mask, corner_mask  # type: ignore
    return full_mask


def generate_mask_from_bbox(
    image: np.ndarray,
    x: int, y: int, width: int, height: int,
    dilation_radius: int = 4
) -> np.ndarray:
    """
    Generate a mask from an explicit bounding box (x, y, width, height).
    Use this if auto-detection fails and you know where the watermark is.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x2 = min(x + width, w)
    y2 = min(y + height, h)
    mask[y:y2, x:x2] = 255

    if dilation_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * dilation_radius + 1, 2 * dilation_radius + 1)
        )
        mask = cv2.dilate(mask, kernel)

    return mask
