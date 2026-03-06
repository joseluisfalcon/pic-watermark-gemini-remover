"""
processor_simple.py - Option A: Simple inpainting using OpenCV.

Uses the Telea or Navier-Stokes algorithm provided by OpenCV.
Fast, CPU-only, no models required.
"""

import cv2
import numpy as np

# Available inpainting methods
METHODS = {
    "telea": cv2.INPAINT_TELEA,   # Fast marching method (better for small areas)
    "ns": cv2.INPAINT_NS,          # Navier-Stokes (better for larger areas)
}


def inpaint_simple(
    image: np.ndarray,
    mask: np.ndarray,
    method: str = "telea",
    inpaint_radius: int = 5,
) -> np.ndarray:
    """
    Remove watermark from image using OpenCV inpainting.

    Parameters
    ----------
    image : np.ndarray
        BGR input image (full resolution).
    mask : np.ndarray
        Binary uint8 mask (0=preserve, 255=inpaint).
    method : str
        'telea' (default) or 'ns'. Telea is faster and usually better for logos.
    inpaint_radius : int
        Pixel neighborhood radius used by the inpainting algorithm.

    Returns
    -------
    np.ndarray
        BGR image with watermark removed.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(METHODS.keys())}")

    cv_method = METHODS[method]

    print(f"[*] Running OpenCV inpainting (method={method}, radius={inpaint_radius}) ...")

    # OpenCV inpaint works on 8-bit images.
    # For large 4K images this might take a few seconds.
    result = cv2.inpaint(image, mask, inpaintRadius=inpaint_radius, flags=cv_method)

    print("[*] Inpainting complete.")
    return result


def preview_mask(image: np.ndarray, mask: np.ndarray, output_path: str = None) -> np.ndarray:
    """
    Overlay the mask in red on the image for visual debugging.
    Useful to verify that the mask covers the watermark correctly.
    """
    overlay = image.copy()
    red_layer = np.zeros_like(overlay)
    red_layer[:, :] = (0, 0, 255)  # BGR red

    # Apply red where mask is set
    overlay[mask == 255] = cv2.addWeighted(
        overlay, 0.4, red_layer, 0.6, 0
    )[mask == 255]

    if output_path:
        cv2.imwrite(output_path, overlay)
        print(f"[OK] Mask preview saved to: {output_path}")

    return overlay
