# ==============================================================================
# perspective_correction.py — Card Perspective Rectification
# ==============================================================================
# Detects the largest quadrilateral in the image and applies a perspective
# transform to produce a flat, rectangular card image.
#
# Algorithm:
#   1. Edge detection (Canny)
#   2. Find contours → filter for quadrilaterals
#   3. Order corner points (TL, TR, BR, BL)
#   4. Warp perspective to axis-aligned rectangle
# ==============================================================================

import cv2
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger("aadhaar_kyc")


def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order four corner points as: top-left, top-right, bottom-right, bottom-left.

    Uses sum and difference heuristics:
        TL has smallest sum (x+y), BR has largest sum.
        TR has smallest diff (y-x), BL has largest diff.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()

    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    rect[1] = pts[np.argmin(d)]   # top-right
    rect[3] = pts[np.argmax(d)]   # bottom-left

    return rect


def _compute_output_size(rect: np.ndarray) -> Tuple[int, int]:
    """Compute output width and height from ordered corner points."""
    (tl, tr, br, bl) = rect

    # Width: max of top-edge and bottom-edge lengths
    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    width = int(max(w_top, w_bot))

    # Height: max of left-edge and right-edge lengths
    h_left = np.linalg.norm(bl - tl)
    h_right = np.linalg.norm(br - tr)
    height = int(max(h_left, h_right))

    return max(width, 1), max(height, 1)


def correct_perspective(image: np.ndarray) -> np.ndarray:
    """
    Detect the largest card-like quadrilateral and warp it to a flat rectangle.

    If no suitable quadrilateral is found, returns the original image unchanged.

    Args:
        image: BGR numpy array.

    Returns:
        Perspective-corrected BGR image, or original if correction failed.
    """
    h, w = image.shape[:2]
    min_area = h * w * 0.10  # Quad must cover at least 10% of image

    # --- Pre-processing ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)

    # Dilate edges to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.dilate(edged, kernel, iterations=1)

    # --- Find contours ---
    contours, _ = cv2.findContours(
        edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best_quad = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            if area > best_area:
                best_area = area
                best_quad = approx

    if best_quad is None:
        logger.debug("No quadrilateral contour found – skipping perspective correction.")
        return image

    # --- Order corners and warp ---
    pts = best_quad.reshape(4, 2).astype(np.float32)
    rect = _order_points(pts)
    out_w, out_h = _compute_output_size(rect)

    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (out_w, out_h))

    logger.info("Perspective correction applied (area=%.0f, output=%dx%d).", best_area, out_w, out_h)
    return warped


def run(image: np.ndarray, **ctx) -> dict:
    """
    Pipeline-compatible wrapper. Perspective correction is a preprocessing
    step so it always 'passes' — the corrected image is stored in ctx.
    """
    corrected = correct_perspective(image)
    return {
        "passed": True,
        "score": 1.0,
        "detail": "Perspective correction applied." if corrected is not image else "No correction needed.",
        "data": {"corrected_image": corrected},
    }
