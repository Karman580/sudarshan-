# ==============================================================================
# utils.py — Shared Utilities for Aadhaar KYC Pipeline
# ==============================================================================
# Provides:
#   - Safe image loading & resizing
#   - Aadhaar number masking
#   - Verhoeff checksum algorithm (used by UIDAI for Aadhaar numbers)
# ==============================================================================

import cv2
import numpy as np
import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger("aadhaar_kyc")


# ==============================================================================
# IMAGE LOADING
# ==============================================================================

def load_image(path: str) -> Optional[np.ndarray]:
    """
    Safely load an image from disk as a BGR numpy array.

    Handles unicode paths via numpy intermediary.
    Returns None on failure.
    """
    if not path or not os.path.isfile(path):
        logger.warning("Image path invalid or not found: %s", path)
        return None

    try:
        # Use numpy to handle unicode paths on all platforms
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            logger.warning("cv2.imdecode returned empty image for: %s", path)
            return None
        return img
    except Exception as e:
        logger.error("Failed to load image %s: %s", path, e)
        return None


def resize_for_processing(
    image: np.ndarray,
    max_dim: int = 1280
) -> np.ndarray:
    """
    Resize image so that its longest dimension is at most max_dim.

    Preserves aspect ratio. Returns original if already within limit.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image

    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ==============================================================================
# AADHAAR NUMBER MASKING
# ==============================================================================

def mask_aadhaar_number(number: str) -> str:
    """
    Mask an Aadhaar number for logging/display: show first 4 digits only.

    Example: "1234 5678 9012" → "1234 XXXX XXXX"
    """
    clean = number.replace(" ", "").replace("-", "")
    if len(clean) < 4:
        return "XXXX XXXX XXXX"
    return f"{clean[:4]} XXXX XXXX"


# ==============================================================================
# VERHOEFF CHECKSUM ALGORITHM
# ==============================================================================
# Aadhaar numbers use the Verhoeff algorithm for their check digit.
# Reference: https://en.wikipedia.org/wiki/Verhoeff_algorithm

# Multiplication table d
VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]

# Permutation table p
VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]

# Inverse table
VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_checksum(number: str) -> int:
    """
    Compute the Verhoeff checksum digit for a numeric string.

    Args:
        number: Numeric string (digits only).

    Returns:
        The check digit (0-9). A valid checksum returns 0.
    """
    c = 0
    digits = [int(d) for d in reversed(number)]
    for i, digit in enumerate(digits):
        c = VERHOEFF_D[c][VERHOEFF_P[i % 8][digit]]
    return c


def validate_verhoeff(number: str) -> bool:
    """
    Validate a number against its Verhoeff check digit.

    The number is valid if verhoeff_checksum returns 0.
    """
    try:
        clean = number.replace(" ", "").replace("-", "")
        if not clean.isdigit():
            return False
        return verhoeff_checksum(clean) == 0
    except (ValueError, IndexError):
        return False
