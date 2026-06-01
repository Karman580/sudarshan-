# ==============================================================================
# color_validator.py — Aadhaar Tricolor Header Validation
# ==============================================================================
# Verifies HSV color distribution for the iconic saffron-white-green
# tricolor header band found on genuine Aadhaar cards.
#
# Checks the top 25% of the image for presence of all three colours
# in expected proportions.
#
# Weight in pipeline: 2%
# ==============================================================================

import cv2
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger("aadhaar_kyc")

# HSV ranges (OpenCV: H 0-180, S 0-255, V 0-255)
SAFFRON_LOW = np.array([5, 80, 150])
SAFFRON_HIGH = np.array([25, 255, 255])

GREEN_LOW = np.array([35, 60, 80])
GREEN_HIGH = np.array([85, 255, 220])

WHITE_LOW = np.array([0, 0, 200])
WHITE_HIGH = np.array([180, 40, 255])


def validate_color_profile(image: np.ndarray) -> Dict:
    """
    Analyse colour distribution for Aadhaar tricolor header.

    Checks the top 25% of the card for saffron, white, and green bands.
    Also checks entire card for sufficient white body area.

    Returns:
        Dict with passed, score, detail.
    """
    if len(image.shape) < 3:
        return {
            "passed": None,
            "score": 0.5,
            "detail": "Grayscale image — colour check skipped.",
        }

    h, w = image.shape[:2]
    header_region = image[0:int(h * 0.25), :]
    hsv_header = cv2.cvtColor(header_region, cv2.COLOR_BGR2HSV)
    hsv_full = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    header_pixels = hsv_header.shape[0] * hsv_header.shape[1]
    total_pixels = hsv_full.shape[0] * hsv_full.shape[1]

    if header_pixels == 0 or total_pixels == 0:
        return {"passed": False, "score": 0.0, "detail": "Empty image."}

    # Header colour ratios
    saffron_mask = cv2.inRange(hsv_header, SAFFRON_LOW, SAFFRON_HIGH)
    saffron_ratio = cv2.countNonZero(saffron_mask) / header_pixels

    green_mask = cv2.inRange(hsv_header, GREEN_LOW, GREEN_HIGH)
    green_ratio = cv2.countNonZero(green_mask) / header_pixels

    white_header_mask = cv2.inRange(hsv_header, WHITE_LOW, WHITE_HIGH)
    white_header_ratio = cv2.countNonZero(white_header_mask) / header_pixels

    # Full card white body
    white_body_mask = cv2.inRange(hsv_full, WHITE_LOW, WHITE_HIGH)
    white_body_ratio = cv2.countNonZero(white_body_mask) / total_pixels

    # --- Scoring ---
    score = 0.0
    has_saffron = saffron_ratio > 0.005
    has_green = green_ratio > 0.005
    has_white_header = white_header_ratio > 0.02
    has_white_body = white_body_ratio > 0.10

    if has_saffron:
        score += 0.25
    if has_green:
        score += 0.25
    if has_white_header:
        score += 0.20
    if has_white_body:
        score += 0.30

    passed = score >= 0.5

    detail = (
        f"Header — Saffron: {saffron_ratio:.1%}, Green: {green_ratio:.1%}, "
        f"White: {white_header_ratio:.1%} | Body White: {white_body_ratio:.1%} — "
        f"{'Tricolour detected' if (has_saffron and has_green) else 'No tricolour'}"
    )

    logger.debug("Colour profile: %s (score=%.2f)", detail, score)

    return {
        "passed": passed,
        "score": min(1.0, score),
        "detail": detail,
    }


def run(image: np.ndarray, **ctx) -> dict:
    """Pipeline-compatible entry point."""
    result = validate_color_profile(image)
    return {
        "passed": result["passed"],
        "score": result["score"],
        "detail": result["detail"],
        "data": {},
    }
