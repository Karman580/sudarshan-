# ==============================================================================
# logo_detector.py — UIDAI Logo & Emblem Detection
# ==============================================================================
# Detects:
#   1. Ashoka Pillar emblem (circular structure in header)
#   2. Aadhaar sun logo
#
# Methods:
#   - Primary: ORB feature matching against reference templates (assets/)
#   - Fallback: Hough Circle Transform for circular emblem detection
#
# Weight in pipeline: 5%
# ==============================================================================

import os
import cv2
import numpy as np
import logging
from typing import Dict, Optional

logger = logging.getLogger("aadhaar_kyc")

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(_MODULE_DIR, "assets")

# Reference template filenames
EMBLEM_TEMPLATE = os.path.join(ASSETS_DIR, "ashoka_emblem.png")
AADHAAR_LOGO_TEMPLATE = os.path.join(ASSETS_DIR, "aadhaar_logo.png")


# ==============================================================================
# ORB FEATURE MATCHING
# ==============================================================================

def _orb_match(image_region: np.ndarray, template_path: str, min_matches: int = 8) -> float:
    """
    Match a template against an image region using ORB features.

    Returns a similarity score in [0, 1].
    """
    if not os.path.isfile(template_path):
        return -1.0  # Template unavailable

    try:
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            return -1.0

        gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY) if len(image_region.shape) == 3 else image_region

        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(gray, None)
        kp2, des2 = orb.detectAndCompute(template, None)

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return 0.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for m_pair in matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        score = min(1.0, len(good_matches) / max(min_matches, 1))
        return score

    except Exception as e:
        logger.debug("ORB matching error for %s: %s", template_path, e)
        return 0.0


# ==============================================================================
# HOUGH CIRCLE FALLBACK
# ==============================================================================

def _hough_circle_detect(image: np.ndarray) -> Dict:
    """
    Detect circular structures in the upper half (emblem region).

    Uses Hough Circle Transform as a rough structural indicator.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    h, w = gray.shape[:2]
    upper_region = gray[0:int(h * 0.5), :]

    blurred = cv2.GaussianBlur(upper_region, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=30,
        param1=100, param2=40,
        minRadius=10, maxRadius=min(h, w) // 6,
    )

    if circles is not None:
        num = len(circles[0])
        return {
            "found": True,
            "count": num,
            "score": min(1.0, 0.5 + num * 0.15),
        }
    return {"found": False, "count": 0, "score": 0.2}


# ==============================================================================
# PUBLIC API
# ==============================================================================

def detect_logos(image: np.ndarray) -> Dict:
    """
    Detect UIDAI logos and emblems on the card.

    Tries ORB template matching first, falls back to Hough circles.

    Returns:
        Dict with passed, score, detail.
    """
    h, w = image.shape[:2]
    upper_region = image[0:int(h * 0.5), :]

    # --- ORB matching ---
    emblem_score = _orb_match(upper_region, EMBLEM_TEMPLATE)
    logo_score = _orb_match(image, AADHAAR_LOGO_TEMPLATE)

    orb_available = emblem_score >= 0 or logo_score >= 0

    if orb_available:
        # At least one template was found
        scores = []
        details = []

        if emblem_score >= 0:
            scores.append(emblem_score)
            details.append(f"Emblem ORB: {emblem_score:.0%}")
        if logo_score >= 0:
            scores.append(logo_score)
            details.append(f"Aadhaar logo ORB: {logo_score:.0%}")

        avg = sum(scores) / len(scores) if scores else 0.0

        if avg >= 0.3:
            return {
                "passed": True,
                "score": avg,
                "detail": " | ".join(details),
            }

    # --- Hough circle fallback ---
    hough = _hough_circle_detect(image)

    detail = f"Hough circles: {'found' if hough['found'] else 'none'} ({hough['count']} candidates)"
    if orb_available:
        # Combine ORB + Hough
        combined = max(
            sum(s for s in [emblem_score, logo_score] if s >= 0) / max(1, sum(1 for s in [emblem_score, logo_score] if s >= 0)),
            hough["score"],
        )
        return {
            "passed": combined >= 0.3,
            "score": combined,
            "detail": f"ORB + {detail}",
        }

    return {
        "passed": hough["found"],
        "score": hough["score"],
        "detail": detail,
    }


def run(image: np.ndarray, **ctx) -> dict:
    """Pipeline-compatible entry point."""
    result = detect_logos(image)
    return {
        "passed": result["passed"],
        "score": result["score"],
        "detail": result["detail"],
        "data": {},
    }
