# ==============================================================================
# layout_validator.py — Aadhaar Card Spatial Layout Verification
# ==============================================================================
# Verifies that detected regions match the expected Aadhaar layout:
#
#   ┌─────────────────────────────────┐
#   │  TRICOLOUR HEADER (top 20%)    │
#   ├──────────┬──────────────────────┤
#   │  PHOTO   │  TEXT CONTENT        │
#   │  (left   │  Name / DOB / Gender │
#   │   35%)   │                      │
#   ├──────────┴──────────────────────┤
#   │  AADHAAR NUMBER (bottom 25%)    │
#   │  QR CODE (right 35%)           │
#   └─────────────────────────────────┘
#
# Uses bounding boxes from YOLO or heuristic zone analysis.
# Weight in pipeline: 10%
# ==============================================================================

import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("aadhaar_kyc")


# ==============================================================================
# EXPECTED SPATIAL RELATIONSHIPS
# ==============================================================================
# Normalised [0, 1] ranges for where each region's CENTER should fall.

EXPECTED_ZONES = {
    "photo_region": {
        "cx_range": (0.05, 0.45),   # left portion
        "cy_range": (0.25, 0.75),   # vertically centered
    },
    "qr_region": {
        "cx_range": (0.55, 0.95),   # right portion
        "cy_range": (0.35, 0.90),   # lower half
    },
    "number_region": {
        "cx_range": (0.20, 0.80),   # horizontally centered
        "cy_range": (0.70, 0.98),   # bottom strip
    },
}


def _normalise_bbox(bbox: List[float], img_w: int, img_h: int) -> Tuple[float, float]:
    """Convert [x1, y1, x2, y2] pixel coords to normalised center (cx, cy)."""
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2) / max(img_w, 1)
    cy = ((y1 + y2) / 2) / max(img_h, 1)
    return (cx, cy)


def _check_zone(
    cx: float, cy: float,
    cx_range: Tuple[float, float],
    cy_range: Tuple[float, float],
) -> float:
    """Score how well a detected region center falls within its expected zone."""
    cx_ok = cx_range[0] <= cx <= cx_range[1]
    cy_ok = cy_range[0] <= cy <= cy_range[1]

    if cx_ok and cy_ok:
        return 1.0

    score = 0.0
    if cx_ok:
        score += 0.4
    if cy_ok:
        score += 0.4

    return score


# ==============================================================================
# HEURISTIC ZONE ANALYSIS (when YOLO detections unavailable)
# ==============================================================================

def _heuristic_aspect_ratio(image: np.ndarray) -> float:
    """Check if image has standard ID card aspect ratio (~1.586)."""
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return 0.0
    ratio = max(w, h) / min(w, h)
    expected = 1.586
    deviation = abs(ratio - expected) / expected
    return max(0.0, 1.0 - deviation)


def _heuristic_header_band(image: np.ndarray) -> float:
    """
    Check if top 20% has colour variation (tricolour header indicator).

    A genuine Aadhaar header has saffron/white/green bands → high hue std.
    """
    h, w = image.shape[:2]
    header = image[0:int(h * 0.20), :]
    if header.size == 0:
        return 0.0

    hsv = cv2.cvtColor(header, cv2.COLOR_BGR2HSV) if len(header.shape) == 3 else None
    if hsv is None:
        return 0.0

    hue_std = np.std(hsv[:, :, 0].astype(float))
    return min(1.0, hue_std / 30.0)


def _heuristic_face_region(image: np.ndarray) -> float:
    """Check if left 35% of the card contains face-like features (edges, skin-tone)."""
    h, w = image.shape[:2]
    left_region = image[:, 0:int(w * 0.35)]
    if left_region.size == 0:
        return 0.0

    # Simple edge density check — faces have more edges than background
    gray = cv2.cvtColor(left_region, cv2.COLOR_BGR2GRAY) if len(left_region.shape) == 3 else left_region
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.count_nonzero(edges) / max(edges.size, 1)

    # Moderate edge density (0.05-0.3) suggests structured content (face)
    if 0.03 <= edge_ratio <= 0.40:
        return min(1.0, edge_ratio / 0.15)
    return 0.2


def _heuristic_qr_region(image: np.ndarray) -> float:
    """Check if right 35% of the card contains QR-like features (high contrast blocks)."""
    h, w = image.shape[:2]
    right_region = image[:, int(w * 0.60):]
    if right_region.size == 0:
        return 0.0

    gray = cv2.cvtColor(right_region, cv2.COLOR_BGR2GRAY) if len(right_region.shape) == 3 else right_region
    # QR codes have very high local contrast
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # QR codes typically have laplacian variance > 500
    return min(1.0, laplacian_var / 2000.0)


def _heuristic_number_region(image: np.ndarray) -> float:
    """Check if bottom 25% has text-like features."""
    h, w = image.shape[:2]
    bottom = image[int(h * 0.75):, :]
    if bottom.size == 0:
        return 0.0

    gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY) if len(bottom.shape) == 3 else bottom
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.count_nonzero(edges) / max(edges.size, 1)

    # Text regions have moderate edge density
    if 0.02 <= edge_ratio <= 0.30:
        return min(1.0, edge_ratio / 0.10)
    return 0.2


# ==============================================================================
# PUBLIC API
# ==============================================================================

def validate_layout(image: np.ndarray, detections: Optional[List[Dict]] = None) -> Dict:
    """
    Score the spatial layout of the Aadhaar card.

    If YOLO detections are provided, validates their spatial positions.
    Otherwise, runs comprehensive heuristic zone analysis.

    Returns:
        Dict with passed, score, detail.
    """
    h, w = image.shape[:2]

    # --- With YOLO/fallback detections ---
    if detections:
        zone_scores = {}
        for det in detections:
            cls = det.get("class_name", "")
            bbox = det.get("bbox", [])
            if cls in EXPECTED_ZONES and len(bbox) == 4:
                cx, cy = _normalise_bbox(bbox, w, h)
                zone = EXPECTED_ZONES[cls]
                score = _check_zone(cx, cy, zone["cx_range"], zone["cy_range"])
                zone_scores[cls] = score
                logger.debug("Layout zone '%s': cx=%.0f%%, cy=%.0f%% → score=%.0f%%", cls, cx*100, cy*100, score*100)

        if zone_scores:
            # Weight card detection if present
            avg_score = sum(zone_scores.values()) / max(len(EXPECTED_ZONES), 1)
            details = [f"{k}: {v:.0%}" for k, v in zone_scores.items()]
            result = {
                "passed": avg_score >= 0.4,
                "score": avg_score,
                "detail": f"Layout zone scores — {', '.join(details)}",
            }
            logger.info("Layout validation (detection-based): %s", result["detail"])
            return result

    # --- Heuristic fallback (no detections available) ---
    ar_score = _heuristic_aspect_ratio(image)
    hdr_score = _heuristic_header_band(image)
    face_score = _heuristic_face_region(image)
    qr_score = _heuristic_qr_region(image)
    num_score = _heuristic_number_region(image)

    # Weighted combination
    combined = (
        ar_score * 0.15 +
        hdr_score * 0.25 +
        face_score * 0.20 +
        qr_score * 0.20 +
        num_score * 0.20
    )

    heuristic_details = (
        f"AR: {ar_score:.0%}, Header: {hdr_score:.0%}, "
        f"FaceZone: {face_score:.0%}, QRZone: {qr_score:.0%}, "
        f"NumZone: {num_score:.0%}"
    )

    result = {
        "passed": combined >= 0.35,
        "score": combined,
        "detail": f"Heuristic layout — {heuristic_details}",
    }
    logger.info("Layout validation (heuristic): %s", result["detail"])
    return result


def run(image: np.ndarray, **ctx) -> dict:
    """Pipeline-compatible entry point."""
    detections = ctx.get("yolo_detections", [])
    result = validate_layout(image, detections=detections)
    return {
        "passed": result["passed"],
        "score": result["score"],
        "detail": result["detail"],
        "data": {},
    }
