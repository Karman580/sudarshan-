# ==============================================================================
# yolo_detector.py — YOLOv8 Aadhaar Card Region Detector
# ==============================================================================
# Detects card-level regions using a YOLOv8 model:
#   - aadhaar_card    (class 0)
#   - photo_region    (class 1)
#   - qr_region       (class 2)
#   - number_region   (class 3)
#
# When YOLO weights are missing, falls back to an OpenCV contour-based
# card detector that estimates approximate regions via heuristic zones.
#
# Weight in pipeline: 15%
# ==============================================================================

import os
import cv2
import logging
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger("aadhaar_kyc")

CLASS_NAMES = {
    0: "aadhaar_card",
    1: "photo_region",
    2: "qr_region",
    3: "number_region",
}

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(_MODULE_DIR, "assets", "aadhaar_yolo.pt")

# Singleton model cache
_model_cache = {}


# ==============================================================================
# YOLO MODEL
# ==============================================================================

def _load_model(weights_path: str):
    """Load YOLOv8 model. Returns None if ultralytics unavailable or weights missing."""
    if not os.path.isfile(weights_path):
        logger.info("YOLO weights not found at %s — using contour fallback.", weights_path)
        return None

    try:
        from ultralytics import YOLO
        model = YOLO(weights_path)
        logger.info("YOLOv8 model loaded from %s", weights_path)
        return model
    except ImportError:
        logger.warning("ultralytics not installed — using contour fallback.")
        return None
    except Exception as e:
        logger.error("Failed to load YOLO model: %s", e)
        return None


def _yolo_detect(image: np.ndarray, weights_path: str, conf_threshold: float = 0.3) -> Optional[List[Dict]]:
    """Run YOLOv8 inference. Returns list of detections or None if model unavailable."""
    if weights_path not in _model_cache:
        _model_cache[weights_path] = _load_model(weights_path)

    model = _model_cache[weights_path]
    if model is None:
        return None

    try:
        results = model(image, conf=conf_threshold, verbose=False)
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                class_name = CLASS_NAMES.get(cls_id, f"unknown_{cls_id}")
                detections.append({
                    "class_name": class_name,
                    "confidence": round(conf, 4),
                    "bbox": [round(c, 1) for c in bbox],
                })
        return detections
    except Exception as e:
        logger.error("YOLO inference failed: %s", e)
        return None


# ==============================================================================
# CONTOUR-BASED FALLBACK DETECTOR
# ==============================================================================

def _find_card_contour(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Find the largest quadrilateral contour that looks like an ID card.

    Returns the 4-point contour or None.
    """
    h, w = image.shape[:2]
    min_area = h * w * 0.15  # Card must cover ≥ 15% of image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Try multiple edge detection approaches
    best_quad = None
    best_area = 0

    for canny_low, canny_high in [(30, 150), (50, 200), (75, 250)]:
        edged = cv2.Canny(blurred, canny_low, canny_high)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edged = cv2.dilate(edged, kernel, iterations=2)

        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and area > best_area:
                best_area = area
                best_quad = approx

    return best_quad


def _contour_fallback_detect(image: np.ndarray) -> List[Dict]:
    """
    Contour-based card detection fallback.

    If a rectangular card contour is found, generates approximate
    bounding boxes for expected regions using heuristic zones:
        - photo_region:  left 35%, middle 50% height
        - qr_region:     right 35%, lower 55% height
        - number_region: center 60%, bottom 25%
    """
    h, w = image.shape[:2]
    card_quad = _find_card_contour(image)

    detections = []

    if card_quad is not None:
        # Card detected
        x, y, cw, ch = cv2.boundingRect(card_quad)
        detections.append({
            "class_name": "aadhaar_card",
            "confidence": 0.6,
            "bbox": [float(x), float(y), float(x + cw), float(y + ch)],
        })

        # Estimate sub-regions relative to card bounding box
        # Photo region: left 35%, vertically 20-70%
        detections.append({
            "class_name": "photo_region",
            "confidence": 0.4,
            "bbox": [
                float(x + cw * 0.05),
                float(y + ch * 0.20),
                float(x + cw * 0.35),
                float(y + ch * 0.70),
            ],
        })

        # QR region: right 35%, vertically 35-90%
        detections.append({
            "class_name": "qr_region",
            "confidence": 0.4,
            "bbox": [
                float(x + cw * 0.60),
                float(y + ch * 0.35),
                float(x + cw * 0.95),
                float(y + ch * 0.90),
            ],
        })

        # Number region: center 60%, bottom 25%
        detections.append({
            "class_name": "number_region",
            "confidence": 0.4,
            "bbox": [
                float(x + cw * 0.20),
                float(y + ch * 0.75),
                float(x + cw * 0.80),
                float(y + ch * 0.98),
            ],
        })

        logger.info("Contour fallback: card detected at (%d,%d,%d,%d), 3 sub-regions estimated.", x, y, cw, ch)
    else:
        # No clear card contour — use full-image heuristic zones
        # Check if aspect ratio is card-like
        ratio = max(w, h) / max(min(w, h), 1)
        if 1.2 <= ratio <= 2.0:
            detections.append({
                "class_name": "aadhaar_card",
                "confidence": 0.3,
                "bbox": [0.0, 0.0, float(w), float(h)],
            })
            logger.info("Contour fallback: no quad found, aspect ratio %.2f is card-like, using full image.", ratio)
        else:
            logger.warning("Contour fallback: no card-like contour or aspect ratio detected.")

    return detections


# ==============================================================================
# PUBLIC API
# ==============================================================================

def detect_regions(
    image: np.ndarray,
    weights_path: str = DEFAULT_WEIGHTS,
    conf_threshold: float = 0.3,
) -> Dict:
    """
    Detect Aadhaar card regions using YOLO (primary) or contour fallback.

    Returns:
        Dict with detections, card_detected, score, detail.
    """
    # --- Try YOLO first ---
    yolo_results = _yolo_detect(image, weights_path, conf_threshold)

    if yolo_results is not None and len(yolo_results) > 0:
        detected_classes = {d["class_name"] for d in yolo_results}
        card_detected = "aadhaar_card" in detected_classes
        expected = {"aadhaar_card", "photo_region", "qr_region", "number_region"}
        found = detected_classes & expected
        score = len(found) / len(expected)

        card_confs = [d["confidence"] for d in yolo_results if d["class_name"] == "aadhaar_card"]
        if card_confs:
            avg_card_conf = sum(card_confs) / len(card_confs)
            score = score * 0.7 + avg_card_conf * 0.3

        detail = f"YOLO: {', '.join(sorted(found)) or 'none'} ({len(yolo_results)} boxes)"
        logger.info("YOLO detection: %s", detail)

        return {
            "detections": yolo_results,
            "card_detected": card_detected,
            "score": min(1.0, score),
            "detail": detail,
        }

    # --- Contour fallback ---
    fallback_results = _contour_fallback_detect(image)

    if fallback_results:
        detected_classes = {d["class_name"] for d in fallback_results}
        card_detected = "aadhaar_card" in detected_classes
        num_regions = len(fallback_results)
        # Slightly lower score for fallback since it's heuristic
        score = min(0.7, num_regions * 0.2)

        detail = f"Contour fallback: {', '.join(sorted(detected_classes))} ({num_regions} regions)"
        logger.info("Contour fallback: %s", detail)

        return {
            "detections": fallback_results,
            "card_detected": card_detected,
            "score": score,
            "detail": detail,
        }

    # --- Both failed ---
    logger.warning("Card detection failed: no YOLO model and no contour detected.")
    return {
        "detections": [],
        "card_detected": False,
        "score": 0.2,
        "detail": "No card detected (YOLO unavailable, contour fallback failed).",
    }


def run(image: np.ndarray, **ctx) -> dict:
    """Pipeline-compatible entry point."""
    result = detect_regions(image)
    return {
        "passed": result["card_detected"] is not False,
        "score": result["score"],
        "detail": result["detail"],
        "data": {
            "detections": result["detections"],
            "card_detected": result["card_detected"],
        },
    }
