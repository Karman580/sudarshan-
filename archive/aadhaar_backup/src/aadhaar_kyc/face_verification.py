# ==============================================================================
# face_verification.py — Face Detection & Optional Selfie Match
# ==============================================================================
# Detection:
#   - Primary: MediaPipe Face Detection (GPU-less, fast)
#   - Fallback: OpenCV Haar Cascade
#
# Aadhaar-specific constraints:
#   - Exactly ONE face expected on card
#   - Face area must be 3-40% of card area
#   - Face center-x must be between 5-40% (left portion)
#   - Face center-y must be between 20-70% (vertically centered)
#   - Faces outside this region are rejected as false positives
#
# Optional selfie match:
#   - Compare card photo vs live selfie using DeepFace
#
# Weights: face detect 5%, selfie match 10%
# ==============================================================================

import cv2
import numpy as np
import logging
import os
from typing import Dict, Optional, List

logger = logging.getLogger("aadhaar_kyc")

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    logger.warning("mediapipe not installed — using OpenCV Haar cascade fallback.")

try:
    from deepface import DeepFace
    HAS_DEEPFACE = True
except ImportError:
    HAS_DEEPFACE = False

# Haar cascade path (ships with OpenCV)
HAAR_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ---------------------------------------------------------------------------
# AADHAAR FACE CONSTRAINTS
# ---------------------------------------------------------------------------
# These define where a genuine Aadhaar card photo should appear,
# as normalised [0, 1] coordinates relative to the card.
FACE_MIN_AREA_RATIO = 0.03   # Face must be ≥ 3% of card area
FACE_MAX_AREA_RATIO = 0.40   # Face must be ≤ 40% of card area
FACE_CX_MIN = 0.05           # Face center-x must be ≥ 5% from left
FACE_CX_MAX = 0.55           # Face center-x must be ≤ 55% from left
FACE_CY_MIN = 0.15           # Face center-y must be ≥ 15% from top
FACE_CY_MAX = 0.85           # Face center-y must be ≤ 85% from top
FACE_MIN_PIXELS = 30         # Minimum face width/height in pixels


# ==============================================================================
# FACE DETECTION
# ==============================================================================

def _detect_faces_mediapipe(image: np.ndarray) -> List[Dict]:
    """Detect faces using MediaPipe Face Detection."""
    if not HAS_MEDIAPIPE:
        return []

    try:
        mp_face = mp.solutions.face_detection
        with mp_face.FaceDetection(
            model_selection=1,  # Full-range model
            min_detection_confidence=0.4,
        ) as detector:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)

            if not results.detections:
                return []

            h, w = image.shape[:2]
            faces = []
            for det in results.detections:
                bb = det.location_data.relative_bounding_box
                x1 = max(0, int(bb.xmin * w))
                y1 = max(0, int(bb.ymin * h))
                x2 = min(w, int((bb.xmin + bb.width) * w))
                y2 = min(h, int((bb.ymin + bb.height) * h))
                conf = det.score[0] if det.score else 0.5
                faces.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(conf),
                    "engine": "mediapipe",
                })
            return faces
    except Exception as e:
        logger.error("MediaPipe face detection error: %s", e)
        return []


def _detect_faces_haar(image: np.ndarray) -> List[Dict]:
    """Detect faces using OpenCV Haar cascade (fallback)."""
    try:
        cascade = cv2.CascadeClassifier(HAAR_CASCADE)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        rects = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(FACE_MIN_PIXELS, FACE_MIN_PIXELS),
        )

        faces = []
        for (x, y, fw, fh) in rects:
            faces.append({
                "bbox": [int(x), int(y), int(x + fw), int(y + fh)],
                "confidence": 0.5,
                "engine": "haar",
            })
        return faces
    except Exception as e:
        logger.error("Haar cascade error: %s", e)
        return []


def _filter_aadhaar_faces(faces: List[Dict], img_w: int, img_h: int) -> List[Dict]:
    """
    Filter detected faces by Aadhaar-specific spatial constraints.

    Removes faces that are:
      - Too small (< 3% of card area)
      - Too large (> 40% of card area)
      - Outside the expected photo region (left part of card)
    """
    card_area = max(img_w * img_h, 1)
    valid = []

    for face in faces:
        x1, y1, x2, y2 = face["bbox"]
        fw = x2 - x1
        fh = y2 - y1

        # Skip tiny faces (noise)
        if fw < FACE_MIN_PIXELS or fh < FACE_MIN_PIXELS:
            logger.debug("Face rejected: too small (%dx%d px).", fw, fh)
            continue

        face_area = fw * fh
        area_ratio = face_area / card_area

        # Area check
        if area_ratio < FACE_MIN_AREA_RATIO:
            logger.debug("Face rejected: area %.1f%% < %.1f%% min.", area_ratio * 100, FACE_MIN_AREA_RATIO * 100)
            continue
        if area_ratio > FACE_MAX_AREA_RATIO:
            logger.debug("Face rejected: area %.1f%% > %.1f%% max.", area_ratio * 100, FACE_MAX_AREA_RATIO * 100)
            continue

        # Position check
        cx = (x1 + x2) / 2 / max(img_w, 1)
        cy = (y1 + y2) / 2 / max(img_h, 1)

        if not (FACE_CX_MIN <= cx <= FACE_CX_MAX):
            logger.debug("Face rejected: cx=%.0f%% outside [%.0f%%, %.0f%%].", cx * 100, FACE_CX_MIN * 100, FACE_CX_MAX * 100)
            continue
        if not (FACE_CY_MIN <= cy <= FACE_CY_MAX):
            logger.debug("Face rejected: cy=%.0f%% outside [%.0f%%, %.0f%%].", cy * 100, FACE_CY_MIN * 100, FACE_CY_MAX * 100)
            continue

        face["area_ratio"] = area_ratio
        face["cx"] = cx
        face["cy"] = cy
        valid.append(face)

    return valid


def detect_card_face(image: np.ndarray) -> Dict:
    """
    Detect faces on the Aadhaar card image with spatial constraint filtering.

    Rules:
        - Exactly 1 valid face → good
        - 0 valid faces → suspicious (but may be due to image quality)
        - >1 valid faces → very suspicious

    Returns:
        Dict with passed, score, detail, faces list, face_detected.
    """
    h, w = image.shape[:2]

    # Try MediaPipe first, then Haar
    raw_faces = _detect_faces_mediapipe(image)
    engine = "mediapipe"

    if not raw_faces:
        raw_faces = _detect_faces_haar(image)
        engine = "haar"

    logger.info("Raw face detections (%s): %d", engine, len(raw_faces))

    # Apply Aadhaar spatial constraints
    valid_faces = _filter_aadhaar_faces(raw_faces, w, h)

    logger.info(
        "Filtered faces: %d valid out of %d raw (%s).",
        len(valid_faces), len(raw_faces), engine,
    )

    num_faces = len(valid_faces)

    if num_faces == 0:
        return {
            "passed": False,
            "score": 0.2,
            "detail": f"No valid face in Aadhaar region ({engine}, {len(raw_faces)} raw detections filtered out).",
            "faces": [],
            "face_detected": False,
        }

    if num_faces > 1:
        return {
            "passed": False,
            "score": 0.1,
            "detail": f"Multiple faces ({num_faces}) in Aadhaar region — possible spoofing.",
            "faces": valid_faces,
            "face_detected": True,
        }

    # Exactly 1 valid face — score based on position quality
    face = valid_faces[0]
    area_ratio = face.get("area_ratio", 0.1)
    cx = face.get("cx", 0.3)
    cy = face.get("cy", 0.5)

    score = 0.6  # Base score for finding exactly 1 valid face

    # Bonus for ideal position (centre-left, vertically centred)
    ideal_cx = 0.25
    ideal_cy = 0.45
    cx_dev = abs(cx - ideal_cx) / 0.5
    cy_dev = abs(cy - ideal_cy) / 0.5
    position_bonus = max(0, 0.2 - (cx_dev + cy_dev) * 0.1)
    score += position_bonus

    # Bonus for ideal size (~10-25% of card)
    if 0.08 <= area_ratio <= 0.30:
        score += 0.15
    elif 0.05 <= area_ratio <= 0.35:
        score += 0.05

    score = max(0.0, min(1.0, score))

    detail = (
        f"1 face detected ({engine}) — "
        f"size: {area_ratio:.1%} of card, "
        f"position: ({cx:.0%}, {cy:.0%}), "
        f"conf: {face.get('confidence', 0):.0%}"
    )

    return {
        "passed": True,
        "score": score,
        "detail": detail,
        "faces": valid_faces,
        "face_detected": True,
    }


# ==============================================================================
# SELFIE MATCH (Optional)
# ==============================================================================

def match_selfie(card_image: np.ndarray, selfie_image: np.ndarray) -> Dict:
    """
    Compare the face on the Aadhaar card against a live selfie.

    Uses DeepFace verification (ArcFace model by default).

    Returns:
        Dict with matched (bool), score, detail.
    """
    if not HAS_DEEPFACE:
        return {
            "matched": None,
            "score": 0.5,
            "detail": "DeepFace not installed — selfie match skipped.",
        }

    try:
        result = DeepFace.verify(
            img1_path=card_image,
            img2_path=selfie_image,
            model_name="ArcFace",
            enforce_detection=False,
            detector_backend="opencv",
        )

        verified = result.get("verified", False)
        distance = result.get("distance", 1.0)
        threshold = result.get("threshold", 0.68)

        # Convert distance to similarity score
        score = max(0.0, 1.0 - distance / max(threshold * 2, 0.01))

        logger.info(
            "Selfie match: verified=%s, distance=%.3f, threshold=%.3f",
            verified, distance, threshold,
        )

        return {
            "matched": verified,
            "score": score,
            "detail": f"Selfie match: {'VERIFIED' if verified else 'FAILED'} (distance={distance:.3f}, threshold={threshold:.3f})",
        }
    except Exception as e:
        logger.error("Selfie match error: %s", e)
        return {
            "matched": None,
            "score": 0.5,
            "detail": f"Selfie match error: {e}",
        }


# ==============================================================================
# PIPELINE-COMPATIBLE API
# ==============================================================================

def run(image: np.ndarray, **ctx) -> dict:
    """Pipeline-compatible entry point for face detection."""
    result = detect_card_face(image)
    return {
        "passed": result["passed"],
        "score": result["score"],
        "detail": result["detail"],
        "data": {
            "faces": result["faces"],
            "face_detected": result["face_detected"],
        },
    }


def run_selfie(image: np.ndarray, **ctx) -> dict:
    """Pipeline-compatible entry point for selfie verification."""
    selfie = ctx.get("selfie_image")
    if selfie is None:
        return {
            "passed": None,
            "score": 0.5,
            "detail": "No selfie provided — match skipped.",
            "data": {"matched": None},
        }

    result = match_selfie(image, selfie)
    return {
        "passed": result["matched"],
        "score": result["score"],
        "detail": result["detail"],
        "data": {"matched": result["matched"]},
    }
