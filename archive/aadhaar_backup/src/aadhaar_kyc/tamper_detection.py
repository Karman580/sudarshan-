# ==============================================================================
# tamper_detection.py — Error Level Analysis (ELA) for Tampering Detection
# ==============================================================================
# Detects localised image manipulations by analysing JPEG compression
# artefact inconsistencies.
#
# Steps:
#   1. Re-compress image at quality 90
#   2. Compute pixel-wise absolute difference
#   3. Measure ELA variance — tampered regions have anomalous error levels
#   4. Detect high-variance patches (copy-paste indicators)
#
# Weight in pipeline: 5%
# ==============================================================================

import cv2
import numpy as np
import logging
import io
from typing import Dict

logger = logging.getLogger("aadhaar_kyc")

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.warning("PIL not installed — ELA disabled.")

try:
    from skimage.util import view_as_blocks
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


# ==============================================================================
# ERROR LEVEL ANALYSIS
# ==============================================================================

def _compute_ela(image: np.ndarray, quality: int = 90) -> np.ndarray:
    """
    Perform Error Level Analysis.

    Re-saves the image at specified JPEG quality, then computes the
    absolute difference between original and re-compressed versions.

    Returns:
        ELA difference image (single channel, amplified).
    """
    if not HAS_PIL:
        # Fallback: use OpenCV JPEG encode/decode
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode(".jpg", image, encode_param)
        recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        diff = cv2.absdiff(image, recompressed)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        return gray_diff

    # PIL path (more faithful to standard ELA)
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)

        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        recompressed_pil = PILImage.open(buffer)

        original_arr = np.array(pil_img, dtype=np.float32)
        recomp_arr = np.array(recompressed_pil, dtype=np.float32)

        diff = np.abs(original_arr - recomp_arr)
        # Average across channels
        gray_diff = np.mean(diff, axis=2).astype(np.uint8)

        return gray_diff

    except Exception as e:
        logger.error("PIL ELA failed, using CV fallback: %s", e)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode(".jpg", image, encode_param)
        recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        diff = cv2.absdiff(image, recompressed)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        return gray_diff


def _analyse_ela_patches(ela_image: np.ndarray, patch_size: int = 32) -> Dict:
    """
    Analyse ELA image in patches to find regions with anomalous error levels.

    Tampered regions show significantly higher or lower ELA values
    compared to the rest of the image.
    """
    h, w = ela_image.shape
    patch_h = h // patch_size
    patch_w = w // patch_size

    if patch_h < 2 or patch_w < 2:
        # Image too small for patch analysis
        overall_mean = float(np.mean(ela_image))
        overall_std = float(np.std(ela_image))
        return {
            "mean": overall_mean,
            "std": overall_std,
            "max_patch_mean": overall_mean,
            "anomaly_ratio": 0.0,
        }

    # Crop to exact patch grid
    cropped = ela_image[:patch_h * patch_size, :patch_w * patch_size]

    # Compute mean ELA per patch
    patch_means = []
    for i in range(patch_h):
        for j in range(patch_w):
            patch = cropped[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            patch_means.append(float(np.mean(patch)))

    patch_means = np.array(patch_means)
    overall_mean = float(np.mean(patch_means))
    overall_std = float(np.std(patch_means))

    # Anomalous patches: mean > 2 standard deviations above overall mean
    if overall_std > 0:
        threshold = overall_mean + 2 * overall_std
        anomalous = np.sum(patch_means > threshold)
        anomaly_ratio = anomalous / len(patch_means)
    else:
        anomaly_ratio = 0.0

    return {
        "mean": overall_mean,
        "std": overall_std,
        "max_patch_mean": float(np.max(patch_means)),
        "anomaly_ratio": float(anomaly_ratio),
    }


# ==============================================================================
# PUBLIC API
# ==============================================================================

def detect_tampering(image: np.ndarray) -> Dict:
    """
    Detect image tampering using Error Level Analysis.

    Returns:
        Dict with tamper_detected, score, detail, ela_stats.
    """
    ela_image = _compute_ela(image)
    stats = _analyse_ela_patches(ela_image)

    # --- Scoring ---
    # Low anomaly ratio → likely untampered → high score
    # High anomaly ratio → likely tampered → low score
    anomaly = stats["anomaly_ratio"]

    # Also check overall ELA variance — very uniform ELA is suspicious
    # (could indicate full AI generation), but moderate variance is normal
    ela_std = stats["std"]

    # Score logic
    if anomaly > 0.15:
        # Significant anomalies found
        tamper_detected = True
        score = max(0.0, 1.0 - anomaly * 3)
    elif ela_std < 0.5:
        # Suspiciously uniform (possible AI generation)
        tamper_detected = True
        score = 0.3
    else:
        tamper_detected = False
        score = min(1.0, 0.7 + (1.0 - anomaly) * 0.3)

    detail = (
        f"ELA — mean: {stats['mean']:.1f}, std: {stats['std']:.1f}, "
        f"max patch: {stats['max_patch_mean']:.1f}, "
        f"anomaly ratio: {anomaly:.1%} — "
        f"{'Tampering suspected' if tamper_detected else 'No significant tampering'}"
    )

    logger.debug("Tamper detection: %s", detail)

    return {
        "tamper_detected": tamper_detected,
        "score": score,
        "detail": detail,
        "ela_stats": stats,
    }


def run(image: np.ndarray, **ctx) -> dict:
    """Pipeline-compatible entry point."""
    result = detect_tampering(image)
    return {
        "passed": not result["tamper_detected"],
        "score": result["score"],
        "detail": result["detail"],
        "data": {
            "tamper_detected": result["tamper_detected"],
            "ela_stats": result["ela_stats"],
        },
    }
