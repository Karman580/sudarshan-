# ==============================================================================
# screen_detector.py — Screenshot / Screen Display Detection
# ==============================================================================
# Detects if an Aadhaar card image was photographed from a digital screen.
#
# Dual-condition detection (both must be true to flag):
#   1. Moiré pattern peaks in FFT spectrum (ratio > 0.02)
#   2. Vertical/horizontal grid frequency detection
#
# Relaxed thresholds to avoid false positives on normal camera images.
#
# Weight in pipeline: 3%
# ==============================================================================

import cv2
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger("aadhaar_kyc")

# Relaxed thresholds — original values caused false positives
MOIRE_RATIO_THRESHOLD = 0.02     # Was 0.004, now 5× more relaxed
NOISE_SCORE_THRESHOLD = 0.55     # Secondary bar
COMBINED_THRESHOLD = 0.55        # Both conditions must exceed this


def _compute_fft_spectrum(gray: np.ndarray) -> np.ndarray:
    """Compute the centered FFT magnitude spectrum (log-scaled)."""
    rows, cols = gray.shape
    opt_rows = cv2.getOptimalDFTSize(rows)
    opt_cols = cv2.getOptimalDFTSize(cols)

    padded = np.zeros((opt_rows, opt_cols), dtype=np.float32)
    padded[:rows, :cols] = gray.astype(np.float32)

    dft = cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft, axes=[0, 1])

    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    magnitude = np.log1p(magnitude)

    return magnitude


def _detect_moire_peaks(spectrum: np.ndarray, threshold_percentile: float = 99.5) -> Dict:
    """
    Detect periodic peaks in the FFT spectrum that indicate moiré patterns.

    Screen-captured images show distinct peaks at regular intervals
    in the frequency domain due to the screen's pixel grid.
    """
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2

    # Exclude the DC component (center region)
    dc_radius = min(h, w) // 15
    mask = np.ones_like(spectrum, dtype=np.uint8)
    cv2.circle(mask, (cx, cy), dc_radius, 0, -1)

    # Also exclude very high frequencies (noise)
    outer_mask = np.zeros_like(spectrum, dtype=np.uint8)
    outer_radius = min(h, w) // 3
    cv2.circle(outer_mask, (cx, cy), outer_radius, 1, -1)

    combined_mask = mask & outer_mask
    masked_vals = spectrum[combined_mask > 0]

    if len(masked_vals) == 0:
        return {"peak_count": 0, "peak_ratio": 0.0, "threshold": 0.0}

    threshold = np.percentile(masked_vals, threshold_percentile)
    peaks = np.sum(masked_vals > threshold)
    total_analysed = len(masked_vals)
    peak_ratio = peaks / max(total_analysed, 1)

    return {
        "peak_count": int(peaks),
        "peak_ratio": float(peak_ratio),
        "threshold": float(threshold),
    }


def _detect_grid_frequency(gray: np.ndarray) -> float:
    """
    Detect vertical/horizontal grid frequencies characteristic of screens.

    Analyses the 1D FFT of row and column averages:
    screens produce periodic spikes at the pixel pitch frequency.

    Returns a score in [0, 1] — higher means more grid-like.
    """
    h, w = gray.shape

    # Row-averaged horizontal profile
    row_avg = np.mean(gray.astype(np.float32), axis=0)
    # Column-averaged vertical profile
    col_avg = np.mean(gray.astype(np.float32), axis=1)

    grid_score = 0.0

    for profile in [row_avg, col_avg]:
        # Detrend
        profile = profile - np.mean(profile)
        if np.std(profile) < 1e-6:
            continue

        fft_mag = np.abs(np.fft.rfft(profile))
        # Skip DC
        fft_mag[0] = 0
        # Skip very low frequencies (first 5%)
        cutoff = max(3, len(fft_mag) // 20)
        fft_mag[:cutoff] = 0

        if len(fft_mag) == 0 or np.max(fft_mag) < 1e-6:
            continue

        # Peak-to-mean ratio: screens have sharp peaks
        mean_mag = np.mean(fft_mag[cutoff:])
        max_mag = np.max(fft_mag[cutoff:])

        if mean_mag > 0:
            ratio = max_mag / mean_mag
            # A ratio > 5 is suspicious; > 10 is very suspicious
            grid_score = max(grid_score, min(1.0, (ratio - 3.0) / 10.0))

    return max(0.0, grid_score)


def _compute_noise_score(gray: np.ndarray) -> float:
    """
    Estimate high-frequency noise level.

    Screen images tend to have more structured high-frequency noise
    than optical photographs.
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = np.var(laplacian)

    # Normalise — values above ~3000 suggest screen capture
    return min(1.0, lap_var / 4000.0)


# ==============================================================================
# PUBLIC API
# ==============================================================================

def detect_screen(image: np.ndarray) -> Dict:
    """
    Analyse image for screen capture / screenshot artifacts.

    Uses dual-condition detection: BOTH moiré AND grid frequency
    must be elevated to flag as screen. This prevents false positives
    on normal camera-captured images.

    Returns:
        Dict with screen_detected, score, detail.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # --- FFT moiré analysis ---
    spectrum = _compute_fft_spectrum(gray)
    moire = _detect_moire_peaks(spectrum)

    # --- Grid frequency analysis ---
    grid_score = _detect_grid_frequency(gray)

    # --- Noise analysis ---
    noise_score = _compute_noise_score(gray)

    # --- Combined scoring (dual-condition) ---
    moire_score = min(1.0, moire["peak_ratio"] / MOIRE_RATIO_THRESHOLD)
    has_moire = moire["peak_ratio"] > MOIRE_RATIO_THRESHOLD
    has_grid = grid_score > 0.3

    # Screen detected ONLY if BOTH conditions are met
    # This dramatically reduces false positives
    screen_detected = has_moire and has_grid

    # For pipeline scoring: invert (higher = more likely REAL / not screen)
    if screen_detected:
        combined = moire_score * 0.4 + grid_score * 0.4 + noise_score * 0.2
        pipeline_score = max(0.0, 1.0 - combined)
    else:
        pipeline_score = 1.0 - (moire_score * 0.2 + grid_score * 0.15 + noise_score * 0.1)
        pipeline_score = max(0.3, min(1.0, pipeline_score))

    detail = (
        f"Moiré: ratio={moire['peak_ratio']:.4f} ({'⚠' if has_moire else '✓'}), "
        f"Grid: {grid_score:.2f} ({'⚠' if has_grid else '✓'}), "
        f"Noise: {noise_score:.2f} — "
        f"{'Screen detected' if screen_detected else 'No screen artifacts'}"
    )

    logger.info("Screen detection: %s", detail)

    return {
        "screen_detected": screen_detected,
        "score": pipeline_score,
        "detail": detail,
        "moire_data": moire,
        "grid_score": grid_score,
        "noise_score": noise_score,
    }


def run(image: np.ndarray, **ctx) -> dict:
    """Pipeline-compatible entry point."""
    result = detect_screen(image)
    return {
        "passed": not result["screen_detected"],
        "score": result["score"],
        "detail": result["detail"],
        "data": {
            "screen_detected": result["screen_detected"],
        },
    }
