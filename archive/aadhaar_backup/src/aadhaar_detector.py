# ==============================================================================
# aadhaar_detector.py - Aadhaar Card Fake/Real Detection Module
# ==============================================================================
# Heuristic-based Aadhaar card validation using structural analysis:
#   - OCR-based Aadhaar number format validation (XXXX XXXX XXXX)
#   - QR code presence and decodability
#   - Expected text pattern matching (UIDAI, Government of India, etc.)
#   - Color histogram analysis (tricolor header)
#   - Aspect ratio check (standard ID card ratio)
#   - OCR confidence scoring for font consistency
#
# Patented under Indian Patent Law.
# Developed @ Thapar Institute of Engineering & Technology, Patiala
# ==============================================================================

import cv2
import numpy as np
import re
import os
from typing import Dict, List, Tuple, Optional
from PIL import Image

# ---------------------------------------------------------------------------
# Optional dependency imports (graceful fallback)
# ---------------------------------------------------------------------------
try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    print("[WARN] pytesseract not installed. OCR checks will be skipped.")

try:
    from pyzbar.pyzbar import decode as decode_qr
    HAS_PYZBAR = True
except ImportError:
    HAS_PYZBAR = False
    print("[WARN] pyzbar not installed. QR code checks will be skipped.")


# ==============================================================================
# CONSTANTS
# ==============================================================================

# Standard ID card aspect ratio (width / height) — ISO/IEC 7810 ID-1
STANDARD_ASPECT_RATIO = 1.586
ASPECT_RATIO_TOLERANCE = 0.35  # Allow generous tolerance

# Aadhaar number regex: 4 digits, separator, 4 digits, separator, 4 digits
AADHAAR_NUMBER_PATTERN = re.compile(r'\b\d{4}\s?\d{4}\s?\d{4}\b')

# VID pattern (Virtual ID): 16 digits
VID_PATTERN = re.compile(r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b')

# Expected keywords on an Aadhaar card (case-insensitive matching)
EXPECTED_KEYWORDS = [
    "government of india",
    "aadhaar",
    "unique identification",
    "uidai",
]

# Secondary keywords (at least some should be present)
SECONDARY_KEYWORDS = [
    "male", "female", "dob", "date of birth",
    "year of birth", "address", "enrolment",
    "/male", "/female",
]

# Aadhaar tricolor header — approximate HSV ranges for saffron/white/green
SAFFRON_HSV_RANGE = ((5, 80, 150), (25, 255, 255))
GREEN_HSV_RANGE = ((35, 60, 80), (85, 255, 220))


# ==============================================================================
# INDIVIDUAL CHECK FUNCTIONS
# ==============================================================================

def check_aspect_ratio(image: np.ndarray) -> Dict:
    """
    Check if the image has a standard ID card aspect ratio.
    
    Standard credit-card-size ID: ~85.6mm × 53.98mm → ratio ≈ 1.586
    We allow both landscape and portrait orientations.
    """
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return {"passed": False, "detail": "Invalid image dimensions.", "score": 0.0}
    
    ratio = max(w, h) / min(w, h)
    deviation = abs(ratio - STANDARD_ASPECT_RATIO) / STANDARD_ASPECT_RATIO
    
    passed = deviation <= ASPECT_RATIO_TOLERANCE
    score = max(0.0, 1.0 - deviation)
    
    return {
        "passed": passed,
        "detail": f"Aspect ratio: {ratio:.3f} (expected ~{STANDARD_ASPECT_RATIO:.3f}, deviation: {deviation:.1%})",
        "score": score
    }


def check_aadhaar_number(ocr_text: str) -> Dict:
    """
    Check if the OCR text contains a valid 12-digit Aadhaar number format.
    
    Valid format: XXXX XXXX XXXX (with or without spaces).
    Also validates Verhoeff checksum if a clean 12-digit number is found.
    """
    if not ocr_text:
        return {"passed": False, "detail": "No text extracted from image.", "score": 0.0}
    
    matches = AADHAAR_NUMBER_PATTERN.findall(ocr_text)
    
    if not matches:
        return {
            "passed": False,
            "detail": "No 12-digit Aadhaar number pattern found.",
            "score": 0.0
        }
    
    # Check Verhoeff checksum for each match
    valid_numbers = []
    for match in matches:
        clean_number = match.replace(" ", "")
        if len(clean_number) == 12 and clean_number.isdigit():
            if _verhoeff_check(clean_number):
                valid_numbers.append(match)
    
    if valid_numbers:
        return {
            "passed": True,
            "detail": f"Valid Aadhaar number found: {valid_numbers[0][:4]} XXXX XXXX (Verhoeff ✓)",
            "score": 1.0
        }
    elif matches:
        # Found 12-digit patterns but Verhoeff failed
        masked = matches[0].replace(" ", "")
        return {
            "passed": True,  # Still pass — could be OCR read error
            "detail": f"12-digit number found: {masked[:4]} XXXX XXXX (checksum unverified)",
            "score": 0.7
        }
    
    return {"passed": False, "detail": "No valid Aadhaar number pattern.", "score": 0.0}


def check_qr_code(image: np.ndarray) -> Dict:
    """
    Check for presence and decodability of QR codes.
    
    All Aadhaar cards issued since 2018 have a Secure QR code.
    Older cards (pre-2018) may have a simple QR or no QR.
    """
    if not HAS_PYZBAR:
        return {
            "passed": None,  # Indeterminate
            "detail": "pyzbar not installed — QR check skipped.",
            "score": 0.5
        }
    
    try:
        # Try multiple preprocessing approaches for robust QR detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Attempt 1: Direct decode
        qr_results = decode_qr(gray)
        
        # Attempt 2: Threshold + decode
        if not qr_results:
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            qr_results = decode_qr(thresh)
        
        # Attempt 3: Adaptive threshold
        if not qr_results:
            adaptive = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            qr_results = decode_qr(adaptive)
        
        # Attempt 4: Upscale small images
        if not qr_results and max(gray.shape) < 800:
            scale = 2
            upscaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            qr_results = decode_qr(upscaled)
        
        if qr_results:
            qr_types = [r.type for r in qr_results]
            return {
                "passed": True,
                "detail": f"QR code detected ({len(qr_results)} code(s), type: {', '.join(qr_types)})",
                "score": 1.0
            }
        else:
            return {
                "passed": False,
                "detail": "No QR code detected. (Could be a pre-2018 card or image quality issue.)",
                "score": 0.0
            }
    except Exception as e:
        return {
            "passed": None,
            "detail": f"QR detection error: {str(e)}",
            "score": 0.3
        }


def check_keyword_presence(ocr_text: str) -> Dict:
    """
    Check for expected Aadhaar-specific keywords in OCR text.
    
    Expects: 'Government of India', 'Aadhaar', 'UIDAI', etc.
    """
    if not ocr_text:
        return {"passed": False, "detail": "No text extracted.", "score": 0.0}
    
    text_lower = ocr_text.lower()
    
    # Check primary keywords
    primary_found = []
    primary_missing = []
    for kw in EXPECTED_KEYWORDS:
        if kw in text_lower:
            primary_found.append(kw)
        else:
            primary_missing.append(kw)
    
    # Check secondary keywords
    secondary_found = []
    for kw in SECONDARY_KEYWORDS:
        if kw in text_lower:
            secondary_found.append(kw)
    
    primary_ratio = len(primary_found) / len(EXPECTED_KEYWORDS) if EXPECTED_KEYWORDS else 0
    has_secondary = len(secondary_found) >= 1
    
    # Decision logic
    if primary_ratio >= 0.5:
        passed = True
        score = primary_ratio * 0.7 + (0.3 if has_secondary else 0.0)
    elif primary_ratio > 0 and has_secondary:
        passed = True
        score = 0.5
    else:
        passed = False
        score = primary_ratio * 0.3
    
    detail_parts = []
    if primary_found:
        detail_parts.append(f"Found: {', '.join(primary_found)}")
    if primary_missing:
        detail_parts.append(f"Missing: {', '.join(primary_missing)}")
    if secondary_found:
        detail_parts.append(f"Secondary: {', '.join(secondary_found[:3])}")
    
    return {
        "passed": passed,
        "detail": " | ".join(detail_parts),
        "score": min(1.0, score)
    }


def check_color_profile(image: np.ndarray) -> Dict:
    """
    Analyze color distribution for Aadhaar-like characteristics.
    
    Checks for presence of saffron and green (tricolor header),
    and sufficient white/light regions (card body).
    """
    if len(image.shape) < 3:
        return {"passed": None, "detail": "Grayscale image — color check skipped.", "score": 0.5}
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    total_pixels = hsv.shape[0] * hsv.shape[1]
    
    if total_pixels == 0:
        return {"passed": False, "detail": "Empty image.", "score": 0.0}
    
    # Check saffron region
    saffron_mask = cv2.inRange(hsv, np.array(SAFFRON_HSV_RANGE[0]), np.array(SAFFRON_HSV_RANGE[1]))
    saffron_ratio = cv2.countNonZero(saffron_mask) / total_pixels
    
    # Check green region
    green_mask = cv2.inRange(hsv, np.array(GREEN_HSV_RANGE[0]), np.array(GREEN_HSV_RANGE[1]))
    green_ratio = cv2.countNonZero(green_mask) / total_pixels
    
    # Check white/light regions (card body)
    white_mask = cv2.inRange(hsv, np.array((0, 0, 200)), np.array((180, 40, 255)))
    white_ratio = cv2.countNonZero(white_mask) / total_pixels
    
    has_tricolor = saffron_ratio > 0.005 and green_ratio > 0.005
    has_white_body = white_ratio > 0.10
    
    score = 0.0
    if has_tricolor:
        score += 0.5
    if has_white_body:
        score += 0.3
    if saffron_ratio > 0.01:
        score += 0.1
    if green_ratio > 0.01:
        score += 0.1
    
    passed = score >= 0.5
    
    detail = (
        f"Saffron: {saffron_ratio:.1%}, Green: {green_ratio:.1%}, "
        f"White: {white_ratio:.1%} — "
        f"{'Tricolor detected' if has_tricolor else 'No tricolor pattern'}"
    )
    
    return {"passed": passed, "detail": detail, "score": min(1.0, score)}


def check_ocr_confidence(image: np.ndarray) -> Dict:
    """
    Evaluate OCR confidence as a proxy for print quality and font consistency.
    
    Genuine Aadhaar cards have consistent, high-quality printing.
    Fakes often have blurry text, unusual fonts, or low print quality.
    """
    if not HAS_TESSERACT:
        return {"passed": None, "detail": "Tesseract not available.", "score": 0.5}
    
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Get detailed OCR data with confidence scores
        data = pytesseract.image_to_data(
            gray, output_type=pytesseract.Output.DICT,
            config='--psm 6'
        )
        
        confidences = [
            int(c) for c in data['conf']
            if str(c).isdigit() and int(c) > 0
        ]
        
        if not confidences:
            return {
                "passed": False,
                "detail": "No text detected with measurable confidence.",
                "score": 0.2
            }
        
        avg_conf = sum(confidences) / len(confidences)
        high_conf_count = sum(1 for c in confidences if c > 60)
        high_conf_ratio = high_conf_count / len(confidences)
        
        if avg_conf > 55 and high_conf_ratio > 0.4:
            passed = True
            score = min(1.0, avg_conf / 100)
        elif avg_conf > 35:
            passed = True
            score = 0.5
        else:
            passed = False
            score = max(0.1, avg_conf / 100)
        
        return {
            "passed": passed,
            "detail": f"Average OCR confidence: {avg_conf:.0f}% ({len(confidences)} words, {high_conf_ratio:.0%} high-confidence)",
            "score": score
        }
    except Exception as e:
        return {"passed": None, "detail": f"OCR confidence check error: {str(e)}", "score": 0.3}


def check_emblem_structure(image: np.ndarray) -> Dict:
    """
    Check for the presence of circular/emblem-like structures.
    
    Uses Hough Circle Transform to detect the Ashoka Pillar emblem region.
    This is a rough structural check, not template matching.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Look for circular structures in the upper portion of the card
    h, w = gray.shape[:2]
    upper_region = gray[0:int(h * 0.5), :]
    
    # Blur and detect circles
    blurred = cv2.GaussianBlur(upper_region, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=30,
        param1=100, param2=40,
        minRadius=10, maxRadius=min(h, w) // 6
    )
    
    if circles is not None:
        num_circles = len(circles[0])
        return {
            "passed": True,
            "detail": f"Circular/emblem structure(s) detected in header region ({num_circles} candidate(s)).",
            "score": min(1.0, 0.5 + num_circles * 0.15)
        }
    else:
        return {
            "passed": False,
            "detail": "No circular emblem structure detected in header region.",
            "score": 0.2
        }


# ==============================================================================
# VERHOEFF CHECKSUM (Aadhaar uses Verhoeff algorithm)
# ==============================================================================

_VERHOEFF_TABLE_D = [
    [0,1,2,3,4,5,6,7,8,9],
    [1,2,3,4,0,6,7,8,9,5],
    [2,3,4,0,1,7,8,9,5,6],
    [3,4,0,1,2,8,9,5,6,7],
    [4,0,1,2,3,9,5,6,7,8],
    [5,9,8,7,6,0,4,3,2,1],
    [6,5,9,8,7,1,0,4,3,2],
    [7,6,5,9,8,2,1,0,4,3],
    [8,7,6,5,9,3,2,1,0,4],
    [9,8,7,6,5,4,3,2,1,0],
]

_VERHOEFF_TABLE_P = [
    [0,1,2,3,4,5,6,7,8,9],
    [1,5,7,6,2,8,3,0,9,4],
    [5,8,0,3,7,9,6,1,4,2],
    [8,9,1,6,0,4,3,5,2,7],
    [9,4,5,3,1,2,6,8,7,0],
    [4,2,8,6,5,7,3,9,0,1],
    [2,7,9,3,8,0,6,4,1,5],
    [7,0,4,6,9,1,3,2,5,8],
]

_VERHOEFF_TABLE_INV = [0,4,3,2,1,5,6,7,8,9]


def _verhoeff_check(number: str) -> bool:
    """Validate a number using the Verhoeff checksum algorithm."""
    try:
        c = 0
        digits = [int(d) for d in reversed(number)]
        for i, digit in enumerate(digits):
            c = _VERHOEFF_TABLE_D[c][_VERHOEFF_TABLE_P[i % 8][digit]]
        return c == 0
    except (ValueError, IndexError):
        return False


# ==============================================================================
# MAIN ANALYSIS FUNCTION
# ==============================================================================

def analyze_aadhaar(image: np.ndarray) -> Dict:
    """
    Perform comprehensive Aadhaar card validation on an image.
    
    Args:
        image: BGR numpy array of the card image.
    
    Returns:
        Dict with keys:
            label: "REAL" | "FAKE" | "NOT_AADHAAR"
            confidence: float 0.0–1.0
            checks: Dict of individual check results
            explanation: str — human-readable summary
    """
    result = {
        "label": "NOT_AADHAAR",
        "confidence": 0.0,
        "checks": {},
        "explanation": ""
    }
    
    if image is None or image.size == 0:
        result["explanation"] = "Failed to load image."
        return result
    
    # ---- Run OCR once, share text across checks ----
    ocr_text = ""
    if HAS_TESSERACT:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            # Try multiple PSM modes for best extraction
            for psm in [6, 3, 4]:
                text = pytesseract.image_to_string(gray, config=f'--psm {psm}')
                if len(text.strip()) > len(ocr_text.strip()):
                    ocr_text = text
        except Exception as e:
            print(f"[WARN] OCR extraction failed: {e}")
    
    # ---- Run all checks ----
    checks = {}
    checks["aspect_ratio"] = check_aspect_ratio(image)
    checks["aadhaar_number"] = check_aadhaar_number(ocr_text)
    checks["qr_code"] = check_qr_code(image)
    checks["keywords"] = check_keyword_presence(ocr_text)
    checks["color_profile"] = check_color_profile(image)
    checks["ocr_confidence"] = check_ocr_confidence(image)
    checks["emblem_structure"] = check_emblem_structure(image)
    
    result["checks"] = checks
    
    # ---- Determine if this is even an Aadhaar card ----
    # Must have at least: Aadhaar number OR (keywords + aspect ratio)
    is_aadhaar_candidate = (
        checks["aadhaar_number"]["passed"] or
        (checks["keywords"]["passed"] and checks["aspect_ratio"]["passed"])
    )
    
    if not is_aadhaar_candidate:
        result["label"] = "NOT_AADHAAR"
        result["confidence"] = 0.0
        result["explanation"] = _build_explanation("NOT_AADHAAR", 0.0, checks)
        return result
    
    # ---- Weighted scoring for REAL vs FAKE ----
    weights = {
        "aadhaar_number": 0.25,
        "qr_code": 0.20,
        "keywords": 0.20,
        "ocr_confidence": 0.15,
        "color_profile": 0.10,
        "emblem_structure": 0.05,
        "aspect_ratio": 0.05,
    }
    
    weighted_score = 0.0
    total_weight = 0.0
    
    for check_name, weight in weights.items():
        check_result = checks.get(check_name, {})
        score = check_result.get("score", 0.0)
        if check_result.get("passed") is not None:  # Skip indeterminate checks
            weighted_score += score * weight
            total_weight += weight
    
    # Normalize
    if total_weight > 0:
        final_score = weighted_score / total_weight
    else:
        final_score = 0.0
    
    # ---- Apply critical failure penalties ----
    # If Aadhaar number is detected but Verhoeff fails, reduce score
    if checks["aadhaar_number"]["passed"] and "unverified" in checks["aadhaar_number"]["detail"].lower():
        final_score *= 0.85
    
    # ---- Final classification ----
    if final_score >= 0.55:
        result["label"] = "REAL"
        result["confidence"] = min(1.0, final_score)
    else:
        result["label"] = "FAKE"
        result["confidence"] = min(1.0, 1.0 - final_score)
    
    result["explanation"] = _build_explanation(result["label"], result["confidence"], checks)
    
    return result


def _build_explanation(label: str, confidence: float, checks: Dict) -> str:
    """Build a human-readable explanation from check results."""
    lines = []
    lines.append("═══ Aadhaar Card Verification Report ═══\n")
    
    if label == "NOT_AADHAAR":
        lines.append("⚠ The uploaded image does not appear to be an Aadhaar card.\n")
        lines.append("The system could not detect sufficient Aadhaar-specific")
        lines.append("features (number format, keywords, structure).\n")
        lines.append("Please upload a clear photo of an Aadhaar card.\n")
    else:
        lines.append(f"Verdict: {label}")
        lines.append(f"Confidence: {confidence:.1%}\n")
    
    lines.append("─── Individual Checks ───\n")
    
    check_labels = {
        "aadhaar_number": "Aadhaar Number Format",
        "qr_code": "QR Code Presence",
        "keywords": "Expected Keywords",
        "color_profile": "Color Profile",
        "ocr_confidence": "Print Quality (OCR)",
        "emblem_structure": "Emblem Structure",
        "aspect_ratio": "Aspect Ratio",
    }
    
    for key, display_name in check_labels.items():
        check = checks.get(key, {})
        passed = check.get("passed")
        detail = check.get("detail", "N/A")
        
        if passed is True:
            icon = "✓"
        elif passed is False:
            icon = "✗"
        else:
            icon = "?"
        
        lines.append(f"  {icon} {display_name}: {detail}")
    
    if label != "NOT_AADHAAR":
        lines.append("\n─── Note ───")
        lines.append("This analysis uses image-level heuristics.")
        lines.append("For authoritative verification, use the official")
        lines.append("UIDAI Aadhaar verification portal.")
    
    lines.append("\n\n⚖ SUDARSHAN is patented under Indian Patent Law.")
    
    return "\n".join(lines)
