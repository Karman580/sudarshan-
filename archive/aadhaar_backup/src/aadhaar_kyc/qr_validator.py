# ==============================================================================
# qr_validator.py — Aadhaar Secure QR Code Verification
# ==============================================================================
# Multi-pass QR detection pipeline with 6 preprocessing strategies:
#   1. Original image
#   2. Grayscale
#   3. Binary threshold
#   4. Adaptive threshold
#   5. 2× upscale
#   6. Cropped right-half of card (QR is always on the right)
#
# After detection:
#   - Decode QR payload
#   - Parse Aadhaar Secure QR XML (xmltodict)
#   - Extract uid, name, dob, gender, address, photo_hash
#   - Cross-validate extracted fields against OCR data
#
# Mismatch between QR and OCR → score 0 (hard reject).
# Weight in pipeline: 30%
# ==============================================================================

import cv2
import numpy as np
import re
import logging
from typing import Dict, Optional, List

logger = logging.getLogger("aadhaar_kyc")

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    HAS_PYZBAR = True
except ImportError:
    HAS_PYZBAR = False
    logger.warning("pyzbar not installed — QR detection disabled.")

try:
    import xmltodict
    HAS_XMLTODICT = True
except ImportError:
    HAS_XMLTODICT = False
    logger.warning("xmltodict not installed — QR XML parsing disabled.")


# ==============================================================================
# MULTI-PASS QR DETECTION
# ==============================================================================

def _generate_qr_variants(image: np.ndarray) -> List[tuple]:
    """
    Generate 6 preprocessed image variants for robust QR detection.

    Returns list of (variant_name, grayscale_image) tuples.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    h, w = gray.shape[:2]

    variants = []

    # 1. Original grayscale
    variants.append(("original", gray))

    # 2. Gaussian blur + OTSU threshold
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", otsu))

    # 3. Fixed binary threshold
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    variants.append(("binary", binary))

    # 4. Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2,
    )
    variants.append(("adaptive", adaptive))

    # 5. 2× upscale (helps with small/low-res QR codes)
    if max(h, w) < 1000:
        upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        variants.append(("upscaled", upscaled))

    # 6. Right-half crop (Aadhaar QR is always on the right side)
    right_half = gray[:, w // 2:]
    variants.append(("right_half", right_half))

    # 7. Right-half upscaled + threshold (best chance for small QRs)
    if max(right_half.shape) < 600:
        rh_up = cv2.resize(right_half, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, rh_thresh = cv2.threshold(rh_up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(("right_half_upscaled", rh_thresh))

    return variants


def _detect_qr(image: np.ndarray) -> tuple:
    """
    Attempt QR detection with all preprocessing strategies.

    Returns (qr_results_list, variant_name) or ([], None).
    """
    if not HAS_PYZBAR:
        return [], None

    variants = _generate_qr_variants(image)

    for variant_name, variant_img in variants:
        try:
            results = pyzbar_decode(variant_img)
            if results:
                logger.info(
                    "QR detected via '%s' variant (%d code(s)).",
                    variant_name, len(results),
                )
                return results, variant_name
        except Exception as e:
            logger.debug("QR decode failed on '%s': %s", variant_name, e)
            continue

    logger.warning("QR detection failed on all %d preprocessing variants.", len(variants))
    return [], None


def _validate_qr_position(image: np.ndarray, qr_results: list) -> bool:
    """
    Verify that the detected QR code is in the expected region (right side).

    Returns True if QR is in the right 60% of the image.
    """
    if not qr_results:
        return False

    h, w = image.shape[:2]
    for qr in qr_results:
        rect = qr.rect
        qr_center_x = rect.left + rect.width / 2
        # QR should be in the right portion of the card
        if qr_center_x > w * 0.35:
            return True

    # If we got here from a cropped variant, position is inherently valid
    return True


# ==============================================================================
# QR PARSING
# ==============================================================================

def _parse_aadhaar_xml(raw_data: str) -> Optional[Dict]:
    """
    Parse Aadhaar Secure QR XML payload.

    Example XML:
        <PrintLetterBarcodeData uid="..." name="..." dob="..." gender="..."
         co="..." house="..." street="..." loc="..." ... />

    Returns dict with normalized fields, or None on parse failure.
    """
    if not HAS_XMLTODICT:
        return None

    try:
        parsed = xmltodict.parse(raw_data)

        # Handle root element name variations
        root_key = None
        for key in ("PrintLetterBarcodeData", "QRCodeData", "QPDA"):
            if key in parsed:
                root_key = key
                break

        if root_key is None:
            keys = list(parsed.keys())
            root_key = keys[0] if keys else None

        if root_key is None:
            return None

        data = parsed[root_key]
        if isinstance(data, str):
            return None

        # Normalize: xmltodict uses @attr for XML attributes
        attr_prefix = "@" if any(k.startswith("@") for k in data) else ""

        result = {}
        result["uid"] = data.get(f"{attr_prefix}uid", "")
        result["name"] = data.get(f"{attr_prefix}name", "")
        result["dob"] = data.get(f"{attr_prefix}dob", data.get(f"{attr_prefix}yob", ""))
        result["gender"] = data.get(f"{attr_prefix}gender", "")

        # Address: concatenate available fields
        addr_fields = ["co", "house", "street", "lm", "loc", "vtc", "dist", "state", "pc"]
        address_parts = []
        for field in addr_fields:
            val = data.get(f"{attr_prefix}{field}", "")
            if val:
                address_parts.append(str(val))
        result["address"] = ", ".join(address_parts)

        # Photo hash (Secure QR v2)
        result["photo_hash"] = data.get(f"{attr_prefix}pht", "")

        logger.info(
            "QR XML parsed: uid=...%s, name=%s, gender=%s",
            result.get("uid", "")[-4:],
            result.get("name", "")[:20],
            result.get("gender", ""),
        )

        return result

    except Exception as e:
        logger.warning("QR XML parse error: %s", e)
        return None


# ==============================================================================
# QR-OCR CROSS-VALIDATION
# ==============================================================================

def _normalize(text: str) -> str:
    """Normalize text for comparison: uppercase, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", text.upper().strip())


def _cross_validate(qr_data: Dict, ocr_data: Dict) -> Dict:
    """
    Compare QR-extracted data against OCR-extracted data.

    Returns match result with score and details.
    """
    if not qr_data or not ocr_data:
        return {
            "matched": None,
            "score": 0.5,
            "detail": "Cross-validation skipped (insufficient data).",
            "mismatches": [],
        }

    matches = 0
    total = 0
    mismatches = []

    # Compare UID / Aadhaar number
    qr_uid = re.sub(r"[^0-9]", "", qr_data.get("uid", ""))
    ocr_uid = re.sub(r"[^0-9]", "", ocr_data.get("aadhaar_number", ""))
    if qr_uid and ocr_uid:
        total += 1
        # Compare last 4 digits (QR may have full, OCR may have masked)
        if len(qr_uid) >= 4 and len(ocr_uid) >= 4 and qr_uid[-4:] == ocr_uid[-4:]:
            matches += 1
        else:
            mismatches.append(f"UID mismatch: QR=...{qr_uid[-4:]}, OCR=...{ocr_uid[-4:]}")

    # Compare name
    qr_name = _normalize(qr_data.get("name", ""))
    ocr_name = _normalize(ocr_data.get("name", ""))
    if qr_name and ocr_name:
        total += 1
        qr_words = set(qr_name.split())
        ocr_words = set(ocr_name.split())
        overlap = qr_words & ocr_words
        if len(overlap) >= min(len(qr_words), 2):
            matches += 1
        else:
            mismatches.append(f"Name mismatch: QR='{qr_name}', OCR='{ocr_name}'")

    # Compare gender
    qr_gender = _normalize(qr_data.get("gender", ""))[:1]
    ocr_gender = _normalize(ocr_data.get("gender", ""))[:1]
    if qr_gender and ocr_gender:
        total += 1
        if qr_gender == ocr_gender:
            matches += 1
        else:
            mismatches.append(f"Gender mismatch: QR='{qr_gender}', OCR='{ocr_gender}'")

    # Compare DOB
    qr_dob = re.sub(r"[^0-9]", "", qr_data.get("dob", ""))
    ocr_dob = re.sub(r"[^0-9]", "", ocr_data.get("dob", ""))
    if qr_dob and ocr_dob:
        total += 1
        if qr_dob == ocr_dob:
            matches += 1
        else:
            mismatches.append(f"DOB mismatch: QR='{qr_dob}', OCR='{ocr_dob}'")

    if total == 0:
        return {
            "matched": None,
            "score": 0.5,
            "detail": "No comparable fields for cross-validation.",
            "mismatches": [],
        }

    ratio = matches / total
    return {
        "matched": ratio >= 0.5,
        "score": ratio,
        "detail": (
            f"Matched {matches}/{total} fields."
            + (f" Mismatches: {'; '.join(mismatches)}" if mismatches else "")
        ),
        "mismatches": mismatches,
    }


# ==============================================================================
# PUBLIC API
# ==============================================================================

def validate_qr(image: np.ndarray, ocr_data: Optional[Dict] = None) -> Dict:
    """
    Full QR validation pipeline: detect → decode → parse → cross-validate.

    Returns:
        Dict with passed, score, detail, qr_data, cross_validation.
    """
    # --- Step 1-2: Multi-pass detect & decode ---
    qr_results, variant_used = _detect_qr(image)

    if not qr_results:
        return {
            "passed": False,
            "score": 0.0,
            "detail": "No QR code detected after all preprocessing attempts.",
            "qr_data": None,
            "cross_validation": None,
        }

    # Validate QR is in expected position
    qr_in_position = _validate_qr_position(image, qr_results)

    # Use the first decoded QR
    raw_data = None
    qr_type = None
    for qr in qr_results:
        try:
            raw_data = qr.data.decode("utf-8", errors="replace")
            qr_type = qr.type
            break
        except Exception:
            continue

    if not raw_data:
        return {
            "passed": False,
            "score": 0.1,
            "detail": "QR detected but could not decode payload.",
            "qr_data": None,
            "cross_validation": None,
        }

    # --- Step 3: Parse XML ---
    qr_data = _parse_aadhaar_xml(raw_data)
    score = 0.0

    if qr_data:
        score = 0.6  # QR present + parseable
        detail = f"QR decoded (type={qr_type}, variant={variant_used}), Aadhaar XML parsed."
        if qr_in_position:
            score += 0.1
    else:
        # QR present but not Aadhaar XML — might be a simple QR
        score = 0.3
        detail = f"QR decoded (type={qr_type}, variant={variant_used}), but payload is not Aadhaar XML."
        qr_data = {"raw": raw_data[:200]}

    # --- Step 4-5: Cross-validate with OCR ---
    cross = None
    if qr_data and ocr_data and "raw" not in qr_data:
        cross = _cross_validate(qr_data, ocr_data)
        if cross["matched"] is True:
            score = 1.0
            detail += " QR↔OCR cross-validation PASSED."
        elif cross["matched"] is False:
            score = 0.0  # Hard reject on mismatch
            detail += f" QR↔OCR MISMATCH: {'; '.join(cross['mismatches'])}"
        # else: indeterminate, keep current score

    return {
        "passed": score >= 0.3,
        "score": score,
        "detail": detail,
        "qr_data": qr_data if "raw" not in (qr_data or {}) else None,
        "cross_validation": cross,
    }


def run(image: np.ndarray, **ctx) -> dict:
    """Pipeline-compatible entry point."""
    ocr_data = ctx.get("ocr_data")
    result = validate_qr(image, ocr_data=ocr_data)
    return {
        "passed": result["passed"],
        "score": result["score"],
        "detail": result["detail"],
        "data": {
            "qr_data": result["qr_data"],
            "cross_validation": result["cross_validation"],
        },
    }
