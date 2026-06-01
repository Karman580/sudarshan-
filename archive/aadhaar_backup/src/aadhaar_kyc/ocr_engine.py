# ==============================================================================
# ocr_engine.py — Advanced Multi-Stage OCR Extraction for Aadhaar Cards
# ==============================================================================
# Uses a 4-stage preprocessing pipeline to maximise OCR reliability:
#   Stage 1: Original colour image
#   Stage 2: Grayscale + Gaussian blur
#   Stage 3: Adaptive threshold (binary)
#   Stage 4: 2× upscaled image
#
# Runs EasyOCR (preferred) with pytesseract fallback on EACH stage,
# then selects the result with the most extracted fields.
#
# Extracts:
#   - Aadhaar number (XXXX XXXX XXXX / XXXXXXXXXXXX / XXXX-XXXX-XXXX)
#   - Name
#   - Gender
#   - Date of Birth
#   - Address
#
# Weight in pipeline: 5%
# ==============================================================================

import re
import logging
import numpy as np
import cv2
from typing import Dict, Optional, List, Tuple

logger = logging.getLogger("aadhaar_kyc")

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False
    logger.warning("easyocr not installed — falling back to pytesseract.")

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

# Lazy-loaded reader singleton
_easyocr_reader = None

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
# Supports: "1234 5678 9012", "123456789012", "1234-5678-9012"
AADHAAR_PATTERN = re.compile(r'\b(\d{4}[\s\-]?\d{4}[\s\-]?\d{4})\b')
DOB_PATTERN = re.compile(r'\b(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})\b')
YEAR_PATTERN = re.compile(r'\b(19\d{2}|20[0-2]\d)\b')
GENDER_KEYWORDS = {"male", "female", "transgender", "/male", "/female"}

# Keywords that indicate text is NOT a name
NON_NAME_KEYWORDS = {
    "government", "india", "aadhaar", "uidai", "unique", "identification",
    "authority", "enrolment", "male", "female", "dob", "date", "birth",
    "year", "address", "vid", "download", "maadhaar", "help",
    "enrollment", "number",
}


def _get_easyocr_reader():
    """Lazy-load EasyOCR reader (one-time GPU/CPU init)."""
    global _easyocr_reader
    if _easyocr_reader is None and HAS_EASYOCR:
        try:
            _easyocr_reader = easyocr.Reader(
                ["en", "hi"],  # English + Hindi for Aadhaar
                gpu=False,
                verbose=False,
            )
            logger.info("EasyOCR reader initialised (en+hi).")
        except Exception as e:
            logger.error("EasyOCR init failed: %s", e)
    return _easyocr_reader


# ==============================================================================
# MULTI-STAGE PREPROCESSING
# ==============================================================================

def _generate_variants(image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Generate multiple preprocessed variants for robust OCR.

    Returns list of (variant_name, image) tuples.
    """
    variants = [("original", image)]

    # --- Stage 2: Grayscale + blur ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Convert back to 3-channel for EasyOCR compatibility
    gray_3ch = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    variants.append(("grayscale", gray_3ch))

    # --- Stage 3: Adaptive threshold ---
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2,
    )
    thresh_3ch = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    variants.append(("threshold", thresh_3ch))

    # --- Stage 4: 2× upscale ---
    h, w = image.shape[:2]
    if max(h, w) < 1200:
        upscaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        variants.append(("upscaled", upscaled))

    return variants


# ==============================================================================
# RAW TEXT EXTRACTION
# ==============================================================================

def _extract_text_easyocr(image: np.ndarray) -> Tuple[List[str], float]:
    """
    Extract text lines using EasyOCR.

    Returns (lines, average_confidence).
    """
    reader = _get_easyocr_reader()
    if reader is None:
        return [], 0.0

    try:
        results = reader.readtext(image, detail=1, paragraph=False)
        lines = []
        confidences = []
        for (bbox, text, conf) in results:
            text = str(text).strip()
            if text:
                lines.append(text)
                confidences.append(float(conf))

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return lines, avg_conf
    except Exception as e:
        logger.error("EasyOCR extraction failed: %s", e)
        return [], 0.0


def _extract_text_tesseract(image: np.ndarray) -> Tuple[List[str], float]:
    """
    Extract text lines using pytesseract.

    Returns (lines, average_confidence).
    """
    if not HAS_TESSERACT:
        return [], 0.0

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Try multiple PSM modes
        best_lines = []
        best_conf = 0.0

        for psm in [6, 3, 4]:
            text = pytesseract.image_to_string(gray, config=f"--psm {psm}")
            lines = [line.strip() for line in text.split("\n") if line.strip()]

            # Get confidence
            data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=f"--psm {psm}")
            confs = [int(c) for c in data["conf"] if str(c).lstrip("-").isdigit() and int(c) > 0]
            avg_conf = sum(confs) / len(confs) / 100.0 if confs else 0.0

            if len(lines) > len(best_lines) or (len(lines) == len(best_lines) and avg_conf > best_conf):
                best_lines = lines
                best_conf = avg_conf

        return best_lines, best_conf
    except Exception as e:
        logger.error("Tesseract extraction failed: %s", e)
        return [], 0.0


def _run_ocr_on_variant(image: np.ndarray) -> Tuple[List[str], float, str]:
    """
    Run OCR on a single image variant.

    Tries EasyOCR first, then Tesseract.
    Returns (lines, confidence, engine_name).
    """
    if HAS_EASYOCR:
        lines, conf = _extract_text_easyocr(image)
        if lines:
            return lines, conf, "easyocr"

    if HAS_TESSERACT:
        lines, conf = _extract_text_tesseract(image)
        if lines:
            return lines, conf, "tesseract"

    return [], 0.0, "none"


# ==============================================================================
# FIELD EXTRACTION
# ==============================================================================

def _normalize_text(text: str) -> str:
    """Normalize: collapse whitespace, strip."""
    return re.sub(r'\s+', ' ', text).strip()


def _extract_aadhaar_number(lines: List[str]) -> str:
    """
    Find the 12-digit Aadhaar number from OCR lines.

    Supports formats: XXXX XXXX XXXX, XXXXXXXXXXXX, XXXX-XXXX-XXXX
    """
    # Pass 1: regex match on each line
    for line in lines:
        match = AADHAAR_PATTERN.search(line)
        if match:
            raw = match.group(1)
            clean = re.sub(r'[\s\-]', '', raw)
            if len(clean) == 12 and clean.isdigit():
                return f"{clean[:4]} {clean[4:8]} {clean[8:12]}"

    # Pass 2: look for any 12-digit sequence in concatenated text
    full_text = " ".join(lines)
    # Remove common separators and find digit runs
    stripped = re.sub(r'[\s\-]', '', full_text)
    twelve_digit = re.findall(r'\d{12}', stripped)
    if twelve_digit:
        d = twelve_digit[0]
        return f"{d[:4]} {d[4:8]} {d[8:12]}"

    # Pass 3: search for partial patterns and reconstruct
    # Look for groups of 4 digits close together
    digit_groups = re.findall(r'\b\d{4}\b', full_text)
    if len(digit_groups) >= 3:
        candidate = "".join(digit_groups[:3])
        if len(candidate) == 12:
            return f"{candidate[:4]} {candidate[4:8]} {candidate[8:12]}"

    return ""


def _extract_gender(lines: List[str]) -> str:
    """Extract gender from OCR lines."""
    for line in lines:
        lower = line.lower()
        if "female" in lower or "/female" in lower:
            return "Female"
        if "male" in lower or "/male" in lower:
            return "Male"
        if "transgender" in lower:
            return "Transgender"
    return ""


def _extract_dob(lines: List[str]) -> str:
    """Extract date of birth or year of birth."""
    for line in lines:
        match = DOB_PATTERN.search(line)
        if match:
            return match.group(1)

    # Fallback: year of birth
    for line in lines:
        lower = line.lower()
        if "year" in lower or "yob" in lower or "birth" in lower:
            year_match = YEAR_PATTERN.search(line)
            if year_match:
                return year_match.group(1)

    # Last resort: any 4-digit year near a birth keyword
    full_text = " ".join(lines).lower()
    if "birth" in full_text or "dob" in full_text or "yob" in full_text:
        years = YEAR_PATTERN.findall(full_text)
        if years:
            return years[0]

    return ""


def _extract_name(lines: List[str]) -> str:
    """
    Heuristic name extraction.

    The name is usually the first substantial text line after header keywords
    that isn't a keyword itself, a number, or a date.
    """
    candidates = []
    for line in lines:
        lower = line.lower()

        # Skip lines that are clearly keywords or metadata
        if any(kw in lower for kw in NON_NAME_KEYWORDS):
            continue

        # Skip lines that are mostly digits
        alpha_count = sum(c.isalpha() for c in line)
        alpha_ratio = alpha_count / max(len(line), 1)
        if alpha_ratio < 0.5:
            continue

        # Skip very short lines
        if len(line) < 3 or alpha_count < 3:
            continue

        # Skip date-like lines
        if DOB_PATTERN.search(line):
            continue

        candidates.append(_normalize_text(line))

    return candidates[0] if candidates else ""


def _extract_address(lines: List[str]) -> str:
    """
    Extract address — usually multi-line text after 'Address' keyword.
    """
    collecting = False
    address_parts = []

    for line in lines:
        lower = line.lower()

        if "address" in lower or collecting:
            collecting = True
            # Stop at Aadhaar number or known end markers
            if AADHAAR_PATTERN.search(line):
                break
            clean = _normalize_text(
                line.replace("Address", "").replace("address", "").replace(":", "")
            )
            if clean and len(clean) > 2:
                address_parts.append(clean)

    return ", ".join(address_parts) if address_parts else ""


# ==============================================================================
# MULTI-STAGE OCR PIPELINE
# ==============================================================================

def _count_fields(fields: Dict) -> int:
    """Count how many useful fields were extracted."""
    count = 0
    if fields.get("aadhaar_number"):
        count += 2  # Weight number more heavily
    if fields.get("name"):
        count += 1
    if fields.get("gender"):
        count += 1
    if fields.get("dob"):
        count += 1
    if fields.get("address"):
        count += 1
    return count


def extract_fields(image: np.ndarray) -> Dict:
    """
    Run multi-stage OCR pipeline and extract structured Aadhaar fields.

    Runs OCR on 4 preprocessed variants and selects the result with
    the most extracted fields (preferring variants that found the
    Aadhaar number).

    Returns:
        Dict with aadhaar_number, name, gender, dob, address,
        raw_lines, engine, variant, confidence.
    """
    variants = _generate_variants(image)

    best_result = None
    best_field_count = -1
    best_confidence = 0.0
    all_lines_combined = []
    attempts_log = []

    for variant_name, variant_img in variants:
        lines, conf, engine = _run_ocr_on_variant(variant_img)

        if not lines:
            attempts_log.append(f"{variant_name}: no text")
            continue

        all_lines_combined.extend(lines)

        # Extract fields from this variant
        fields = {
            "aadhaar_number": _extract_aadhaar_number(lines),
            "name": _extract_name(lines),
            "gender": _extract_gender(lines),
            "dob": _extract_dob(lines),
            "address": _extract_address(lines),
            "raw_lines": lines,
            "engine": engine,
            "variant": variant_name,
            "confidence": conf,
        }

        field_count = _count_fields(fields)
        attempts_log.append(
            f"{variant_name}/{engine}: {len(lines)} lines, "
            f"{field_count} fields, conf={conf:.2f}"
        )

        # Select best result
        if field_count > best_field_count or (
            field_count == best_field_count and conf > best_confidence
        ):
            best_result = fields
            best_field_count = field_count
            best_confidence = conf

    logger.info("OCR multi-stage results: %s", " | ".join(attempts_log))

    # If no variant extracted an Aadhaar number, try from combined lines
    if best_result and not best_result.get("aadhaar_number"):
        combined_number = _extract_aadhaar_number(all_lines_combined)
        if combined_number:
            best_result["aadhaar_number"] = combined_number
            logger.info("Aadhaar number found via combined OCR lines.")

    # Fill any missing fields from combined lines
    if best_result:
        if not best_result.get("name"):
            best_result["name"] = _extract_name(all_lines_combined)
        if not best_result.get("gender"):
            best_result["gender"] = _extract_gender(all_lines_combined)
        if not best_result.get("dob"):
            best_result["dob"] = _extract_dob(all_lines_combined)

    if best_result is None:
        logger.warning("No OCR text extracted from any preprocessing variant.")
        return {
            "aadhaar_number": "",
            "name": "",
            "gender": "",
            "dob": "",
            "address": "",
            "raw_lines": [],
            "engine": "none",
            "variant": "none",
            "confidence": 0.0,
        }

    from src.aadhaar_kyc.utils import mask_aadhaar_number
    masked = mask_aadhaar_number(best_result["aadhaar_number"]) if best_result["aadhaar_number"] else "N/A"
    logger.info(
        "OCR best result [%s/%s]: number=%s, name=%s, gender=%s, dob=%s (conf=%.2f)",
        best_result["variant"], best_result["engine"],
        masked,
        best_result.get("name", "")[:20],
        best_result.get("gender", ""),
        best_result.get("dob", ""),
        best_result.get("confidence", 0.0),
    )

    return best_result


def run(image: np.ndarray, **ctx) -> dict:
    """Pipeline-compatible entry point."""
    fields = extract_fields(image)

    # Score: how many fields were extracted
    expected_fields = ["aadhaar_number", "name", "gender", "dob"]
    found = sum(1 for f in expected_fields if fields.get(f))
    score = found / len(expected_fields)

    # Boost score by OCR confidence
    ocr_conf = fields.get("confidence", 0.0)
    score = score * 0.8 + min(ocr_conf, 1.0) * 0.2

    detail_parts = []
    if fields["aadhaar_number"]:
        from src.aadhaar_kyc.utils import mask_aadhaar_number
        detail_parts.append(f"Aadhaar: {mask_aadhaar_number(fields['aadhaar_number'])}")
    if fields["name"]:
        detail_parts.append(f"Name: {fields['name'][:20]}")
    if fields["gender"]:
        detail_parts.append(f"Gender: {fields['gender']}")
    if fields["dob"]:
        detail_parts.append(f"DOB: {fields['dob']}")
    if not detail_parts:
        detail_parts.append("No fields extracted")
    detail_parts.append(f"[{fields.get('variant', '?')}/{fields.get('engine', '?')}]")

    detail = " | ".join(detail_parts)

    return {
        "passed": score >= 0.25,
        "score": score,
        "detail": detail,
        "data": fields,
    }
