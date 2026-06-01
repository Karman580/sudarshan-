# ==============================================================================
# number_validator.py — Strict Aadhaar Number Validation
# ==============================================================================
# Rules:
#   1. Must be exactly 12 digits
#   2. Must pass Verhoeff checksum
#   3. Reject trivial patterns (all-same, sequential, etc.)
#
# Weight in pipeline: 5%
# ==============================================================================

import re
import logging
import numpy as np
from typing import Dict

from src.aadhaar_kyc.utils import validate_verhoeff, mask_aadhaar_number

logger = logging.getLogger("aadhaar_kyc")

# Patterns guaranteed to be invalid Aadhaar numbers
REJECTED_PATTERNS = {
    "000000000000",
    "111111111111",
    "222222222222",
    "333333333333",
    "444444444444",
    "555555555555",
    "666666666666",
    "777777777777",
    "888888888888",
    "999999999999",
    "123456789012",
    "012345678901",
    "987654321098",
    "098765432109",
}


def validate_aadhaar_number(number: str) -> Dict:
    """
    Validate an extracted Aadhaar number string.

    Args:
        number: Raw Aadhaar number string (may contain spaces/dashes).

    Returns:
        Dict with passed, score, detail.
    """
    if not number:
        return {
            "passed": False,
            "score": 0.0,
            "detail": "No Aadhaar number provided.",
        }

    clean = re.sub(r"[^0-9]", "", number)

    # --- Length check ---
    if len(clean) != 12:
        return {
            "passed": False,
            "score": 0.0,
            "detail": f"Number has {len(clean)} digits (expected 12).",
        }

    # --- Pattern rejection ---
    if clean in REJECTED_PATTERNS:
        return {
            "passed": False,
            "score": 0.0,
            "detail": f"Rejected trivial pattern: {mask_aadhaar_number(clean)}.",
        }

    # First digit must not be 0 or 1 (UIDAI rule)
    if clean[0] in ("0", "1"):
        return {
            "passed": False,
            "score": 0.1,
            "detail": f"Aadhaar numbers cannot start with {clean[0]}.",
        }

    # --- Verhoeff checksum ---
    if validate_verhoeff(clean):
        logger.info("Aadhaar number valid: %s (Verhoeff ✓)", mask_aadhaar_number(clean))
        return {
            "passed": True,
            "score": 1.0,
            "detail": f"Valid: {mask_aadhaar_number(clean)} (Verhoeff ✓).",
        }
    else:
        logger.warning("Aadhaar number Verhoeff failed: %s", mask_aadhaar_number(clean))
        return {
            "passed": False,
            "score": 0.3,
            "detail": f"Verhoeff checksum failed for {mask_aadhaar_number(clean)}.",
        }


def run(image: np.ndarray, **ctx) -> dict:
    """
    Pipeline-compatible entry point.

    Expects ctx["ocr_data"]["aadhaar_number"] to be populated
    by the OCR engine before this module runs.
    """
    ocr_data = ctx.get("ocr_data", {})
    number = ocr_data.get("aadhaar_number", "")

    result = validate_aadhaar_number(number)
    return {
        "passed": result["passed"],
        "score": result["score"],
        "detail": result["detail"],
        "data": {"clean_number": re.sub(r"[^0-9]", "", number) if number else ""},
    }
