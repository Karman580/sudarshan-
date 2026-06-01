# ==============================================================================
# pipeline.py — Aadhaar KYC Multi-Layer Verification Pipeline
# ==============================================================================
# Orchestrates all 13 verification modules in sequence:
#
#   1. Load image → resize
#   2. Perspective correction
#   3. YOLO card detection        (15%)
#   4. OCR extraction              (5%)
#   5. Aadhaar number validation   (5%)
#   6. QR code verification       (30%)
#   7. Layout validation          (10%)
#   8. Logo/emblem detection       (5%)
#   9. Face detection              (5%)
#  10. Selfie matching            (10%)  — only if selfie provided
#  11. AI-fake detection           (5%)
#  12. Screen detection            (3%)
#  13. Tamper detection            (5%)
#  14. Colour profile              (2%)
#
# Final: weighted score → REAL if ≥ 0.75
#
# Every module is wrapped in try/except for crash-proof operation.
# Failed modules return neutral score (0.5) and are logged clearly.
#
# Patented under Indian Patent Law.
# Developed @ Thapar Institute of Engineering & Technology, Patiala
# ==============================================================================

import time
import logging
import traceback
import numpy as np
from typing import Dict, Optional

from src.aadhaar_kyc.utils import (
    load_image,
    resize_for_processing,
    mask_aadhaar_number,
)
from src.aadhaar_kyc import perspective_correction
from src.aadhaar_kyc import yolo_detector
from src.aadhaar_kyc import ocr_engine
from src.aadhaar_kyc import number_validator
from src.aadhaar_kyc import qr_validator
from src.aadhaar_kyc import layout_validator
from src.aadhaar_kyc import logo_detector
from src.aadhaar_kyc import face_verification
from src.aadhaar_kyc import ai_fake_detector
from src.aadhaar_kyc import screen_detector
from src.aadhaar_kyc import tamper_detection
from src.aadhaar_kyc import color_validator
from src.aadhaar_kyc.visual_explainer import generate_visual_explanation
import os
import cv2

logger = logging.getLogger("aadhaar_kyc")

# ==============================================================================
# MODULE WEIGHTS (must sum to 1.0 when selfie is NOT provided)
# ==============================================================================

WEIGHTS_NO_SELFIE = {
    "yolo_detector":     0.15,
    "ocr_engine":        0.05,
    "number_validator":  0.05,
    "qr_validator":      0.30,
    "layout_validator":  0.10,
    "logo_detector":     0.05,
    "face_verification": 0.05,
    "ai_fake_detector":  0.05,
    "screen_detector":   0.03,
    "tamper_detection":  0.05,
    "color_validator":   0.02,
    # Remaining 10% (selfie_match) is redistributed below
}

WEIGHTS_WITH_SELFIE = {
    "yolo_detector":     0.15,
    "ocr_engine":        0.05,
    "number_validator":  0.05,
    "qr_validator":      0.30,
    "layout_validator":  0.10,
    "logo_detector":     0.05,
    "face_verification": 0.05,
    "selfie_match":      0.10,
    "ai_fake_detector":  0.05,
    "screen_detector":   0.03,
    "tamper_detection":  0.05,
    "color_validator":   0.02,
}

# Redistribute selfie's 10% weight when selfie is absent
WEIGHTS_NO_SELFIE["qr_validator"] += 0.05
WEIGHTS_NO_SELFIE["layout_validator"] += 0.03
WEIGHTS_NO_SELFIE["face_verification"] += 0.02

# REAL threshold
REAL_THRESHOLD = 0.75


# ==============================================================================
# SAFE MODULE RUNNER
# ==============================================================================

def _safe_run(module_name: str, func, image: np.ndarray, **ctx) -> dict:
    """
    Run a module with crash-proof exception handling.

    If a module raises ANY exception:
      - Logs the full traceback
      - Returns neutral score (0.5) with passed=None
      - Does NOT crash the pipeline
    """
    try:
        result = func(image, **ctx)
        if not isinstance(result, dict):
            logger.error(
                "Module '%s' returned %s instead of dict — treating as neutral.",
                module_name, type(result).__name__,
            )
            return _neutral_result(module_name, f"Bad return type: {type(result).__name__}")
        return result
    except Exception as e:
        logger.error(
            "Module '%s' CRASHED: %s\n%s",
            module_name, e, traceback.format_exc(),
        )
        return _neutral_result(module_name, f"Module error: {e}")


def _neutral_result(module_name: str, detail: str) -> dict:
    """Return a neutral/indeterminate result for a failed module."""
    return {
        "passed": None,
        "score": 0.5,
        "detail": detail,
        "data": {},
    }


# ==============================================================================
# MAIN ANALYSIS FUNCTION
# ==============================================================================

def analyze(
    image_path: str,
    selfie_path: Optional[str] = None,
) -> Dict:
    """
    Run the complete Aadhaar KYC multi-layer verification pipeline.

    Args:
        image_path: Path to the Aadhaar card image.
        selfie_path: Optional path to a live selfie for face matching.

    Returns:
        Structured dict with classification, scores, extracted data,
        and detailed module results.
    """
    t0 = time.time()
    result = {
        "is_aadhaar_card": False,
        "classification": "ERROR",
        "confidence_score": 0.0,
        "confidence_band": "LOW",
        "aadhaar_number": "",
        "qr_data": None,
        "ocr_data": None,
        "face_detected": False,
        "tamper_detected": False,
        "screen_detected": False,
        "ai_generated_probability": 0.0,
        "module_scores": {},
        "explanation": "Pipeline crashed",
        "annotated_image_path": None,
        "processing_time_ms": 0,
        "risk_flags": []
    }

    try:
        # --- Initialise internal state ---
        pass
    
        # --- Step 1: Load image ---
        image = load_image(image_path)
        if image is None:
            result["explanation"] = "Failed to load image file."
            logger.error("Pipeline aborted: image load failed for %s", image_path)
            return result
    
        image = resize_for_processing(image, max_dim=1280)
        logger.info("Pipeline started: image %dx%d from %s", image.shape[1], image.shape[0], image_path)
    
        # --- Step 2: Perspective correction ---
        pc_result = _safe_run("perspective_correction", perspective_correction.run, image)
        corrected_image = pc_result.get("data", {}).get("corrected_image", image)
        img = corrected_image
    
        # --- Step 3: YOLO / contour detection ---
        yolo_result = _safe_run("yolo_detector", yolo_detector.run, img)
        yolo_detections = yolo_result.get("data", {}).get("detections", [])
        card_detected = yolo_result.get("data", {}).get("card_detected")
    
        # --- Step 4: OCR extraction ---
        ocr_result = _safe_run("ocr_engine", ocr_engine.run, img)
        ocr_data = ocr_result.get("data", {})
    
        # --- Step 5: Number validation ---
        num_result = _safe_run("number_validator", number_validator.run, img, ocr_data=ocr_data)
    
        # --- Step 6: QR code verification (uses OCR data for cross-validation) ---
        qr_result = _safe_run("qr_validator", qr_validator.run, img, ocr_data=ocr_data)
        qr_data = qr_result.get("data", {}).get("qr_data")
    
        # --- Step 7: Layout validation (uses YOLO detections) ---
        layout_result = _safe_run("layout_validator", layout_validator.run, img, yolo_detections=yolo_detections)
    
        # --- Step 8: Logo detection ---
        logo_result = _safe_run("logo_detector", logo_detector.run, img)
    
        # --- Step 9: Face detection ---
        face_result = _safe_run("face_verification", face_verification.run, img)
        face_detected = face_result.get("data", {}).get("face_detected", False)
    
        # --- Step 10: Selfie match (optional) ---
        selfie_result = None
        has_selfie = False
        if selfie_path:
            selfie_img = load_image(selfie_path)
            if selfie_img is not None:
                has_selfie = True
                selfie_result = _safe_run(
                    "selfie_match", face_verification.run_selfie,
                    img, selfie_image=selfie_img,
                )
                logger.info("Selfie match executed.")
            else:
                logger.warning("Selfie image could not be loaded: %s", selfie_path)
    
        # --- Step 11: AI-fake detection ---
        ai_result = _safe_run("ai_fake_detector", ai_fake_detector.run, img)
        ai_prob = ai_result.get("data", {}).get("ai_generated_probability", 0.0)
    
        # --- Step 12: Screen detection ---
        screen_result = _safe_run("screen_detector", screen_detector.run, img)
        screen_detected = screen_result.get("data", {}).get("screen_detected", False)
    
        # --- Step 13: Tamper detection ---
        tamper_result = _safe_run("tamper_detection", tamper_detection.run, img)
        tamper_detected = tamper_result.get("data", {}).get("tamper_detected", False)
    
        # --- Step 14: Colour validation ---
        color_result = _safe_run("color_validator", color_validator.run, img)
    
        # ==================================================================
        # WEIGHTED SCORING
        # ==================================================================
    
        module_results = {
            "yolo_detector":     yolo_result,
            "ocr_engine":        ocr_result,
            "number_validator":  num_result,
            "qr_validator":      qr_result,
            "layout_validator":  layout_result,
            "logo_detector":     logo_result,
            "face_verification": face_result,
            "ai_fake_detector":  ai_result,
            "screen_detector":   screen_result,
            "tamper_detection":  tamper_result,
            "color_validator":   color_result,
        }
    
        if has_selfie and selfie_result:
            module_results["selfie_match"] = selfie_result
    
        weights = WEIGHTS_WITH_SELFIE if has_selfie else WEIGHTS_NO_SELFIE
    
        weighted_sum = 0.0
        total_weight = 0.0
        total_modules = 0
        passed_modules = 0
        critical_failed = False
        module_scores = {}
    
        for mod_name, mod_result in module_results.items():
            weight = weights.get(mod_name, 0.0)
            score = mod_result.get("score", 0.5)
            passed = mod_result.get("passed")
    
            print(f"[PIPELINE] {mod_name} passed: {passed} | score: {score:.2f}")
    
            module_scores[mod_name] = {
                "score": round(score, 4),
                "weight": weight,
                "weighted": round(score * weight, 4),
                "passed": passed,
                "detail": mod_result.get("detail", ""),
            }
    
            # All modules contribute to the weighted score
            # Indeterminate (passed=None) modules still contribute their score
            weighted_sum += score * weight
            total_weight += weight
            
            total_modules += 1
            if passed is True:
                passed_modules += 1
                
            if mod_name in ["qr_validator", "ocr_engine", "number_validator"] and passed is False:
                critical_failed = True
    
        base_score = weighted_sum / max(total_weight, 0.01)
        
        # Add Soft Aggregation
        final_score = 0.7 * base_score + 0.3 * (passed_modules / max(total_modules, 1))
    
        # Add Critical Weighting Penalty
        if critical_failed:
            final_score *= 0.7
    
        # --- Determine if this looks like an Aadhaar card at all ---
        aadhaar_number = ocr_data.get("aadhaar_number", "")
        has_number = bool(aadhaar_number)
        has_ocr = ocr_result.get("passed", False)
        has_layout = layout_result.get("passed", False)
        has_card = card_detected is True or card_detected is None  # Include neutral YOLO
    
        # More relaxed Aadhaar detection: any strong signal counts
        is_aadhaar = has_number or (has_ocr and has_layout) or (has_card and has_layout)
    
        # --- Classification ---
        if not is_aadhaar:
            classification = "NOT_AADHAAR"
            confidence = 0.0
            confidence_band = "LOW"
        elif final_score >= 0.75:
            classification = "REAL"
            confidence = final_score
            confidence_band = "HIGH"
        elif final_score >= 0.55:
            classification = "SUSPICIOUS"
            confidence = final_score
            confidence_band = "MEDIUM"
        else:
            classification = "FAKE"
            confidence = final_score
            confidence_band = "LOW"
    
        print(f"[PIPELINE] SCORE: {final_score}")
    
        # --- Generate Annotations (XAI) ---
        try:
            annotated_img = generate_visual_explanation(img, module_results)
            os.makedirs("outputs", exist_ok=True)
            annotated_image_path = "outputs/annotated_aadhaar.jpg"
            cv2.imwrite(annotated_image_path, annotated_img)
        except Exception as e:
            print("[ERROR] Image save failed:", e)
            annotated_image_path = None
        
        # --- Generate Risk Flags & Expanation ---
        risk_flags = []
        failed_modules = []
        
        if module_scores.get("qr_validator", {}).get("passed") is False:
            risk_flags.append("QR not detected")
            failed_modules.append("QR code not detected")
            print("[PIPELINE] QR: Failed")
        else:
            print("[PIPELINE] QR: Passed")
            
        if module_scores.get("ocr_engine", {}).get("passed") is False:
            risk_flags.append("OCR weak")
            failed_modules.append("OCR failed")
            print("[PIPELINE] OCR: Failed")
        else:
            print("[PIPELINE] OCR: Passed")
            
        if module_scores.get("face_verification", {}).get("passed") is False:
            risk_flags.append("Face misaligned")
            failed_modules.append("Face not in correct position")
            
        if module_scores.get("layout_validator", {}).get("passed") is False:
            failed_modules.append("Layout mismatch")
    
        # Build dynamic explanation
        if failed_modules:
            explanation = ". ".join(failed_modules)
        else:
            explanation = _build_explanation(classification, final_score, module_scores)
            
        # Fallback explanation if somehow empty
        if not explanation or len(explanation.strip()) == 0:
            explanation = "System could not fully analyze all features. Some modules failed or returned low confidence."
    
        elapsed_ms = int((time.time() - t0) * 1000)
    
        # --- Populate result ---
        result["is_aadhaar_card"] = is_aadhaar
        result["classification"] = classification
        result["confidence_score"] = round(confidence, 4)
        result["confidence_band"] = confidence_band
        result["annotated_image_path"] = annotated_image_path
        result["risk_flags"] = risk_flags
        result["aadhaar_number"] = mask_aadhaar_number(aadhaar_number) if aadhaar_number else ""
        result["qr_data"] = qr_data
        result["ocr_data"] = {
            k: v for k, v in ocr_data.items()
            if k not in ("raw_lines",)  # Exclude raw OCR lines from output
        } if ocr_data else None
        result["face_detected"] = face_detected
        result["tamper_detected"] = tamper_detected
        result["screen_detected"] = screen_detected
        result["ai_generated_probability"] = round(ai_prob, 4)
        result["module_scores"] = module_scores
        result["explanation"] = explanation
        result["processing_time_ms"] = elapsed_ms
    
        logger.info(
            "Pipeline complete: %s (score=%.3f, time=%dms, modules=%d)",
            classification, final_score, elapsed_ms, len(module_scores),
        )
    
        return result
    except Exception as e:
        print("[FATAL ERROR]", e)
        return {
            "classification": "ERROR",
            "confidence_score": 0.5,
            "explanation": "System fallback triggered",
            "annotated_image_path": image_path
        }


# ==============================================================================
# EXPLANATION BUILDER
# ==============================================================================

DISPLAY_NAMES = {
    "yolo_detector": "Card Detection",
    "ocr_engine": "OCR Extraction",
    "number_validator": "Aadhaar Number",
    "qr_validator": "QR Code Verification",
    "layout_validator": "Layout Structure",
    "logo_detector": "Logo/Emblem Detection",
    "face_verification": "Face Detection",
    "selfie_match": "Selfie Match",
    "ai_fake_detector": "AI-Fake Detection",
    "screen_detector": "Screen Detection",
    "tamper_detection": "Tamper Detection",
    "color_validator": "Colour Profile",
}


def _build_explanation(classification: str, score: float, module_scores: Dict) -> str:
    """Build a human-readable verification report with per-module diagnostics."""
    lines = []
    lines.append("═══ Aadhaar KYC Verification Report ═══\n")

    if classification == "NOT_AADHAAR":
        lines.append("⚠ The uploaded image does not appear to be an Aadhaar card.\n")
        lines.append("The system could not detect sufficient Aadhaar-specific")
        lines.append("features (number format, layout, card structure).\n")
        lines.append("Please upload a clear photo of an Aadhaar card.\n")
    else:
        verdict_icon = "✓" if classification == "REAL" else "✗"
        lines.append(f"Verdict: {verdict_icon} {classification}")
        lines.append(f"Pipeline Score: {score:.1%}\n")

    lines.append("─── Module Results ───\n")

    for key, info in module_scores.items():
        name = DISPLAY_NAMES.get(key, key)
        passed = info.get("passed")
        detail = info.get("detail", "")
        weight_pct = info.get("weight", 0) * 100
        mod_score = info.get("score", 0)

        if passed is True:
            icon = "✓"
        elif passed is False:
            icon = "✗"
        else:
            icon = "?"

        # Show score as a percentage for clarity
        lines.append(f"  {icon} {name} [{weight_pct:.0f}%] (score: {mod_score:.0%})")
        if detail:
            # Indent detail under the module name
            lines.append(f"      {detail}")

    # Summary of failures
    failed_modules = [
        DISPLAY_NAMES.get(k, k)
        for k, v in module_scores.items()
        if v.get("passed") is False
    ]
    if failed_modules and classification != "NOT_AADHAAR":
        lines.append(f"\n─── Failed Checks ({len(failed_modules)}) ───")
        for fm in failed_modules:
            lines.append(f"  ✗ {fm}")

    if classification != "NOT_AADHAAR":
        lines.append("\n─── Note ───")
        lines.append("This analysis combines 13 verification layers.")
        lines.append("For authoritative verification, use the official")
        lines.append("UIDAI Aadhaar verification portal.")

    lines.append("\n\n⚖ SUDARSHAN is patented under Indian Patent Law.")

    return "\n".join(lines)
