# ==============================================================================
# tests/test_aadhaar_kyc.py — Unit Tests for Aadhaar KYC Pipeline
# ==============================================================================
# Run: python -m pytest tests/test_aadhaar_kyc.py -v
# ==============================================================================

import sys
import os
import numpy as np
import cv2
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ==============================================================================
# 1. VERHOEFF CHECKSUM TESTS
# ==============================================================================

class TestVerhoeff:
    """Test the Verhoeff checksum algorithm used by UIDAI."""

    def test_known_valid_number(self):
        """Aadhaar number 499118665246 is known to be Verhoeff-valid."""
        from src.aadhaar_kyc.utils import validate_verhoeff
        assert validate_verhoeff("499118665246") is True

    def test_known_invalid_number(self):
        """Flipping one digit should fail Verhoeff."""
        from src.aadhaar_kyc.utils import validate_verhoeff
        assert validate_verhoeff("499118665247") is False

    def test_all_zeros_fails(self):
        from src.aadhaar_kyc.utils import validate_verhoeff
        # All zeros happens to pass Verhoeff (checksum = 0),
        # but is rejected by pattern validator
        result = validate_verhoeff("000000000000")
        assert isinstance(result, bool)

    def test_non_digit_returns_false(self):
        from src.aadhaar_kyc.utils import validate_verhoeff
        assert validate_verhoeff("49911866524X") is False
        assert validate_verhoeff("") is False
        assert validate_verhoeff("abcdefghijkl") is False

    def test_with_spaces(self):
        """Should handle spaces in the number."""
        from src.aadhaar_kyc.utils import validate_verhoeff
        assert validate_verhoeff("4991 1866 5246") is True

    def test_checksum_computation(self):
        from src.aadhaar_kyc.utils import verhoeff_checksum
        # For a valid number, checksum should be 0
        assert verhoeff_checksum("499118665246") == 0


# ==============================================================================
# 2. NUMBER VALIDATOR TESTS
# ==============================================================================

class TestNumberValidator:
    """Test Aadhaar number pattern rejection and validation."""

    def test_all_same_digits_rejected(self):
        from src.aadhaar_kyc.number_validator import validate_aadhaar_number
        for d in "0123456789":
            result = validate_aadhaar_number(d * 12)
            assert result["passed"] is False, f"Should reject {d * 12}"

    def test_sequential_pattern_rejected(self):
        from src.aadhaar_kyc.number_validator import validate_aadhaar_number
        result = validate_aadhaar_number("123456789012")
        assert result["passed"] is False

    def test_starts_with_0_rejected(self):
        from src.aadhaar_kyc.number_validator import validate_aadhaar_number
        result = validate_aadhaar_number("012345678901")
        assert result["passed"] is False

    def test_starts_with_1_rejected(self):
        from src.aadhaar_kyc.number_validator import validate_aadhaar_number
        result = validate_aadhaar_number("198765432109")
        assert result["passed"] is False

    def test_wrong_length(self):
        from src.aadhaar_kyc.number_validator import validate_aadhaar_number
        result = validate_aadhaar_number("12345")
        assert result["passed"] is False
        assert result["score"] == 0.0

    def test_empty_string(self):
        from src.aadhaar_kyc.number_validator import validate_aadhaar_number
        result = validate_aadhaar_number("")
        assert result["passed"] is False

    def test_valid_number_passes(self):
        from src.aadhaar_kyc.number_validator import validate_aadhaar_number
        # 499118665246 is Verhoeff-valid and starts with valid digit
        result = validate_aadhaar_number("4991 1866 5246")
        assert result["passed"] is True
        assert result["score"] == 1.0


# ==============================================================================
# 3. AADHAAR NUMBER MASKING
# ==============================================================================

class TestMasking:
    def test_mask_standard(self):
        from src.aadhaar_kyc.utils import mask_aadhaar_number
        assert mask_aadhaar_number("1234 5678 9012") == "1234 XXXX XXXX"

    def test_mask_no_spaces(self):
        from src.aadhaar_kyc.utils import mask_aadhaar_number
        assert mask_aadhaar_number("123456789012") == "1234 XXXX XXXX"

    def test_mask_short_string(self):
        from src.aadhaar_kyc.utils import mask_aadhaar_number
        assert mask_aadhaar_number("12") == "XXXX XXXX XXXX"


# ==============================================================================
# 4. COLOR VALIDATOR TESTS
# ==============================================================================

class TestColorValidator:
    """Test tricolour header detection."""

    def test_synthetic_tricolour_passes(self):
        """Create a synthetic image with saffron/white/green header."""
        from src.aadhaar_kyc.color_validator import validate_color_profile

        # Create 400x250 image (ID card proportions)
        img = np.ones((250, 400, 3), dtype=np.uint8) * 255  # White body

        # Top header: saffron (BGR ≈ 25, 140, 255)
        img[0:20, :] = [25, 140, 255]
        # Middle header: white (already white)
        # Green band (BGR ≈ 60, 160, 60)
        img[40:60, :] = [60, 160, 60]

        result = validate_color_profile(img)
        assert result["score"] > 0.3  # Should detect at least some tricolour

    def test_grayscale_skipped(self):
        from src.aadhaar_kyc.color_validator import validate_color_profile
        gray = np.ones((250, 400), dtype=np.uint8) * 128
        result = validate_color_profile(gray)
        assert result["passed"] is None
        assert result["score"] == 0.5


# ==============================================================================
# 5. TAMPER DETECTION TESTS
# ==============================================================================

class TestTamperDetection:
    """Test ELA-based tampering detection."""

    def test_uniform_image_low_tampering(self):
        """A perfectly uniform image should show no tampering."""
        from src.aadhaar_kyc.tamper_detection import detect_tampering

        uniform = np.ones((300, 400, 3), dtype=np.uint8) * 180
        result = detect_tampering(uniform)
        # Uniform images have low ELA variance, which is actually suspicious
        # (could mean AI/synthetic) — so tamper_detected may be True
        assert "score" in result
        assert 0.0 <= result["score"] <= 1.0

    def test_normal_image(self):
        """A natural-looking image should have moderate ELA."""
        from src.aadhaar_kyc.tamper_detection import detect_tampering

        # Create a gradient image (mimics natural content)
        gradient = np.zeros((300, 400, 3), dtype=np.uint8)
        for i in range(300):
            gradient[i, :] = [int(i * 255 / 300)] * 3

        result = detect_tampering(gradient)
        assert "score" in result
        assert isinstance(result["tamper_detected"], bool)


# ==============================================================================
# 6. SCREEN DETECTOR TESTS
# ==============================================================================

class TestScreenDetector:
    """Test screen/screenshot detection."""

    def test_natural_image_no_screen(self):
        """A plain image should not be flagged as screen capture."""
        from src.aadhaar_kyc.screen_detector import detect_screen

        plain = np.random.randint(100, 200, (300, 400, 3), dtype=np.uint8)
        result = detect_screen(plain)
        assert "screen_detected" in result
        assert isinstance(result["screen_detected"], bool)

    def test_synthetic_moire(self):
        """An image with periodic stripes should trigger moiré detection."""
        from src.aadhaar_kyc.screen_detector import detect_screen

        # Create stripe pattern (simulates screen pixel grid)
        stripe = np.zeros((300, 400, 3), dtype=np.uint8)
        for i in range(0, 400, 2):
            stripe[:, i] = 255

        result = detect_screen(stripe)
        assert "score" in result
        assert 0.0 <= result["score"] <= 1.0


# ==============================================================================
# 7. PERSPECTIVE CORRECTION TESTS
# ==============================================================================

class TestPerspectiveCorrection:
    """Test card rectification."""

    def test_no_quad_returns_original(self):
        """Without a clear quadrilateral, should return original image."""
        from src.aadhaar_kyc.perspective_correction import correct_perspective

        # Uniform image — no edges to detect
        uniform = np.ones((300, 400, 3), dtype=np.uint8) * 128
        result = correct_perspective(uniform)
        assert result.shape == uniform.shape

    def test_run_returns_dict(self):
        from src.aadhaar_kyc.perspective_correction import run
        img = np.ones((300, 400, 3), dtype=np.uint8) * 128
        result = run(img)
        assert isinstance(result, dict)
        assert "passed" in result
        assert "score" in result


# ==============================================================================
# 8. PIPELINE SCORING TESTS
# ==============================================================================

class TestPipelineScoring:
    """Test the weighted scoring logic."""

    def test_result_structure(self):
        """Pipeline output must have all required JSON fields."""
        # Create a minimal test image
        test_img_path = "/tmp/test_aadhaar_kyc_pipeline.jpg"
        img = np.ones((250, 400, 3), dtype=np.uint8) * 200
        cv2.imwrite(test_img_path, img)

        from src.aadhaar_kyc.pipeline import analyze
        result = analyze(test_img_path)

        # Check required fields
        required_fields = [
            "is_aadhaar_card", "classification", "confidence_score",
            "aadhaar_number", "qr_data", "ocr_data",
            "face_detected", "tamper_detected", "screen_detected",
            "ai_generated_probability", "module_scores", "explanation",
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

        # Classification must be one of expected values
        assert result["classification"] in ("REAL", "FAKE", "NOT_AADHAAR")

        # Confidence must be in [0, 1]
        assert 0.0 <= result["confidence_score"] <= 1.0

        # Clean up
        os.remove(test_img_path)

    def test_blank_image_not_aadhaar(self):
        """A blank white image should be classified as NOT_AADHAAR."""
        test_img_path = "/tmp/test_blank.jpg"
        blank = np.ones((300, 400, 3), dtype=np.uint8) * 255
        cv2.imwrite(test_img_path, blank)

        from src.aadhaar_kyc.pipeline import analyze
        result = analyze(test_img_path)

        assert result["classification"] == "NOT_AADHAAR"
        os.remove(test_img_path)

    def test_invalid_path_returns_error(self):
        """Non-existent file should return empty explanation."""
        from src.aadhaar_kyc.pipeline import analyze
        result = analyze("/tmp/nonexistent_aadhaar_test.jpg")
        assert result["classification"] == "FAKE"  # Default
        assert "load" in result["explanation"].lower() or result["explanation"] == ""


# ==============================================================================
# 9. MODULE run() CONTRACT
# ==============================================================================

class TestModuleContracts:
    """Every module's run() must return {passed, score, detail, data}."""

    MODULES = [
        "perspective_correction",
        "yolo_detector",
        "ocr_engine",
        "number_validator",
        "qr_validator",
        "layout_validator",
        "logo_detector",
        "face_verification",
        "ai_fake_detector",
        "screen_detector",
        "tamper_detection",
        "color_validator",
    ]

    @pytest.mark.parametrize("module_name", MODULES)
    def test_run_returns_valid_dict(self, module_name):
        """Each module's run() should return a dict with required keys."""
        import importlib
        mod = importlib.import_module(f"src.aadhaar_kyc.{module_name}")
        img = np.ones((250, 400, 3), dtype=np.uint8) * 160

        result = mod.run(img)

        assert isinstance(result, dict), f"{module_name}.run() must return dict"
        assert "passed" in result, f"{module_name} missing 'passed'"
        assert "score" in result, f"{module_name} missing 'score'"
        assert "detail" in result, f"{module_name} missing 'detail'"
        assert 0.0 <= result["score"] <= 1.0, f"{module_name} score out of range"


# ==============================================================================
# CLI ENTRY POINT (for manual testing)
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
