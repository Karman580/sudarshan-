# ==============================================================================
# ai_fake_detector.py — AI-Generated / Fake Aadhaar Image Classifier
# ==============================================================================
# Lightweight CNN classifier to distinguish:
#   - real_aadhaar (class 0)
#   - fake_aadhaar (class 1)
#   - ai_generated (class 2)
#
# Architecture: EfficientNet-B0 / MobileNetV3-Small transfer learning.
# Graceful fallback: returns neutral score (0.5) when weights unavailable.
#
# Weight in pipeline: 5%
# ==============================================================================

import os
import cv2
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger("aadhaar_kyc")

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("torch/torchvision not installed — AI-fake detector disabled.")

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(_MODULE_DIR, "assets", "ai_fake_classifier.pth")

CLASS_NAMES = {0: "real_aadhaar", 1: "fake_aadhaar", 2: "ai_generated"}

# Image transforms matching EfficientNet-B0 training
_TRANSFORM = None
if HAS_TORCH:
    _TRANSFORM = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

# Singleton model cache
_model_cache = {}


# ==============================================================================
# MODEL LOADING
# ==============================================================================

def _build_model(num_classes: int = 3):
    """Build a MobileNetV3-Small model for 3-class classification."""
    model = models.mobilenet_v3_small(weights=None)
    # Replace classifier head
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _load_model(weights_path: str):
    """Load trained weights. Returns None if unavailable."""
    if not HAS_TORCH:
        return None

    if not os.path.isfile(weights_path):
        logger.warning("AI-fake weights not found at %s — using neutral score.", weights_path)
        return None

    try:
        model = _build_model()
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        logger.info("AI-fake classifier loaded from %s", weights_path)
        return model
    except Exception as e:
        logger.error("Failed to load AI-fake model: %s", e)
        return None


# ==============================================================================
# INFERENCE
# ==============================================================================

def classify_image(image: np.ndarray, weights_path: str = DEFAULT_WEIGHTS) -> Dict:
    """
    Classify whether an Aadhaar card image is real, fake, or AI-generated.

    Returns:
        Dict with class_name, probabilities, score, detail.
    """
    if not HAS_TORCH:
        return {
            "class_name": None,
            "probabilities": {},
            "score": 0.5,
            "detail": "PyTorch not available — AI-fake check skipped.",
        }

    # Load model (cached)
    if weights_path not in _model_cache:
        _model_cache[weights_path] = _load_model(weights_path)
    model = _model_cache[weights_path]

    if model is None:
        return {
            "class_name": None,
            "probabilities": {},
            "score": 0.5,
            "detail": "AI-fake model weights unavailable — skipped.",
        }

    try:
        # Preprocess
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = _TRANSFORM(rgb).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        class_idx = int(np.argmax(probs))
        class_name = CLASS_NAMES.get(class_idx, f"unknown_{class_idx}")
        probabilities = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))}

        # Score: probability of being real
        real_prob = probabilities.get("real_aadhaar", 0.0)

        return {
            "class_name": class_name,
            "probabilities": probabilities,
            "score": real_prob,
            "detail": f"Predicted: {class_name} (real={real_prob:.0%}, fake={probabilities.get('fake_aadhaar', 0):.0%}, AI={probabilities.get('ai_generated', 0):.0%})",
        }

    except Exception as e:
        logger.error("AI-fake inference error: %s", e)
        return {
            "class_name": None,
            "probabilities": {},
            "score": 0.5,
            "detail": f"AI-fake inference error: {e}",
        }


# ==============================================================================
# PIPELINE API
# ==============================================================================

def run(image: np.ndarray, **ctx) -> dict:
    """Pipeline-compatible entry point."""
    result = classify_image(image)
    ai_prob = result["probabilities"].get("ai_generated", 0.0)
    return {
        "passed": result["score"] >= 0.4,
        "score": result["score"],
        "detail": result["detail"],
        "data": {
            "class_name": result["class_name"],
            "probabilities": result["probabilities"],
            "ai_generated_probability": ai_prob,
        },
    }
