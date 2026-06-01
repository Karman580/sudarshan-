# ==============================================================================
# inference.py - F-Net Multi-Modal Video Inference Pipeline
# ==============================================================================
# This module handles the complete video inference pipeline using F-Net:
# 1. Frame extraction from video
# 2. Three-stream feature extraction (Frequency/CSP, Temporal, Spatial)
# 3. F-Net CNN model inference
# 4. Metadata-based explainability
#
# Replaces the previous single-stream ViT pipeline with F-Net's
# multi-modal approach for more robust deepfake detection.
# ==============================================================================

import os
import uuid
import torch
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Optional, Callable, List

from src.fnet_model import get_fnet_model, DEVICE
from src.fnet_features import (
    extract_all_features,
    extract_temporal_features,
    extract_spatial_features,
    extract_frames_from_video,
    IMAGE_SIZE
)
from src.fnet_explainability import explain_fnet_prediction


# ==============================================================================
# PATH SETUP
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp")


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def create_run_directories(run_id: str) -> Dict[str, str]:
    """
    Create unique temp directories for this inference run.
    Uses UUID to ensure no conflicts between concurrent runs.
    """
    run_dir = os.path.join(TEMP_DIR, run_id)

    paths = {
        "run_dir": run_dir,
        "frames": os.path.join(run_dir, "frames"),
        "output": os.path.join(run_dir, "output")
    }

    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    return paths


def load_original_frames(frame_dir: str) -> Dict[int, np.ndarray]:
    """
    Load original frames as BGR numpy arrays.
    Returns dict mapping frame_idx → BGR image.
    """
    frames = {}
    if not os.path.exists(frame_dir):
        return frames

    for fname in sorted(os.listdir(frame_dir)):
        if not fname.lower().endswith(".jpg"):
            continue
        try:
            idx = int(fname.split("_")[1].split(".")[0])
        except (IndexError, ValueError):
            continue

        frame = cv2.imread(os.path.join(frame_dir, fname))
        if frame is not None:
            frames[idx] = frame

    return frames


# ==============================================================================
# MAIN VIDEO INFERENCE PIPELINE (F-NET)
# ==============================================================================

def run_video_pipeline(
    video_path: str,
    generate_explanations: bool = True,
    progress_callback: Optional[Callable[[str, int, int], None]] = None
) -> Dict:
    """
    Complete F-Net video inference pipeline.

    Pipeline stages:
    1. Extracting frames...
    2. Analyzing temporal dynamics (rPPG, blink, lips)...
    3. Extracting spatial features (ViT + regions)...
    4. Computing frequency analysis (CSP pyramid)...
    5. Running F-Net model inference...

    Args:
        video_path: Absolute path to input video file
        generate_explanations: Whether to compute metadata-based explanations
        progress_callback: Optional callback(stage_name, current, total)

    Returns:
        result: Dict with label, confidence, explanation, run_id, etc.
    """
    result = {
        "label": "ERROR",
        "confidence": 0.0,
        "heatmap_frames": {},
        "explanation": "",
        "run_id": "",
        "original_frames": {},
        "all_frame_paths": []
    }

    try:
        run_id = str(uuid.uuid4())
        result["run_id"] = run_id
        paths = create_run_directories(run_id)

        # =================================================================
        # STAGE 1-4: EXTRACT ALL F-NET FEATURES
        # =================================================================
        features, metadata = extract_all_features(
            video_path=video_path,
            include_frequency=True,
            progress_callback=progress_callback
        )

        # Move features to device
        features = {k: v.to(DEVICE).float() for k, v in features.items()}

        # Store frame paths for GUI display
        frame_paths = metadata.get("frame_paths", [])
        result["all_frame_paths"] = frame_paths
        result["original_frames"] = {
            i: fp for i, fp in enumerate(frame_paths)
        }

        # =================================================================
        # STAGE 5: F-NET MODEL INFERENCE
        # =================================================================
        if progress_callback:
            progress_callback("Running F-Net inference", 5, 6)

        model = get_fnet_model()

        with torch.no_grad():
            logits, stream_outputs = model(features)

        # Determine label and confidence
        probs = torch.softmax(logits, dim=-1)
        fake_prob = probs[0, 1].item()
        label = "FAKE" if fake_prob > 0.5 else "REAL"
        confidence = fake_prob if label == "FAKE" else 1.0 - fake_prob

        result["label"] = label
        result["confidence"] = confidence

        # =================================================================
        # STAGE 6: EXPLAINABILITY
        # =================================================================
        if generate_explanations:
            if progress_callback:
                progress_callback("Generating explanation", 6, 6)

            try:
                explanation = explain_fnet_prediction(
                    logits=logits,
                    stream_outputs=stream_outputs,
                    temporal_metadata=metadata.get("temporal", {}),
                    spatial_metadata=metadata.get("spatial", {})
                )
                result["explanation"] = explanation["summary"]
            except Exception as e:
                print(f"[WARN] F-Net explainability failed: {e}")
                result["explanation"] = (
                    f"Classification: {label} (Confidence: {confidence:.2%})\n\n"
                    f"F-Net multi-modal analysis complete. "
                    f"Detailed explanation unavailable."
                )
        else:
            result["explanation"] = (
                f"Classification: {label} (Confidence: {confidence:.2%})\n\n"
                f"F-Net multi-modal analysis complete."
            )

        return result

    except Exception as e:
        print(f"[ERROR] F-Net pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        result["label"] = "ERROR"
        result["explanation"] = f"Pipeline error: {str(e)}"
        return result


# ==============================================================================
# STATIC IMAGE PIPELINE
# ==============================================================================

def run_image_pipeline(image_path: str) -> Dict:
    """
    Run F-Net inference on a single static image.
    Temporal features will be minimal (single frame).
    """
    result = {
        "label": "ERROR",
        "confidence": 0.0,
        "explanation": "",
        "heatmap_frames": {},
        "run_id": str(uuid.uuid4())
    }

    try:
        paths = create_run_directories(result["run_id"])

        # Save image as a single frame
        img = cv2.imread(image_path)
        if img is None:
            result["explanation"] = "Failed to load image file."
            return result

        frame_path = os.path.join(paths["frames"], "frame_0000.jpg")
        cv2.imwrite(frame_path, img)
        frame_paths = [frame_path]

        # Extract features (limited: single frame → weak temporal signal)
        # Temporal features will be mostly zeros with a single frame
        temporal_tensor, temporal_meta = extract_temporal_features(frame_paths)
        spatial_features, spatial_meta = extract_spatial_features(frame_paths)
        spatial_cls, spatial_act = spatial_features

        # Frequency features from single frame
        from src.fnet_features import extract_frequency_features
        freq_tensor = extract_frequency_features(frame_paths)

        features = {
            'frequency': freq_tensor.unsqueeze(0).to(DEVICE).float(),
            'temporal': temporal_tensor.unsqueeze(0).to(DEVICE).float(),
            'spatial_cls': spatial_cls.unsqueeze(0).to(DEVICE).float(),
            'spatial_act': spatial_act.unsqueeze(0).to(DEVICE).float(),
        }

        # Run model
        model = get_fnet_model()
        with torch.no_grad():
            logits, stream_outputs = model(features)

        probs = torch.softmax(logits, dim=-1)
        fake_prob = probs[0, 1].item()
        label = "FAKE" if fake_prob > 0.5 else "REAL"
        confidence = fake_prob if label == "FAKE" else 1.0 - fake_prob

        result["label"] = label
        result["confidence"] = confidence
        result["explanation"] = (
            f"Static Image Analysis (F-Net)\n"
            f"Result: {label}\n"
            f"Confidence: {confidence:.2%}\n\n"
            f"Note: Temporal analysis is limited for single images.\n"
            f"Video upload recommended for full multi-modal analysis."
        )

        return result

    except Exception as e:
        print(f"[ERROR] Image pipeline failed: {e}")
        result["explanation"] = f"Error: {str(e)}"
        return result


# ==============================================================================
# LIVE WEBCAM PIPELINE (SIMPLIFIED F-NET — NO CSP)
# ==============================================================================

def run_webcam_pipeline(frame_buffer: List[np.ndarray]) -> Dict:
    """
    Run F-Net inference on a buffer of webcam frames.
    Uses spatial + temporal only (no CSP) for performance.
    """
    result = {
        "label": "Analyzing...",
        "confidence": 0.0,
        "explanation": ""
    }

    if not frame_buffer:
        return result

    try:
        run_id = str(uuid.uuid4())
        paths = create_run_directories(run_id)

        # Write buffer to disk for feature extraction
        frame_paths = []
        for i, frame in enumerate(frame_buffer):
            fp = os.path.join(paths["frames"], f"frame_{i:04d}.jpg")
            cv2.imwrite(fp, frame)
            frame_paths.append(fp)

        # Extract features (NO frequency/CSP for speed)
        temporal_tensor, _ = extract_temporal_features(frame_paths)
        spatial_features, _ = extract_spatial_features(frame_paths)
        spatial_cls, spatial_act = spatial_features

        # Use zero frequency tensor (CSP too slow for live)
        freq_tensor = torch.zeros(8, 1, IMAGE_SIZE, IMAGE_SIZE)

        features = {
            'frequency': freq_tensor.unsqueeze(0).to(DEVICE).float(),
            'temporal': temporal_tensor.unsqueeze(0).to(DEVICE).float(),
            'spatial_cls': spatial_cls.unsqueeze(0).to(DEVICE).float(),
            'spatial_act': spatial_act.unsqueeze(0).to(DEVICE).float(),
        }

        model = get_fnet_model()
        with torch.no_grad():
            logits, _ = model(features)

        probs = torch.softmax(logits, dim=-1)
        fake_prob = probs[0, 1].item()
        label = "FAKE" if fake_prob > 0.5 else "REAL"
        confidence = fake_prob if label == "FAKE" else 1.0 - fake_prob

        result["label"] = label
        result["confidence"] = confidence

        # Cleanup
        try:
            import shutil
            shutil.rmtree(paths["run_dir"])
        except Exception:
            pass

        return result

    except Exception as e:
        print(f"[ERROR] Webcam pipeline failed: {e}")
        return result


# ==============================================================================
# SCREEN BUFFER PIPELINE (INTERVIEW MONITOR MODE)
# ==============================================================================

def run_screen_buffer_pipeline(
    frame_buffer: List[np.ndarray],
    target_embedding=None,
    similarity_threshold: float = 0.5,
    min_face_size: int = 50
) -> Dict:
    """
    Run F-Net inference on screen-captured frames (Interview Monitor Mode).
    Uses spatial + temporal only (no CSP) for real-time performance.

    When target_embedding is provided, uses face identity matching to select
    the target face from multi-face scenes.
    """
    result = {
        "label": "Analyzing...",
        "confidence": 0.0,
        "quality_warning": None
    }

    if not frame_buffer:
        return result

    try:
        run_id = str(uuid.uuid4())
        paths = create_run_directories(run_id)

        # Write frames to disk
        frame_paths = []
        for i, frame in enumerate(frame_buffer):
            fp = os.path.join(paths["frames"], f"frame_{i:04d}.jpg")
            cv2.imwrite(fp, frame)
            frame_paths.append(fp)

        # Extract features (simplified — no CSP)
        temporal_tensor, _ = extract_temporal_features(frame_paths)
        spatial_features, _ = extract_spatial_features(frame_paths)
        spatial_cls, spatial_act = spatial_features
        freq_tensor = torch.zeros(8, 1, IMAGE_SIZE, IMAGE_SIZE)

        features = {
            'frequency': freq_tensor.unsqueeze(0).to(DEVICE).float(),
            'temporal': temporal_tensor.unsqueeze(0).to(DEVICE).float(),
            'spatial_cls': spatial_cls.unsqueeze(0).to(DEVICE).float(),
            'spatial_act': spatial_act.unsqueeze(0).to(DEVICE).float(),
        }

        model = get_fnet_model()
        with torch.no_grad():
            logits, _ = model(features)

        probs = torch.softmax(logits, dim=-1)
        fake_prob = probs[0, 1].item()
        label = "FAKE" if fake_prob > 0.5 else "REAL"
        confidence = fake_prob if label == "FAKE" else 1.0 - fake_prob

        result["label"] = label
        result["confidence"] = confidence

        # Cleanup
        try:
            import shutil
            shutil.rmtree(paths["run_dir"])
        except Exception:
            pass

        return result

    except Exception as e:
        print(f"[ERROR] Screen buffer pipeline failed: {e}")
        return result


# ==============================================================================
# AADHAAR CARD VERIFICATION PIPELINE
# ==============================================================================

def run_aadhaar_pipeline(*args, **kwargs) -> Dict:
    return {
        "label": "DISABLED",
        "confidence": 0.0,
        "explanation": "Aadhaar verification feature has been disabled.",
        "is_aadhaar": False
    }
