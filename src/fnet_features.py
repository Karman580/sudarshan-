# ==============================================================================
# fnet_features.py - F-Net Feature Extraction for Single-Video Inference
# ==============================================================================
# Extracts three feature streams from a single video:
#   1. Frequency (CSP):  (8, 1, 256, 256) phase-only steerable pyramid
#   2. Temporal:         (2, 256, 256) rPPG + blink + lip dynamics
#   3. Spatial:          (2, 256, 256) ViT features + region masks
#
# Adapted from F_NET/Sptial.py, F_NET/temporal.py, F_NET/pyramids/
# for single-video inference (not batch dataset preprocessing).
# ==============================================================================

import os
import sys
import cv2
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# ==============================================================================
# CONSTANTS
# ==============================================================================

IMAGE_SIZE = 256
CHANNELS = 2
PATCH_SIZE = 16
NUM_FRAMES = 32
NUM_CSP_SCALES = 8

# MediaPipe face landmark indices
ROI_LANDMARKS = {"forehead": 151, "left_cheek": 50, "right_cheek": 280}
ROI_RADIUS = 31
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155,
            133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382,
             362, 398, 384, 385, 386, 387, 388, 466]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321,
              375, 291, 308, 324, 318, 402, 317, 14, 87]
LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# LAZY-LOADED SINGLETONS
# ==============================================================================

_face_mesh = None
_vit_model = None
_vit_preprocess = None


def _get_face_mesh():
    """Lazy-load MediaPipe FaceMesh."""
    global _face_mesh
    if _face_mesh is None:
        try:
            import mediapipe as mp
            mp_face_mesh = mp.solutions.face_mesh
            _face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True
            )
            print("[INFO] MediaPipe FaceMesh loaded")
        except Exception as e:
            print(f"[WARN] MediaPipe load failed: {e}. Temporal/spatial features "
                  "will use fallback (zero tensors).")
            _face_mesh = None
    return _face_mesh


def _get_vit_model():
    """Lazy-load ViT model for spatial feature extraction."""
    global _vit_model, _vit_preprocess
    if _vit_model is None:
        try:
            import timm
        except ImportError:
            print("[WARN] timm not installed. Attempting auto-install...")
            import subprocess
            import sys
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "timm"])
                import timm
                print("[INFO] Successfully auto-installed timm.")
            except Exception as e:
                print(f"[ERROR] Failed to auto-install timm: {e}")
                print("Please install manually using: python3 -m pip install timm")
                return None, None

        from torchvision import transforms

        _vit_model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            features_only=False
        )
        _vit_model = _vit_model.to(DEVICE)
        _vit_model.eval()

        _vit_preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        print("[INFO] ViT spatial feature extractor loaded")
    return _vit_model, _vit_preprocess


# ==============================================================================
# FRAME EXTRACTION
# ==============================================================================

def extract_frames_from_video(video_path: str, num_frames: int = NUM_FRAMES) -> List[str]:
    """
    Extract frames from video and return list of file paths.
    Saves frames to a temp directory next to the video.

    Returns:
        List of absolute frame file paths
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count == 0:
        cap.release()
        return []

    frame_ids = [int(i * frame_count / num_frames) for i in range(num_frames)]

    # Create temp dir for frames
    import uuid
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
    temp_dir = os.path.join(PROJECT_ROOT, "temp", f"fnet_{uuid.uuid4().hex[:8]}", "frames")
    os.makedirs(temp_dir, exist_ok=True)

    frame_paths = []
    for i, frame_id in enumerate(frame_ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if ret:
            path = os.path.join(temp_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(path, frame)
            frame_paths.append(path)

    cap.release()
    return frame_paths


# ==============================================================================
# MEDIAPIPE FACE MESH UTILITIES
# ==============================================================================

def _landmark_to_coords(landmarks, indices, shape):
    """Convert MediaPipe landmark indices to pixel coordinates."""
    h, w = shape[:2]
    points = np.array([
        (int(landmarks.landmark[idx].x * w),
         int(landmarks.landmark[idx].y * h))
        for idx in indices
    ], dtype=np.int32)
    return points


def _create_region_masks(image, face_mesh_instance):
    """
    Create facial region masks using MediaPipe Face Mesh.

    Returns:
        mask: (H, W) uint8 array with labels 1-5
        centers: dict of region name → (x, y)
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    results = face_mesh_instance.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        return None, None

    landmarks = results.multi_face_landmarks[0]

    region_labels = {"forehead": 1, "left_cheek": 2, "right_cheek": 3}
    centers = {}
    for region, idx in ROI_LANDMARKS.items():
        lm = landmarks.landmark[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        centers[region] = (x, y)
        cv2.circle(mask, (x, y), ROI_RADIUS, region_labels[region], -1)

    # Eyes
    left_eye = _landmark_to_coords(landmarks, LEFT_EYE, image.shape)
    right_eye = _landmark_to_coords(landmarks, RIGHT_EYE, image.shape)
    cv2.fillPoly(mask, [left_eye], 4)
    cv2.fillPoly(mask, [right_eye], 4)

    # Lips
    lips_outer = _landmark_to_coords(landmarks, LIPS_OUTER, image.shape)
    lips_inner = _landmark_to_coords(landmarks, LIPS_INNER, image.shape)
    lips_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(lips_mask, [lips_outer], 5)
    cv2.fillPoly(lips_mask, [lips_inner], 0)
    mask[lips_mask == 5] = 5

    return mask, centers


def _get_temporal_union_mask(frame_paths: List[str], face_mesh_instance) -> Tuple:
    """
    Create a Temporal Union Mask — aggregates masks across all frames.
    A pixel gets a region id if it was EVER that region in any frame.
    """
    all_masks = []
    first_centers = None

    for fp in frame_paths:
        img = cv2.imread(fp)
        if img is None:
            continue
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        current_mask, centers = _create_region_masks(img, face_mesh_instance)

        if current_mask is not None:
            all_masks.append(current_mask)
            if first_centers is None:
                first_centers = centers

    if not all_masks:
        return None, None

    union_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for m in all_masks:
        for region_id in range(1, 6):
            union_mask[m == region_id] = region_id

    return union_mask, first_centers


# ==============================================================================
# TEMPORAL FEATURE EXTRACTION
# ==============================================================================

def _extract_rppg(frame_paths, mask, region_label):
    """Extract rPPG signal from a facial region."""
    signal = []
    for f in frame_paths:
        img = cv2.imread(f)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        green = img[:, :, 1][mask == region_label]
        signal.append(float(green.mean()) if green.size else 0.)

    signal = np.array(signal)
    detrended = signal - signal.mean()
    fft_vals = np.abs(np.fft.rfft(detrended))
    peak_freq = int(np.argmax(fft_vals))
    mean_val = float(signal.mean())
    return mean_val, peak_freq


def _extract_blink_metrics(frame_paths, mask, region_label):
    """Extract blink acceleration and rate."""
    lid_positions = []
    for f in frame_paths:
        img = cv2.imread(f)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        lid_pos = (img[:, :, 1][mask == region_label].mean()
                   if (mask == region_label).sum() else 0.)
        lid_positions.append(float(lid_pos))

    lid_positions = np.array(lid_positions)
    acc = (float(np.abs(np.diff(np.diff(lid_positions))).mean())
           if len(lid_positions) > 2 else 0.)
    rate = (float((np.diff(lid_positions) < -5).sum() / len(lid_positions))
            if len(lid_positions) else 0.)
    return acc, rate


def _extract_lip_movement(frame_paths, mask, region_label):
    """Extract lip movement magnitude."""
    lips = []
    for f in frame_paths:
        img = cv2.imread(f)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        lips.append(
            img[:, :, 1][mask == region_label].mean()
            if (mask == region_label).sum() else 0.
        )
    lips = np.array(lips)
    movement = float(np.abs(np.diff(lips)).mean()) if len(lips) > 1 else 0.
    return movement


def _extract_interframe_corr(frame_paths):
    """Extract inter-frame Spearman correlation."""
    try:
        from scipy.stats import spearmanr
    except ImportError:
        print("[WARN] scipy not installed — inter-frame correlation will be 0.0")
        return 0.0

    corrs = []
    for i in range(len(frame_paths) - 1):
        img1 = cv2.imread(frame_paths[i])
        img2 = cv2.imread(frame_paths[i + 1])
        if img1 is None or img2 is None:
            continue
        img1g = cv2.cvtColor(cv2.resize(img1, (IMAGE_SIZE, IMAGE_SIZE)),
                             cv2.COLOR_BGR2GRAY).flatten()
        img2g = cv2.cvtColor(cv2.resize(img2, (IMAGE_SIZE, IMAGE_SIZE)),
                             cv2.COLOR_BGR2GRAY).flatten()
        corr, _ = spearmanr(img1g, img2g)
        corrs.append(float(corr) if not np.isnan(corr) else 0.)

    return float(np.mean(corrs)) if corrs else 0.


def extract_temporal_features(frame_paths: List[str]) -> Tuple[torch.Tensor, Dict]:
    """
    Extract temporal features from video frames.

    Returns:
        tensor: (2, 256, 256) temporal feature tensor
            Ch0: Regional dynamics (rPPG, blink, lip)
            Ch1: Global temporal stability (inter-frame correlation)
        metadata: dict with extracted feature values
    """
    face_mesh = _get_face_mesh()

    if face_mesh is None:
        print("[WARN] MediaPipe unavailable — using zero temporal tensor")
        return torch.zeros(2, IMAGE_SIZE, IMAGE_SIZE), {}

    mask, centers = _get_temporal_union_mask(frame_paths, face_mesh)

    if mask is None:
        print("[WARN] No face detected — using zero temporal tensor")
        return torch.zeros(2, IMAGE_SIZE, IMAGE_SIZE), {}

    # Extract all temporal features
    mean_rppg_forehead, peak_rppg_forehead = _extract_rppg(frame_paths, mask, 1)
    mean_rppg_lcheek, peak_rppg_lcheek = _extract_rppg(frame_paths, mask, 2)
    mean_rppg_rcheek, peak_rppg_rcheek = _extract_rppg(frame_paths, mask, 3)
    blink_acc, blink_rate = _extract_blink_metrics(frame_paths, mask, 4)
    lip_movement = _extract_lip_movement(frame_paths, mask, 5)
    inter_corr = _extract_interframe_corr(frame_paths)

    # Build tensor map
    channel0 = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    channel0[mask == 1] = mean_rppg_forehead
    channel0[mask == 2] = mean_rppg_lcheek
    channel0[mask == 3] = mean_rppg_rcheek
    channel0[mask == 4] = blink_acc + blink_rate
    channel0[mask == 5] = lip_movement

    channel1 = np.ones((IMAGE_SIZE, IMAGE_SIZE)) * inter_corr

    tensor = torch.stack([
        torch.tensor(channel0, dtype=torch.float32),
        torch.tensor(channel1, dtype=torch.float32)
    ])  # (2, 256, 256)

    metadata = {
        "features": {
            "forehead": {"mean_rppg": mean_rppg_forehead, "peak_rppg": peak_rppg_forehead},
            "left_cheek": {"mean_rppg": mean_rppg_lcheek, "peak_rppg": peak_rppg_lcheek},
            "right_cheek": {"mean_rppg": mean_rppg_rcheek, "peak_rppg": peak_rppg_rcheek},
            "eyes": {"blink_acceleration": blink_acc, "blink_rate": blink_rate},
            "lips": {"lip_movement": lip_movement},
            "interframe_corr": inter_corr
        },
        "region_centers": centers,
        "mask": mask
    }

    return tensor, metadata


# ==============================================================================
# SPATIAL FEATURE EXTRACTION
# ==============================================================================

def extract_spatial_features(frame_paths: List[str]) -> Tuple[torch.Tensor, Dict]:
    """
    Extract spatial features using pretrained ViT + MediaPipe region masks.

    Returns:
        tensor: (2, 256, 256) spatial feature tensor
            Ch0: ViT spatial embeddings
            Ch1: Auxiliary features / attention
        metadata: dict with mask and region info
    """
    vit_model, preprocess = _get_vit_model()
    face_mesh = _get_face_mesh()

    if vit_model is None or preprocess is None:
        print("[WARN] ViT model unavailable — using zero spatial tensor")
        return (torch.zeros(768), torch.zeros(1, 14, 14)), {}

    # Prepare frame tensors for ViT
    spatial_tensors = []
    for fp in frame_paths:
        img = cv2.imread(fp)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = preprocess(img)
        spatial_tensors.append(img_tensor)

    if not spatial_tensors:
        print("[WARN] No valid frames for spatial extraction")
        return (torch.zeros(768), torch.zeros(1, 14, 14)), {}

    batch_tensor = torch.stack(spatial_tensors).to(DEVICE)

    # Extract ViT features
    with torch.no_grad():
        features = vit_model.forward_features(batch_tensor)  # (B, 197, 768)
        
        cls_token = features[:, 0, :]  # (B, 768)
        patch_tokens = features[:, 1:, :]  # (B, 196, 768)
        
        # Calculate activation map by averaging across channel dim
        act_map = patch_tokens.mean(dim=-1)  # (B, 196)
        act_map = act_map.view(batch_tensor.size(0), 14, 14)  # (B, 14, 14)

    cls_token = cls_token.cpu()
    act_map = act_map.cpu()

    # Average across all frames -> (768,) and (1, 14, 14)
    video_cls_token = cls_token.mean(dim=0)
    video_act_map = act_map.mean(dim=0).unsqueeze(0)
    
    video_tensor = (video_cls_token, video_act_map)

    # Extract union mask for metadata
    metadata = {}
    if face_mesh is not None:
        all_masks = []
        first_centers = None
        for fp in frame_paths:
            img = cv2.imread(fp)
            if img is None:
                continue
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            current_mask, centers = _create_region_masks(img, face_mesh)
            if current_mask is not None:
                all_masks.append(current_mask)
                if first_centers is None:
                    first_centers = centers

        if all_masks:
            union_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            for m in all_masks:
                for region_id in range(1, 6):
                    union_mask[m == region_id] = region_id

            metadata = {
                "mask": union_mask,
                "region_centers": first_centers,
            }

    return video_tensor, metadata


# ==============================================================================
# FREQUENCY (CSP) FEATURE EXTRACTION
# ==============================================================================

def extract_frequency_features(frame_paths: List[str]) -> torch.Tensor:
    """
    Extract frequency features using Complex Steerable Pyramid (CSP).

    Decomposes each frame into 8 frequency scales, extracts phase information,
    and averages across frames to produce a (8, 1, 256, 256) tensor.

    Returns:
        tensor: (8, 1, 256, 256) phase-only frequency features
    """
    try:
        # Add pyramids directory to path
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
        pyramids_dir = os.path.join(PROJECT_ROOT, "F_NET", "pyramids")

        if pyramids_dir not in sys.path:
            sys.path.insert(0, pyramids_dir)

        from steerable_pyramid import SteerablePyramid

        # Create pyramid decomposer
        pyr = SteerablePyramid(depth=8, orientations=1, complex_pyr=True)

        all_scale_phases = [[] for _ in range(NUM_CSP_SCALES)]

        for fp in frame_paths:
            img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float64)

            # Build pyramid
            filters, crops = pyr.get_filters(img, cropped=True)
            pyramid = pyr.build_pyramid(img, filters, crops, freq=False)

            # Extract phase from each scale
            # pyramid is a list: [lo, (scale1_band0), (scale2_band0), ..., hi]
            # We need 8 scales
            for scale_idx in range(min(NUM_CSP_SCALES, len(pyramid) - 2)):
                band = pyramid[scale_idx + 1]  # skip lowpass
                if isinstance(band, (list, tuple)):
                    band = band[0]  # single orientation

                # Extract phase
                phase = np.angle(band) if np.iscomplexobj(band) else band

                # Resize to standard size
                phase_resized = cv2.resize(
                    phase.real if np.iscomplexobj(phase) else phase,
                    (IMAGE_SIZE, IMAGE_SIZE)
                )
                all_scale_phases[scale_idx].append(phase_resized)

        # Average across frames → (8, 1, 256, 256)
        csp_tensor = torch.zeros(NUM_CSP_SCALES, 1, IMAGE_SIZE, IMAGE_SIZE)

        for scale_idx in range(NUM_CSP_SCALES):
            if all_scale_phases[scale_idx]:
                avg_phase = np.mean(all_scale_phases[scale_idx], axis=0)
                csp_tensor[scale_idx, 0] = torch.tensor(avg_phase, dtype=torch.float32)

        return csp_tensor

    except Exception as e:
        print(f"[WARN] CSP extraction failed: {e}. Using zero frequency tensor.")
        return torch.zeros(NUM_CSP_SCALES, 1, IMAGE_SIZE, IMAGE_SIZE)


# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================

def extract_all_features(
    video_path: str,
    include_frequency: bool = True,
    progress_callback=None
) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """
    Extract all F-Net features from a single video.

    Args:
        video_path: Path to input video file
        include_frequency: Whether to extract CSP frequency features
                          (disable for real-time/webcam modes)
        progress_callback: Optional callback(stage_name, current, total)

    Returns:
        features: Dict with 'frequency', 'temporal', 'spatial' tensors
        metadata: Dict with temporal and spatial metadata
    """
    if progress_callback:
        progress_callback("Extracting frames", 1, 5)

    # 1. Extract frames
    frame_paths = extract_frames_from_video(video_path, NUM_FRAMES)

    if not frame_paths:
        raise ValueError(f"No frames extracted from {video_path}")

    # 2. Extract temporal features
    if progress_callback:
        progress_callback("Analyzing temporal dynamics", 2, 5)
    temporal_tensor, temporal_meta = extract_temporal_features(frame_paths)

    # 3. Extract spatial features
    if progress_callback:
        progress_callback("Extracting spatial features", 3, 5)
    spatial_tensor, spatial_meta = extract_spatial_features(frame_paths)

    # 4. Extract frequency features (optional — slow)
    if progress_callback:
        progress_callback("Computing frequency analysis", 4, 5)

    if include_frequency:
        freq_tensor = extract_frequency_features(frame_paths)
    else:
        freq_tensor = torch.zeros(NUM_CSP_SCALES, 1, IMAGE_SIZE, IMAGE_SIZE)

    # 5. Bundle
    if progress_callback:
        progress_callback("Preparing model input", 5, 5)

    spatial_cls, spatial_act = spatial_tensor

    features = {
        'frequency': freq_tensor.unsqueeze(0),    # (1, 8, 1, 256, 256)
        'temporal': temporal_tensor.unsqueeze(0),  # (1, 2, 256, 256)
        'spatial_cls': spatial_cls.unsqueeze(0),    # (1, 768)
        'spatial_act': spatial_act.unsqueeze(0),    # (1, 1, 14, 14)
    }

    metadata = {
        'temporal': temporal_meta,
        'spatial': spatial_meta,
        'frame_paths': frame_paths,
    }

    # Cleanup temp frames
    try:
        temp_dir = os.path.dirname(frame_paths[0])
        import shutil
        shutil.rmtree(os.path.dirname(temp_dir), ignore_errors=True)
    except Exception:
        pass

    return features, metadata
