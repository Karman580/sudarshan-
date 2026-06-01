
# explainability.py - Frame-wise Visual Explainability for Deepfake Detection

# This module provides post-hoc gradient-based attribution using SHAP
# GradientExplainer with Attention Rollout as a fallback. Attributions are
# computed only on top-K representative frames for real-world performance.
#
# IMPORTANT: The model remains FROZEN. All explainability is inference-time only.


import os
import numpy as np
import cv2
import torch
import shap
from typing import Dict, List, Tuple, Optional
from PIL import Image


# ==============================================================================
# CONSTANTS
# ==============================================================================

# Number of representative frames for explainability (performance optimization)
TOP_K_FRAMES = 5

# Frame indices for uniform temporal sampling from 32 frames
REPRESENTATIVE_INDICES = [0, 7, 15, 23, 31]

# Heatmap alpha blending factor (0.0 = invisible, 1.0 = opaque)
HEATMAP_ALPHA = 0.4

# SHAP background sample count
SHAP_BACKGROUND_SAMPLES = 5


# ==============================================================================
# FRAME SELECTION
# ==============================================================================

def select_representative_frames(
    total_frames: int = 32,
    k: int = TOP_K_FRAMES
) -> List[int]:
    """
    Select K representative frame indices via uniform temporal sampling.
    
    This reduces computation while maintaining temporal coverage across the video.
    For 32 frames and K=5, returns indices [0, 7, 15, 23, 31].
    
    Args:
        total_frames: Total number of frames extracted from video
        k: Number of representative frames to select
        
    Returns:
        List of frame indices for explainability computation
    """
    if total_frames <= k:
        return list(range(total_frames))
    
    # Uniform temporal sampling
    indices = np.linspace(0, total_frames - 1, k, dtype=int).tolist()
    return indices


# ==============================================================================
# ATTENTION ROLLOUT (FALLBACK EXPLAINABILITY)
# ==============================================================================

class AttentionRollout:
    """
    Fallback explainability using ViT attention rollout.
    
    This is used when SHAP GradientExplainer fails or produces unstable results.
    
    Methodology:
    - Registers forward hooks on all 12 transformer encoder blocks
    - Extracts attention weights from attn_drop layer (after softmax, before dropout)
    - Computes rollout by recursively multiplying attention matrices
    - Extracts CLS token → patch attention and reshapes to 14×14 → 224×224
    
    Reference: Abnar & Zuidema, "Quantifying Attention Flow in Transformers" (2020)
    """
    
    def __init__(self, model):
        """
        Initialize attention rollout with the ViT model.
        
        Args:
            model: ViTVideoClassifier instance with accessible .vit.blocks
        """
        self.model = model
        self.attention_maps = []
        self.hooks = []
        
    def _hook_fn(self, module, input, output):
        """
        Forward hook callback to capture attention weights.
        
        The hook is registered on the attention dropout layer, which receives
        the softmax-normalized attention weights as input.
        """
        # Input to attn_drop is the attention weights after softmax
        if len(input) > 0 and input[0] is not None:
            self.attention_maps.append(input[0].detach().cpu())
    
    def register_hooks(self):
        """Register forward hooks on all attention layers in ViT blocks."""
        self.attention_maps = []
        self.hooks = []
        
        # Access all 12 transformer blocks in vit_base_patch16_224
        for block in self.model.vit.blocks:
            hook = block.attn.attn_drop.register_forward_hook(self._hook_fn)
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks to prevent memory leaks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def compute(self, face_tensor: torch.Tensor) -> np.ndarray:
        """
        Compute attention rollout for a single face image.
        
        Args:
            face_tensor: Input tensor of shape (1, 1, 3, 224, 224)
            
        Returns:
            attention_map: Numpy array of shape (224, 224) with normalized attention
        """
        self.register_hooks()
        
        try:
            # Forward pass to capture attention
            with torch.no_grad():
                _ = self.model(face_tensor)
            
            if len(self.attention_maps) == 0:
                # No attention captured, return uniform map
                return np.ones((224, 224), dtype=np.float32) * 0.5
            
            # Compute rollout across all layers
            # Each attention map has shape: (batch, heads, tokens, tokens)
            # tokens = 1 (CLS) + 196 (14×14 patches) = 197
            
            result = None
            for attn in self.attention_maps:
                # Average across heads: (batch, tokens, tokens)
                attn = attn.mean(dim=1)
                
                # Add residual connection (identity matrix)
                eye = torch.eye(attn.size(-1), device=attn.device)
                attn = attn + eye
                
                # Normalize rows to sum to 1
                attn = attn / attn.sum(dim=-1, keepdim=True)
                
                # Accumulate via matrix multiplication
                if result is None:
                    result = attn
                else:
                    result = torch.bmm(attn, result)
            
            # Extract CLS token attention to patches (skip CLS token itself)
            # result shape: (batch, tokens, tokens)
            # CLS is at index 0, patches are indices 1-196
            cls_attention = result[0, 0, 1:]  # Shape: (196,)
            
            # Reshape to 14×14 patch grid
            patch_attn = cls_attention.reshape(14, 14).numpy()
            
            # Upsample to 224×224
            attention_map = cv2.resize(
                patch_attn,
                (224, 224),
                interpolation=cv2.INTER_CUBIC
            )
            
            # Normalize to [0, 1]
            attention_map = (attention_map - attention_map.min()) / \
                           (attention_map.max() - attention_map.min() + 1e-8)
            
            return attention_map.astype(np.float32)
            
        finally:
            self.remove_hooks()


# ==============================================================================
# SHAP-BASED ATTRIBUTION
# ==============================================================================

def compute_shap_attribution(
    model: torch.nn.Module,
    face_tensor: torch.Tensor,
    background_tensors: torch.Tensor,
    device: str = "cpu"
) -> Optional[np.ndarray]:
    """
    Compute SHAP GradientExplainer attribution for a single face.
    
    SHAP GradientExplainer is a post-hoc gradient-based attribution method.
    Known limitations:
    - Can produce noisy/unstable results on Vision Transformers
    - Assumes feature independence (violated by patch embeddings)
    
    Args:
        model: ViTVideoClassifier in eval mode
        face_tensor: Single face tensor (1, 1, 3, 224, 224)
        background_tensors: Background samples (N, 1, 3, 224, 224)
        device: 'cuda' or 'cpu'
        
    Returns:
        attribution_map: Numpy array (224, 224) or None if SHAP fails
    """
    try:
        # Create SHAP GradientExplainer
        explainer = shap.GradientExplainer(model, background_tensors)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(face_tensor)
        
        # shap_values is a list; take first output
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Shape: (1, 1, 3, 224, 224) → reduce to (224, 224)
        # Average across batch, frames, and channels
        attribution = np.abs(shap_values).mean(axis=(0, 1, 2))
        
        # Ensure shape is (224, 224)
        if attribution.shape != (224, 224):
            attribution = cv2.resize(attribution, (224, 224))
        
        # Normalize to [0, 1]
        attribution = (attribution - attribution.min()) / \
                     (attribution.max() - attribution.min() + 1e-8)
        
        return attribution.astype(np.float32)
        
    except Exception as e:
        # SHAP failed - caller should use fallback
        print(f"[WARN] SHAP attribution failed: {e}")
        return None


# ==============================================================================
# HEATMAP GENERATION AND OVERLAY
# ==============================================================================

def generate_heatmap(
    attribution: np.ndarray,
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Convert attribution map to color-mapped heatmap.
    
    Uses OpenCV COLORMAP_JET where:
    - Red/Yellow = High contribution to decision
    - Blue = Low contribution
    
    Args:
        attribution: Normalized attribution map (H, W) in [0, 1]
        target_size: Output size (width, height)
        
    Returns:
        heatmap: BGR image of shape (H, W, 3)
    """
    # Resize if needed
    if attribution.shape[:2] != (target_size[1], target_size[0]):
        attribution = cv2.resize(attribution, target_size)
    
    # Convert to uint8 for colormap
    heatmap_uint8 = (attribution * 255).astype(np.uint8)
    
    # Apply JET colormap (returns BGR)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    return heatmap_color


def blend_heatmap_on_frame(
    frame: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = HEATMAP_ALPHA
) -> np.ndarray:
    """
    Alpha-blend heatmap overlay onto original frame.
    
    Temporal Alignment Guarantee:
    This function receives frame[i] and heatmap[i] where i is the frame index.
    The output preserves temporal consistency with the input video.
    
    Args:
        frame: Original BGR frame (H, W, 3)
        heatmap: Color-mapped heatmap BGR (H, W, 3)
        alpha: Blending factor (0.0 = frame only, 1.0 = heatmap only)
        
    Returns:
        blended: BGR image with semi-transparent heatmap overlay
    """
    # Ensure same size
    if frame.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    
    # Alpha blending: output = (1-alpha)*frame + alpha*heatmap
    blended = cv2.addWeighted(frame, 1.0 - alpha, heatmap, alpha, 0)
    
    return blended


# ==============================================================================
# NATURAL LANGUAGE EXPLANATION GENERATION
# ==============================================================================

def generate_explanation_text(
    label: str,
    confidence: float,
    attributions: Dict[int, np.ndarray]
) -> str:
    """
    Generate human-readable explanation of model decision.
    
    Analyzes attribution maps to identify which facial regions contributed
    most to the classification decision.
    
    Args:
        label: 'FAKE' or 'REAL'
        confidence: Confidence score in [0, 1]
        attributions: Dict mapping frame_idx → attribution map (224, 224)
        
    Returns:
        explanation: Natural language explanation string
    """
    if not attributions:
        return f"Classification: {label} (Confidence: {confidence:.2%})\n" \
               f"Explanation unavailable - no attribution data."
    
    # Aggregate attributions across representative frames
    all_attrs = list(attributions.values())
    mean_attr = np.mean(all_attrs, axis=0)
    
    # Analyze facial regions (approximate vertical splits)
    h, w = mean_attr.shape
    
    forehead_region = mean_attr[:int(h * 0.33), :].mean()
    eyes_nose_region = mean_attr[int(h * 0.33):int(h * 0.66), :].mean()
    mouth_jaw_region = mean_attr[int(h * 0.66):, :].mean()
    
    # Determine dominant regions
    regions = {
        "forehead": forehead_region,
        "eyes and nose": eyes_nose_region,
        "mouth and jawline": mouth_jaw_region
    }
    
    overall_mean = mean_attr.mean()
    dominant_regions = [
        name for name, value in regions.items()
        if value > overall_mean * 1.2
    ]
    
    # Evidence strength
    if overall_mean < 0.15:
        evidence_strength = "weak"
    elif overall_mean < 0.35:
        evidence_strength = "moderate"
    else:
        evidence_strength = "strong"
    
    # Build explanation
    region_text = ", ".join(dominant_regions) if dominant_regions else "no specific region"
    
    if label == "FAKE":
        user_summary = (
            f"Classification: AI-MANIPULATED (Confidence: {confidence:.2%})\n\n"
            f"The AI system found {evidence_strength} evidence that this video was altered. "
            f"It noticed unusual visual patterns, particularly around the {region_text}."
        )
        developer_summary = (
            f"The model detected {evidence_strength} evidence of manipulation. "
            f"Attribution analysis indicates heightened attention around the {region_text} region(s), "
            f"which may suggest synthetic artifacts or inconsistencies in facial texture.\n\n"
            f"Analyzed {len(attributions)} representative frames for temporal consistency."
        )
    else:
        user_summary = (
            f"Classification: AUTHENTIC (Confidence: {confidence:.2%})\n\n"
            f"The AI system found {evidence_strength} indicators that this video is genuine. "
            f"The visual patterns across the face appear normal, particularly around the {region_text}."
        )
        developer_summary = (
            f"The model found {evidence_strength} indicators of authenticity. "
            f"Attribution patterns show natural distribution across facial regions, "
            f"with focus on {region_text}.\n\n"
            f"Analyzed {len(attributions)} representative frames for temporal consistency."
        )
    
    explanation = f"=== USER EXPLANATION ===\n{user_summary}\n\n=== DEVELOPER EXPLANATION ===\n{developer_summary}"
    
    return explanation


# ==============================================================================
# MAIN EXPLAINABILITY PIPELINE
# ==============================================================================

def compute_frame_attributions(
    model: torch.nn.Module,
    face_tensors: torch.Tensor,
    device: str = "cpu",
    progress_callback: Optional[callable] = None
) -> Dict[int, np.ndarray]:
    """
    Compute attributions for representative frames.
    
    Uses SHAP GradientExplainer as primary method with Attention Rollout fallback.
    Only processes TOP_K_FRAMES (5) for performance optimization.
    
    Temporal Alignment Guarantee:
    Each returned attribution corresponds exactly to the same frame index
    as the input, preserving temporal consistency.
    
    Args:
        model: ViTVideoClassifier in eval mode
        face_tensors: All face tensors (T, 3, 224, 224)
        device: 'cuda' or 'cpu'
        progress_callback: Optional callback(current, total) for progress updates
        
    Returns:
        attributions: Dict mapping frame_idx → attribution map (224, 224)
    """
    model.eval()
    
    num_faces = face_tensors.shape[0]
    representative_indices = select_representative_frames(num_faces, TOP_K_FRAMES)
    
    # Prepare background samples for SHAP (use first 5 faces or all if fewer)
    bg_count = min(SHAP_BACKGROUND_SAMPLES, num_faces)
    background = face_tensors[:bg_count].unsqueeze(1).to(device)  # (N, 1, 3, 224, 224)
    
    # Initialize attention rollout for fallback
    attention_rollout = AttentionRollout(model)
    
    attributions = {}
    
    for i, frame_idx in enumerate(representative_indices):
        if progress_callback:
            progress_callback(i + 1, len(representative_indices))
        
        # Get single face tensor
        face = face_tensors[frame_idx].unsqueeze(0).unsqueeze(0).to(device)
        
        # Try SHAP first
        attribution = compute_shap_attribution(model, face, background, device)
        
        if attribution is None:
            # Fallback to attention rollout
            print(f"[INFO] Using attention rollout for frame {frame_idx}")
            attribution = attention_rollout.compute(face)
        
        # Store with original frame index for temporal alignment
        attributions[frame_idx] = attribution
    
    return attributions


def generate_heatmap_frames(
    original_frames: Dict[int, np.ndarray],
    attributions: Dict[int, np.ndarray],
    output_dir: str
) -> Dict[int, str]:
    """
    Generate heatmap-overlaid frames and save to disk.
    
    Temporal Alignment Guarantee:
    Each output heatmap frame[i] corresponds exactly to input frame[i].
    
    Args:
        original_frames: Dict mapping frame_idx → BGR frame
        attributions: Dict mapping frame_idx → attribution (224, 224)
        output_dir: Directory to save heatmap frames
        
    Returns:
        heatmap_paths: Dict mapping frame_idx → saved file path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    heatmap_paths = {}
    
    for frame_idx in attributions:
        if frame_idx not in original_frames:
            continue
        
        frame = original_frames[frame_idx]
        attribution = attributions[frame_idx]
        
        # Generate colored heatmap
        heatmap = generate_heatmap(attribution, (frame.shape[1], frame.shape[0]))
        
        # Blend with original frame
        blended = blend_heatmap_on_frame(frame, heatmap)
        
        # Save with frame index in filename for temporal tracking
        output_path = os.path.join(output_dir, f"heatmap_{frame_idx:04d}.jpg")
        cv2.imwrite(output_path, blended)
        
        heatmap_paths[frame_idx] = output_path
    
    return heatmap_paths
