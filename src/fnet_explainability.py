# ==============================================================================
# fnet_explainability.py - F-Net Metadata-Based Explainability
# ==============================================================================
# Generates human-readable explanations of F-Net predictions by analyzing:
#   1. Model confidence and stream importance
#   2. Temporal metadata anomalies (rPPG, blink, lip movement)
#   3. Spatial region analysis
#
# Adapted from F_NET/F_Net_CNN/metadata_explainer.py for runtime use.
# Unlike the original, this receives metadata directly from feature extraction
# instead of loading from batch JSON files.
# ==============================================================================

import numpy as np
import torch
from typing import Dict, List, Any, Optional


# ==============================================================================
# REGION LABELS
# ==============================================================================

REGION_LABELS = {
    0: "background",
    1: "forehead",
    2: "left cheek",
    3: "right cheek",
    4: "eyes",
    5: "lips"
}

# Approximate baselines for real videos (from training data analysis)
# These are rough median values for real videos — used as comparison reference
REAL_BASELINES = {
    "forehead_mean_rppg": {"median": 120.0, "iqr": 15.0},
    "left_cheek_mean_rppg": {"median": 118.0, "iqr": 14.0},
    "right_cheek_mean_rppg": {"median": 119.0, "iqr": 14.0},
    "forehead_peak_rppg": {"median": 3, "iqr": 2},
    "left_cheek_peak_rppg": {"median": 3, "iqr": 2},
    "right_cheek_peak_rppg": {"median": 3, "iqr": 2},
    "eyes_blink_acceleration": {"median": 1.5, "iqr": 1.0},
    "eyes_blink_rate": {"median": 0.15, "iqr": 0.1},
    "lips_lip_movement": {"median": 2.0, "iqr": 1.5},
    "interframe_corr": {"median": 0.95, "iqr": 0.05},
}


# ==============================================================================
# ANOMALY DETECTION
# ==============================================================================

def detect_anomalies(temporal_meta: Dict, confidence: float) -> List[Dict]:
    """
    Detect anomalies in temporal metadata vs real video baselines.

    Args:
        temporal_meta: Dict from fnet_features temporal extraction
        confidence: Model confidence (fake probability)

    Returns:
        List of anomaly dicts sorted by severity
    """
    anomalies = []
    features = temporal_meta.get("features", {})

    for region, data in features.items():
        if isinstance(data, dict):
            for key, val in data.items():
                if isinstance(val, (int, float)):
                    baseline_key = f"{region}_{key}"
                    baseline = REAL_BASELINES.get(baseline_key)
                    if baseline:
                        zscore = abs(val - baseline["median"]) / (baseline["iqr"] + 1e-6)
                        if zscore > 1.5:
                            anomalies.append({
                                "type": baseline_key,
                                "region": region,
                                "value": val,
                                "zscore": zscore,
                                "explanation": (
                                    f"Abnormal {key} in {region} "
                                    f"(value: {val:.3f} vs normal {baseline['median']:.3f})"
                                )
                            })

    # Inter-frame correlation
    corr = features.get("interframe_corr", 0.95)
    baseline = REAL_BASELINES.get("interframe_corr", {"median": 0.95, "iqr": 0.05})
    if abs(corr - baseline["median"]) > 0.1:
        anomalies.append({
            "type": "interframe_corr",
            "region": "global",
            "value": corr,
            "zscore": abs(corr - baseline["median"]) / 0.1,
            "explanation": (
                f"Poor frame consistency "
                f"(corr: {corr:.3f} vs normal {baseline['median']:.3f})"
            )
        })

    return sorted(anomalies, key=lambda x: x["zscore"], reverse=True)[:5]


# ==============================================================================
# MODAL IMPORTANCE
# ==============================================================================

def compute_modal_importance(stream_outputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Approximate per-modal importance from feature norms.

    Args:
        stream_outputs: Dict with 'frequency', 'temporal', 'spatial' tensors
    """
    scores = {}
    for key in ('frequency', 'temporal', 'spatial'):
        if key in stream_outputs and isinstance(stream_outputs[key], torch.Tensor):
            scores[key] = float(torch.norm(stream_outputs[key], p=2).item())

    total = sum(scores.values()) or 1.0
    return {k: v / total for k, v in scores.items()}


# ==============================================================================
# MAIN EXPLANATION GENERATOR
# ==============================================================================

def explain_fnet_prediction(
    logits: torch.Tensor,
    stream_outputs: Dict[str, torch.Tensor],
    temporal_metadata: Dict,
    spatial_metadata: Dict
) -> Dict[str, Any]:
    """
    Generate comprehensive explanation of F-Net prediction.

    Args:
        logits: Model output (1, 2) — class logits [real, fake]
        stream_outputs: Per-stream feature vectors from model
        temporal_metadata: From fnet_features.extract_temporal_features
        spatial_metadata: From fnet_features.extract_spatial_features

    Returns:
        Dict with prediction, confidence, summary, anomalies, etc.
    """
    # Compute probabilities
    probs = torch.softmax(logits, dim=-1)
    fake_prob = probs[0, 1].item()
    prediction = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = fake_prob if prediction == "FAKE" else 1.0 - fake_prob

    # Detect anomalies from temporal metadata
    anomalies = detect_anomalies(temporal_metadata, fake_prob)

    # Compute per-stream importance
    modal_importance = compute_modal_importance(stream_outputs)

    # Identify suspect regions from anomalies
    suspect_regions = set()
    for a in anomalies:
        region = a.get("region", "")
        if region and region != "global":
            suspect_regions.add(region)

    suspect_list = sorted(suspect_regions)

    # Build human-readable summary
    if prediction == "FAKE":
        evidence_parts = []
        if anomalies:
            evidence_parts.append(
                "; ".join([a["explanation"] for a in anomalies[:3]])
            )
        if suspect_list:
            evidence_parts.append(
                f"Suspect regions: {', '.join(suspect_list)}"
            )

        # Modal contribution
        modal_parts = []
        for k, v in sorted(modal_importance.items(), key=lambda x: -x[1]):
            modal_parts.append(f"{k}: {v:.0%}")
        if modal_parts:
            evidence_parts.append(
                f"Stream importance: {', '.join(modal_parts)}"
            )

        evidence_text = ". ".join(evidence_parts) if evidence_parts else "No specific anomalies detected."

        user_summary = (
            f"Classification: AI-MANIPULATED (Confidence: {confidence:.2%})\n\n"
            f"The AI system detected signs that this video has been altered. "
            f"It found unnatural patterns in the facial features, such as irregular blinking, "
            f"inconsistent lip movements, or unnatural micro-color changes (heartbeat)."
        )
        
        developer_summary = (
            f"F-Net multi-modal analysis detected evidence of manipulation.\n"
            f"{evidence_text}\n\n"
            f"Analyzed using frequency (CSP), temporal (rPPG/blink/lip), "
            f"and spatial (ViT + region masks) streams."
        )
    else:
        user_summary = (
            f"Classification: AUTHENTIC (Confidence: {confidence:.2%})\n\n"
            f"The AI system found no strong evidence of manipulation. "
            f"Facial movements, blinking, and micro-color changes (heartbeat) "
            f"appear natural and consistent."
        )
        
        developer_summary = (
            f"F-Net multi-modal analysis found no strong evidence of manipulation. "
            f"Temporal physiological signals (rPPG, blink patterns, lip movement) "
            f"appear consistent with authentic video.\n\n"
            f"Analyzed using frequency (CSP), temporal (rPPG/blink/lip), "
            f"and spatial (ViT + region masks) streams."
        )

    summary = f"=== USER EXPLANATION ===\n{user_summary}\n\n=== DEVELOPER EXPLANATION ===\n{developer_summary}"

    return {
        "prediction": prediction,
        "confidence": confidence,
        "fake_probability": fake_prob,
        "summary": summary,
        "top_anomalies": anomalies,
        "suspect_regions": suspect_list,
        "modal_importance": modal_importance,
    }
