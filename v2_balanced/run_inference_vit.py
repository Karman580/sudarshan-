"""
Run Inference, Save JSON, and Generate Heatmaps (v2_balanced ViT Pipeline)
=============================================================================
Tests 4 videos natively using the trained v2_balanced ViT Model and triggers 
the spatial / temporal explainers to generate explanations & heatmaps.
"""

import sys
import torch
import logging
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

# Add paths to borrow the exact same explainers from the `new` folder
current_dir = Path(__file__).resolve().parent
new_dir = current_dir.parent / "new"
fnet_cnn_dir = current_dir.parent / "F_Net_CNN"

sys.path.insert(0, str(new_dir))
sys.path.insert(0, str(fnet_cnn_dir))

from data_loader_utils import VideoLabelMapper, FrequencyDataLoader
from vit_metadata_explainer import ViTMetadataExplainer

# Import our robust trained model
from model_vit import MultiModalFusionClassifierViT, Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    config = Config()
    config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # We use the exactly matched FR paths from our Config
    BASE_DIR = Path(config.BASE_DIR)
    TEMPORAL_DIR = Path(config.TEMPORAL_DIR)
    SPATIAL_DIR = Path(config.SPATIAL_DIR)
    
    # The output checkpoint path we used in run_training_vit.py
    MODEL_PATH = Path(config.CHECKPOINT_DIR) / "best_vit_model.pth"
    OUTPUT_DIR = Path("explanations")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    from run_training_vit import collect_video_ids
    
    logger.info("Loading video labels from folder structure...")
    data_dict = collect_video_ids(BASE_DIR)
    all_video_ids = list(data_dict.keys())
    labels = {vid: d["label"] for vid, d in data_dict.items()}
    
    # We still need the metadata files for the explainer heatmaps
    metadata_files = list(TEMPORAL_DIR.glob("*metadata*.json")) + \
                     list(SPATIAL_DIR.glob("*metadata*.json"))
    
    # Strictly test EXACTLY 4 videos (e.g., first 2 real, first 2 fake)
    reals = [vid for vid in all_video_ids if labels[vid] == 0]
    fakes = [vid for vid in all_video_ids if labels[vid] == 1]
    
    if len(reals) < 2 or len(fakes) < 2:
        logger.error("Not enough videos found in the dataset to run inference tests (need at least 2 real and 2 fake).")
        return
        
    test_video_ids = reals[:2] + fakes[:2]
    logger.info(f"✓ Testing explicitly 4 videos: {test_video_ids}")
    
    explainer = ViTMetadataExplainer(
        temporal_meta_dir=str(TEMPORAL_DIR),
        spatial_meta_dir=str(SPATIAL_DIR),
    )
    freq_loader = FrequencyDataLoader(BASE_DIR)
    
    logger.info("Loading trained robust v2_balanced ViT model...")
    model = MultiModalFusionClassifierViT(config)
    
    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location=config.DEVICE, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict']) 
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"✓ Weights loaded from {MODEL_PATH}!")
    else:
        logger.warning(f"! No model weights found at {MODEL_PATH}. Did you run training first? Proceeding with uninitialized weights to test pipelines.")
        
    model.to(config.DEVICE).eval()
    
    results = []
    with torch.no_grad():
        for video_id in tqdm(test_video_ids, desc="Analyzing 4 Videos"):
            label = labels.get(video_id, 1) # Default to fake if missing
            
            try:
                # 1. Frequency
                expected_freq_shape = (8, 1, 256, 256)
                try:
                    # In v2_balanced config we aggregate Phase over time
                    freq_feat = freq_loader.load_video_csp_features(
                        video_id, label, use_phase_only=True, aggregate='mean'
                    )
                    if freq_feat.shape != expected_freq_shape: 
                        freq_feat = torch.zeros(*expected_freq_shape)
                except Exception:
                    freq_feat = torch.zeros(*expected_freq_shape)
                
                # 2. Temporal    
                temp_path = TEMPORAL_DIR / f"{video_id}_temporal_features.pt"
                temp_feat = torch.load(temp_path, map_location='cpu') if temp_path.exists() else torch.zeros(2, 256, 256)
                
                # 3. Spatial ViT (CLS Token + Activation Map)
                spat_emb_path = SPATIAL_DIR / f"{video_id}_spatial_embedding.pt"
                spat_emb = torch.load(spat_emb_path, map_location='cpu').flatten() if spat_emb_path.exists() else torch.zeros(768)
                
                spat_map_path = SPATIAL_DIR / f"{video_id}_activation_map.pt"
                spat_map = torch.load(spat_map_path, map_location='cpu') if spat_map_path.exists() else torch.zeros(14, 14)
                if spat_map.dim() == 2:
                    spat_map = spat_map.unsqueeze(0) # [1, 14, 14]
                
                # Build dict for v2_balanced model format
                features = {
                    'frequency': freq_feat.unsqueeze(0).to(config.DEVICE).float(),
                    'temporal': temp_feat.unsqueeze(0).to(config.DEVICE).float(),
                    'spatial_cls': spat_emb.unsqueeze(0).to(config.DEVICE).float(),
                    'spatial_act': spat_map.unsqueeze(0).to(config.DEVICE).float()
                }
                
                # Forward Pass
                logits, stream_outputs = model(features)

                # Generate Explanation Text using the original class
                modal_importance = {'spatial': 0.8, 'temporal': 0.15, 'frequency': 0.05} 
                
                explanation = explainer.explain_prediction(
                    video_id,
                    logits,
                    modal_importance=modal_importance,
                    freq_meta=None,
                )
                
                explanation['true_label'] = 'real' if label == 0 else 'fake'
                explanation['video_id'] = video_id
                results.append(explanation)
                
                # EXTRACT FRAME FOR HEATMAP
                # We pull it directly from the label_mapper which already parsed the JSON
                frame_path = None
                for meta_file in metadata_files:
                    if not meta_file.exists(): continue
                    with open(meta_file, 'r') as f:
                        data = json.load(f)
                    
                    if isinstance(data, dict):
                        if 'video_id' not in data: data = list(data.values())
                        else: data = [data]
                        
                    for entry in data:
                        if isinstance(entry, dict) and entry.get('video_id') == video_id:
                            f_paths = entry.get('frame_files', entry.get('framefiles', []))
                            if isinstance(f_paths, list) and f_paths:
                                frame_path = f_paths[len(f_paths)//2]
                            elif isinstance(f_paths, str):
                                frame_path = f_paths
                            break
                    if frame_path: break
                
                # GENERATE HEATMAP VISUALS
                act_map_np = spat_map.squeeze(0).cpu().numpy() # [14, 14]
                explainer.generate_heatmaps(video_id, explanation, act_map_np, frame_path, str(OUTPUT_DIR))
                
            except Exception as e:
                logger.error(f"❌ Failed processing video {video_id}: {e}")
                continue
    
    # Save the huge generic explanations file
    output_file = OUTPUT_DIR / "valid_explanations.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Saved initial explain dump to {output_file}")
    
    # Trigger final formatting to human-readable format
    try:
        import format_explanations
        logger.info("Formatting explanations to final file (formatted_explanations.txt)...")
        
        # Override output directory if format_explanations has that option, or run it
        # Note: we temporarily copy it if needed, or run the original.
        format_explanations.main()
    except Exception as e:
        logger.warning(f"! Could not run format_explanations.py: {e}")

if __name__ == "__main__":
    main()
