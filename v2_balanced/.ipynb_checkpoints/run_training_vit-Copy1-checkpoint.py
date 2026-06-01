"""
Deepfake Detection Training Script — ViT Spatial Version
===========================================================
Uses the highly discriminative FR_10Mar dataset ViT features 
(768-D CLS token + 14x14 attention map) combined with balanced validation, 
proper CrossEntropy class weights, and gradient clipping.
"""

import sys
import os
import glob
import json
import random
import logging
import argparse
from pathlib import Path
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix
)
from tqdm import tqdm

from model_vit import Config, MultiModalFusionClassifierViT


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("training_vit.log"),
        ],
    )
    return logging.getLogger(__name__)


# ============================================================================
# DATA COLLECTION
# ============================================================================

def collect_video_ids(base_dir):
    video_data = {}
    for fake_real in ["Fake", "Real"]:
        label = 1 if fake_real == "Fake" else 0
        for face_type in ["Single Face", "Multi Face"]:
            category = os.path.join(base_dir, fake_real, face_type)
            if not os.path.exists(category):
                continue
            for video_id in sorted(os.listdir(category)):
                csp_dir = os.path.join(category, video_id, "face_extracted", "csp_tensors_pt")
                if os.path.exists(csp_dir):
                    tensor_files = sorted(glob.glob(os.path.join(csp_dir, "*.pt")))
                    if tensor_files:
                        video_data[video_id] = {
                            "label": label,
                            "csp_dir": csp_dir,
                            "tensor_files": tensor_files,
                            "num_frames": len(tensor_files),
                        }
    return video_data


# ============================================================================
# DATASET
# ============================================================================

class CSPMultiModalDatasetViT(Dataset):
    def __init__(
        self, video_ids, data_dict, labels,
        temporal_dir, spatial_dir,
        max_frames=32, freq_aggregate="mean",
    ):
        self.video_ids = video_ids
        self.data_dict = data_dict
        self.labels = labels
        self.temporal_dir = Path(temporal_dir)
        self.spatial_dir = Path(spatial_dir)
        self.max_frames = max_frames
        self.freq_aggregate = freq_aggregate

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        label = self.labels[video_id]

        # ===== Frequency (CSP) Features =====
        tensor_files = self.data_dict[video_id]["tensor_files"][: self.max_frames]
        csp_tensors = []
        for tf in tensor_files:
            try:
                # Expected tensor shape (8, 2, 256, 256) or (8, 1, 256, 256)
                tensor = torch.load(tf, map_location="cpu")  
                csp_tensors.append(tensor)
            except Exception:
                continue

        if len(csp_tensors) == 0:
            freq_feature = torch.zeros(8, 1, 256, 256)
        else:
            stacked = torch.stack(csp_tensors)        # [T, 8, C, 256, 256]
            # Assuming phase is at channel index 1 (safe fallback to channel 0 if only 1 channel exists)
            c_idx = 1 if stacked.size(2) > 1 else 0
            phase = stacked[:, :, c_idx:c_idx+1, :, :]  # [T, 8, 1, 256, 256]

            if self.freq_aggregate == "mean":
                freq_feature = phase.mean(dim=0)      
            elif self.freq_aggregate == "var":
                freq_feature = phase.var(dim=0, unbiased=False) + 1e-6
            else:
                freq_feature = phase[-1]

        # ===== Temporal Features =====
        temporal_file = self.temporal_dir / f"{video_id}_temporal_features.pt"
        if temporal_file.exists():
            temporal = torch.load(temporal_file, map_location="cpu")
        else:
            temporal = torch.zeros(2, 256, 256)

        # ===== Spatial Features (ViT) =====
        cls_file = self.spatial_dir / f"{video_id}_spatial_embedding.pt"
        act_file = self.spatial_dir / f"{video_id}_activation_map.pt"
        
        if cls_file.exists():
            spatial_cls = torch.load(cls_file, map_location="cpu").flatten()
        else:
            spatial_cls = torch.zeros(768)
            
        if act_file.exists():
            spatial_act = torch.load(act_file, map_location="cpu")
            if spatial_act.dim() == 2:
                spatial_act = spatial_act.unsqueeze(0) # [1, 14, 14] required
        else:
            spatial_act = torch.zeros(1, 14, 14)

        features = {
            "frequency": freq_feature.float(),
            "temporal": temporal.float(),
            "spatial_cls": spatial_cls.float(),
            "spatial_act": spatial_act.float(),
        }

        return features, torch.tensor(label, dtype=torch.long)


# ============================================================================
# EARLY STOPPING
# ============================================================================

class EarlyStopping:
    def __init__(self, patience=8, delta=0.001, path="best_vit_model.pth"):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)


#--------------------------------------------------
#To help you catch that "accuracy flip" in real-time, we’ll implement a Running Class-Accuracy Tracker. This will prevent you from waiting 90 minutes for an epoch to finish only to realize the #model has biased itself again.
#--------------------------------------------------
class ClassAccuracyTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        # Index 0: Real, Index 1: Fake
        self.correct = {0: 0, 1: 0}
        self.total = {0: 0, 1: 0}

    def update(self, logits, labels):
        _, predicted = logits.max(1)
        for label in [0, 1]:
            mask = (labels == label)
            self.correct[label] += (predicted[mask] == labels[mask]).sum().item()
            self.total[label] += mask.sum().item()

    def get_stats(self):
        real_acc = (self.correct[0] / self.total[0] * 100) if self.total[0] > 0 else 0.0
        fake_acc = (self.correct[1] / self.total[1] * 100) if self.total[1] > 0 else 0.0
        return real_acc, fake_acc

# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, logger):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger
        self.device = config.get_device()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=4
        )

        self.criterion = None

        self.best_auc = 0.0
        self.best_raw_acc = 0.0
        self.use_amp = bool(getattr(config, "USE_AMP", True) and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler('cuda',enabled=self.use_amp)

        ckpt_dir = Path(config.CHECKPOINT_DIR)
        ckpt_dir.mkdir(exist_ok=True)
        self.early_stopping = EarlyStopping(
            patience=8, path=str(ckpt_dir / "best_vit_model.pth")
        )

    def set_criterion(self, class_weights, label_smoothing=0.0):
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(self.device),
            label_smoothing=label_smoothing,
        )

    def train_epoch(self, epoch):
         self.model.train()
         tracker = ClassAccuracyTracker()
         total_loss = 0.0
         
         pbar = tqdm(self.train_loader, desc=f"Train Ep {epoch}")

         for i, (features, labels) in enumerate(pbar):
             features = {k: v.to(self.device).float() for k, v in features.items()}
             labels = labels.to(self.device).long()

             self.optimizer.zero_grad()
             with torch.amp.autocast('cuda',enabled=self.use_amp):
                 logits, _ = self.model(features)
                 loss = self.criterion(logits, labels)

             self.scaler.scale(loss).backward()
             self.scaler.unscale_(self.optimizer)
             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.MAX_GRAD_NORM)
             self.scaler.step(self.optimizer)
             self.scaler.update()

             # Update and Log Per-Class Stats
             tracker.update(logits, labels)
             total_loss += loss.item()

             if i % 10 == 0:
                 r_acc, f_acc = tracker.get_stats()
                 pbar.set_postfix({
                     "loss": f"{loss.item():.3f}",
                     "Real%": f"{r_acc:.1f}",
                     "Fake%": f"{f_acc:.1f}"
                 })

         avg_loss = total_loss / len(self.train_loader)
         return avg_loss
        

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        for features, labels in tqdm(self.val_loader, desc="Val"):
            features = {k: v.to(self.device).float() for k, v in features.items()}
            labels = labels.to(self.device).long()

            logits, _ = self.model(features)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            probs = F.softmax(logits, dim=1)
            _, predicted = logits.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)

        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        real_acc = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fake_acc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        balanced_acc = (real_acc + fake_acc) / 2.0

        raw_acc = accuracy_score(all_labels, all_preds)

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.0

        self.logger.info(
            f"Val Epoch {epoch}: Loss={avg_loss:.4f}, BalAcc={balanced_acc * 100:.2f}%, "
            f"RawAcc={raw_acc * 100:.2f}%, RealAcc={real_acc * 100:.1f}%, FakeAcc={fake_acc * 100:.1f}%, AUC={auc:.4f}"
        )

        return avg_loss, balanced_acc, raw_acc, auc

    def train(self):
        self.logger.info(f"Starting training for {self.config.NUM_EPOCHS} epochs")

        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            _ = self.train_epoch(epoch)
            _, bal_acc, raw_acc, auc = self.validate(epoch)

            self.early_stopping(auc, self.model)
            if self.early_stopping.early_stop:
                self.logger.warning(
                    f"Early stopping at Epoch {epoch}! "
                    f"Best AUC={self.best_auc:.4f}, Best RawAcc={self.best_raw_acc*100:.2f}%"
                )
                break

            self.scheduler.step(auc)

            is_best_auc = auc > self.best_auc
            if is_best_auc:
                self.best_auc = auc
                ckpt = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_auc": self.best_auc,
                    "best_raw_acc": self.best_raw_acc,
                }
                torch.save(ckpt, Path(self.config.CHECKPOINT_DIR) / "best_auc_model_full.pth")
                self.logger.info(f"Saved best model — AUC={self.best_auc:.4f}")

            is_best_acc = raw_acc > self.best_raw_acc
            if is_best_acc:
                self.best_raw_acc = raw_acc
                ckpt = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_auc": self.best_auc,
                    "best_raw_acc": self.best_raw_acc,
                }
                torch.save(ckpt, Path(self.config.CHECKPOINT_DIR) / "best_acc_model_full.pth")
                self.logger.info(f"Saved best model — RawAcc={self.best_raw_acc * 100:.2f}%")

        self.logger.info("Training completed!")
        self.logger.info(
            f"Best validation AUC: {self.best_auc:.4f} | "
            f"Best validation RawAcc: {self.best_raw_acc * 100:.2f}%"
        )


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run quick 3-epoch test")
    parser.add_argument(
        "--dataset",
        choices=["FR", "FR_10Mar"],
        default="FR",
        help="Dataset root to use for training",
    )
    parser.add_argument(
        "--ce-weight-power",
        type=float,
        default=0.25,
        help="Class-weight exponent. 0.5 is stronger, 0.25 is milder.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="Label smoothing for CrossEntropyLoss.",
    )
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate override")
    parser.add_argument(
        "--weight-decay", type=float, default=2e-4, help="Weight decay override"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size override")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Prefetch batches per worker (used when num_workers>0)",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="Disable mixed precision (AMP).",
    )
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("====================================================================")
    logger.info("ViT Spatial Features — Deepfake Detection Training")
    logger.info("====================================================================")

    config = Config()
    # Keep 32-frame setup for faster training while stabilizing the pipeline.
    config.MAX_FRAMES = 32
    config.LEARNING_RATE = args.lr
    config.WEIGHT_DECAY = args.weight_decay
    config.BATCH_SIZE = args.batch_size
    config.NUM_WORKERS = args.num_workers
    config.USE_AMP = not args.disable_amp

    # Explicit dataset selection to avoid hidden FR vs FR_10Mar confusion.
    if args.dataset == "FR_10Mar":
        config.BASE_DIR = "/workspace/FR_10Mar/ProcessedDataset"
        config.TEMPORAL_DIR = "/workspace/FR_10Mar/ProcessedDataset/temporal_features_18feb"
        config.SPATIAL_DIR = "/workspace/FR_10Mar/ProcessedDataset/spatial_features_with_masks_updated"
    else:
        config.BASE_DIR = "/workspace/FR/ProcessedDataset"
        config.TEMPORAL_DIR = "/workspace/FR/ProcessedDataset/temporal_features_18feb"
        config.SPATIAL_DIR = "/workspace/FR/ProcessedDataset/spatial_features_with_masks_updated"

    device = config.get_device()

    if args.debug:
        logger.warning("!!! DEBUG MODE !!!")

    logger.info(f"[1/4] Collecting Videos from {args.dataset}")
    logger.info(f"BASE_DIR={config.BASE_DIR}")
    logger.info(f"TEMPORAL_DIR={config.TEMPORAL_DIR}")
    logger.info(f"SPATIAL_DIR={config.SPATIAL_DIR}")
    logger.info(
        f"Hyperparams: lr={config.LEARNING_RATE}, wd={config.WEIGHT_DECAY}, "
        f"label_smoothing={args.label_smoothing}, ce_weight_power={args.ce_weight_power}, "
        f"max_frames={config.MAX_FRAMES}, batch_size={config.BATCH_SIZE}, "
        f"num_workers={config.NUM_WORKERS}, amp={config.USE_AMP}"
    )
    data_dict = collect_video_ids(config.BASE_DIR)
    
    if len(data_dict) == 0:
        logger.error("No videos found! Check BASE_DIR.")
        return

    all_video_ids = list(data_dict.keys())
    labels = {vid: d["label"] for vid, d in data_dict.items()}

    logger.info(f"✓ Found {len(all_video_ids)} videos (Real: {list(labels.values()).count(0)}, Fake: {list(labels.values()).count(1)})")

    logger.info("\n[2/4] Creating Balanced Val Split")
    labels_list = [labels[vid] for vid in all_video_ids]
    train_ids, val_ids = train_test_split(all_video_ids, test_size=0.25, stratify=labels_list, random_state=42)

    val_real = [v for v in val_ids if labels[v] == 0]
    val_fake = [v for v in val_ids if labels[v] == 1]
    min_val = min(len(val_real), len(val_fake))
    
    random.seed(42)
    val_ids_balanced = random.sample(val_real, min_val) + random.sample(val_fake, min_val)
    random.shuffle(val_ids_balanced)

    if args.debug:
        train_ids = [v for v in train_ids if labels[v] == 0][:16] + [v for v in train_ids if labels[v] == 1][:16]
        val_ids_balanced = val_ids_balanced[:16]
        config.NUM_EPOCHS = 3

    logger.info("\n[3/4] Creating Datasets & Loaders")
    train_dataset = CSPMultiModalDatasetViT(train_ids, data_dict, labels, config.TEMPORAL_DIR, config.SPATIAL_DIR, config.MAX_FRAMES)
    val_dataset = CSPMultiModalDatasetViT(val_ids_balanced, data_dict, labels, config.TEMPORAL_DIR, config.SPATIAL_DIR, config.MAX_FRAMES)

    train_labels = [labels[vid] for vid in train_ids]
    class_counts = [train_labels.count(0), train_labels.count(1)]
    #target_ratios = torch.tensor([0.30, 0.70]) # 30% Real, 70% Fake
    #weights = target_ratios / torch.tensor(class_counts, dtype=torch.float)
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [weights[l] for l in train_labels]

    train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    loader_common = {
        "batch_size": config.BATCH_SIZE,
        "num_workers": config.NUM_WORKERS,
        "pin_memory": True if device.type == "cuda" else False,
    }
    if config.NUM_WORKERS > 0:
        loader_common["persistent_workers"] = True
        loader_common["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        **loader_common,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_common,
    )

    logger.info("\n[4/4] Initializing ViT Model")
    model = MultiModalFusionClassifierViT(config).to(device)

    trainer = Trainer(model, train_loader, val_loader, config, logger)

    # Mild class weights (reduced from sqrt to lower over-focus on minority class).
    ce_weights = torch.tensor([
        1.0 / (class_counts[0] ** args.ce_weight_power),
        1.0 / (class_counts[1] ** args.ce_weight_power),
    ])
    ce_weights = ce_weights / ce_weights.sum() * 2.0
    trainer.set_criterion(ce_weights, label_smoothing=args.label_smoothing)
    logger.info(
        f"CE weights => Real={ce_weights[0]:.4f}, Fake={ce_weights[1]:.4f}, "
        f"ratio={ce_weights[0]/ce_weights[1]:.3f}"
    )

    logger.info("\nStarting Training...")
    
    trainer.train()

if __name__ == "__main__":
    main()
