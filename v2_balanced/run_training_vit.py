"""
Deepfake Detection Training Script — ViT Spatial Version
===========================================================
- Resumes from 'best_auc_model_full.pth' if available.
- Optimized for DGX nodes (Multi-worker, High Batch Size).
- Includes ClassAccuracyTracker for real-time bias monitoring.
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
# LOGGING & TRACKING
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
            if mask.any():
                self.correct[label] += (predicted[mask] == labels[mask]).sum().item()
                self.total[label] += mask.sum().item()

    def get_stats(self):
        real_acc = (self.correct[0] / self.total[0] * 100) if self.total[0] > 0 else 0.0
        fake_acc = (self.correct[1] / self.total[1] * 100) if self.total[1] > 0 else 0.0
        return real_acc, fake_acc

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

        # Frequency (CSP)
        tensor_files = self.data_dict[video_id]["tensor_files"][: self.max_frames]
        csp_tensors = []
        for tf in tensor_files:
            try:
                tensor = torch.load(tf, map_location="cpu")  
                csp_tensors.append(tensor)
            except: continue

        if len(csp_tensors) == 0:
            freq_feature = torch.zeros(8, 1, 256, 256)
        else:
            stacked = torch.stack(csp_tensors)
            c_idx = 1 if stacked.size(2) > 1 else 0
            phase = stacked[:, :, c_idx:c_idx+1, :, :]
            freq_feature = phase.mean(dim=0) if self.freq_aggregate == "mean" else phase[-1]

        # Temporal
        temporal_file = self.temporal_dir / f"{video_id}_temporal_features.pt"
        temporal = torch.load(temporal_file, map_location="cpu") if temporal_file.exists() else torch.zeros(2, 256, 256)

        # Spatial
        cls_file = self.spatial_dir / f"{video_id}_spatial_embedding.pt"
        act_file = self.spatial_dir / f"{video_id}_activation_map.pt"
        spatial_cls = torch.load(cls_file, map_location="cpu").flatten() if cls_file.exists() else torch.zeros(768)
        spatial_act = torch.load(act_file, map_location="cpu") if act_file.exists() else torch.zeros(1, 14, 14)
        if spatial_act.dim() == 2: spatial_act = spatial_act.unsqueeze(0)

        return {
            "frequency": freq_feature.float(),
            "temporal": temporal.float(),
            "spatial_cls": spatial_cls.float(),
            "spatial_act": spatial_act.float(),
        }, torch.tensor(label, dtype=torch.long)


# ============================================================================
# TRAINER & EARLY STOPPING
# ============================================================================

class EarlyStopping:
    def __init__(self, patience=8, delta=0.001, path="best_vit_model.pth"):
        self.patience, self.delta, self.path = patience, delta, path
        self.counter, self.best_score, self.early_stop = 0, None, False

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score, self.counter = score, 0
            self.save_checkpoint(model)

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, logger):
        self.model, self.train_loader, self.val_loader = model, train_loader, val_loader
        self.config, self.logger, self.device = config, logger, config.get_device()

        self.optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5, patience=4)
        
        self.best_auc, self.best_raw_acc = 0.0, 0.0
        self.use_amp = bool(getattr(config, "USE_AMP", True) and self.device.type == "cuda")
        
        # Modern AMP syntax
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.early_stopping = EarlyStopping(patience=8, path=str(Path(config.CHECKPOINT_DIR) / "best_vit_model.pth"))

    def set_criterion(self, class_weights, label_smoothing=0.0):
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device), label_smoothing=label_smoothing)

    def train_epoch(self, epoch):
        self.model.train()
        tracker = ClassAccuracyTracker()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Train Ep {epoch}")

        for i, (features, labels) in enumerate(pbar):
            features = {k: v.to(self.device).float() for k, v in features.items()}
            labels = labels.to(self.device).long()

            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits, _ = self.model(features)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.MAX_GRAD_NORM)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            tracker.update(logits, labels)
            total_loss += loss.item()

            if i % 10 == 0:
                r_acc, f_acc = tracker.get_stats()
                pbar.set_postfix({"loss": f"{loss.item():.3f}", "R%": f"{r_acc:.1f}", "F%": f"{f_acc:.1f}"})

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []

        for features, labels in tqdm(self.val_loader, desc="Val"):
            features = {k: v.to(self.device).float() for k, v in features.items()}
            labels = labels.to(self.device).long()
            logits, _ = self.model(features)
            probs = F.softmax(logits, dim=1)
            all_preds.extend(logits.max(1)[1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        real_acc = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fake_acc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        balanced_acc = (real_acc + fake_acc) / 2.0
        raw_acc = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)

        self.logger.info(f"Val Ep {epoch}: Loss={all_probs[0]:.4f}, BalAcc={balanced_acc*100:.2f}%, RealAcc={real_acc*100:.1f}%, FakeAcc={fake_acc*100:.1f}%, AUC={auc:.4f}")
        return balanced_acc, raw_acc, auc

    def train(self, start_epoch=1):
        self.logger.info(f"Starting training from Epoch {start_epoch}")
        for epoch in range(start_epoch, self.config.NUM_EPOCHS + 1):
            _ = self.train_epoch(epoch)
            bal_acc, raw_acc, auc = self.validate(epoch)

            self.early_stopping(auc, self.model)
            self.scheduler.step(auc)

            if auc > self.best_auc:
                self.best_auc = auc
                self.save_full_checkpoint(epoch, "best_auc_model_full.pth")
            
            if raw_acc > self.best_raw_acc:
                self.best_raw_acc = raw_acc
                self.save_full_checkpoint(epoch, "best_acc_model_full.pth")

            if self.early_stopping.early_stop: break

    def save_full_checkpoint(self, epoch, name):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_auc": self.best_auc,
            "best_raw_acc": self.best_raw_acc,
        }
        torch.save(ckpt, Path(self.config.CHECKPOINT_DIR) / name)
        self.logger.info(f"Saved checkpoint: {name}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["FR", "FR_10Mar"], default="FR")
    parser.add_argument("--ce-weight-power", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-5) # Resuming often allows higher LR
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128) # Higher for DGX
    parser.add_argument("--num-workers", type=int, default=8)  # Higher for DGX
    args = parser.parse_args()

    logger = setup_logging()
    config = Config()
    config.LEARNING_RATE, config.WEIGHT_DECAY = args.lr, args.weight_decay
    config.BATCH_SIZE, config.NUM_WORKERS = args.batch_size, args.num_workers

    if args.dataset == "FR_10Mar":
        config.BASE_DIR = "/workspace/FR_10Mar/ProcessedDataset"
    else:
        config.BASE_DIR = "/workspace/FR/ProcessedDataset"
    
    config.TEMPORAL_DIR = f"{config.BASE_DIR}/temporal_features_18feb"
    config.SPATIAL_DIR = f"{config.BASE_DIR}/spatial_features_with_masks_updated"

    data_dict = collect_video_ids(config.BASE_DIR)
    all_vids = list(data_dict.keys())
    labels = {vid: d["label"] for vid, d in data_dict.items()}
    train_ids, val_ids = train_test_split(all_vids, test_size=0.25, stratify=[labels[v] for v in all_vids], random_state=42)

    # Balanced Val
    v_real, v_fake = [v for v in val_ids if labels[v]==0], [v for v in val_ids if labels[v]==1]
    min_v = min(len(v_real), len(v_fake))
    val_ids_balanced = random.sample(v_real, min_v) + random.sample(v_fake, min_v)

    train_ds = CSPMultiModalDatasetViT(train_ids, data_dict, labels, config.TEMPORAL_DIR, config.SPATIAL_DIR)
    val_ds = CSPMultiModalDatasetViT(val_ids_balanced, data_dict, labels, config.TEMPORAL_DIR, config.SPATIAL_DIR)

    train_labels = [labels[vid] for vid in train_ids]
    counts = [train_labels.count(0), train_labels.count(1)]
    weights = 1.0 / torch.tensor(counts, dtype=torch.float)
    sampler = WeightedRandomSampler([weights[l] for l in train_labels], len(train_labels), replacement=True)

    loader_cfg = {"batch_size": config.BATCH_SIZE, "num_workers": config.NUM_WORKERS, "pin_memory": True}
    train_loader = DataLoader(train_ds, sampler=sampler, **loader_cfg)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_cfg)

    model = MultiModalFusionClassifierViT(config).to(config.get_device())
    trainer = Trainer(model, train_loader, val_loader, config, logger)

    # ==========================================
    # RESUME LOGIC
    # ==========================================
    start_epoch = 1
    checkpoint_path = Path(config.CHECKPOINT_DIR) / "best_auc_model_full.pth"
    
    if checkpoint_path.exists():
        logger.info(f"==> Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config.get_device())
        
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        trainer.best_auc = checkpoint.get('best_auc', 0.0)
        trainer.best_raw_acc = checkpoint.get('best_raw_acc', 0.0)
        start_epoch = checkpoint.get('epoch', 1) + 1
        logger.info(f"Checkpoint loaded. Resuming at Epoch {start_epoch} with baseline AUC {trainer.best_auc:.4f}")

    ce_weights = torch.tensor([1.0/(counts[0]**args.ce_weight_power), 1.0/(counts[1]**args.ce_weight_power)])
    trainer.set_criterion(ce_weights / ce_weights.sum() * 2.0, label_smoothing=args.label_smoothing)

    trainer.train(start_epoch=start_epoch)

if __name__ == "__main__":
    main()