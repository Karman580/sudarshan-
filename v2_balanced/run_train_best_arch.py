"""
Train Best NAS Architecture from Scratch
==========================================
Reads the best architecture from `best_architecture.json` (output of
`run_nas_search.py`) and trains it with full schedule + early stopping.

Usage:
    python run_train_best_arch.py --arch nas_results/best_architecture.json
    python run_train_best_arch.py --arch nas_results/best_architecture.json \
           --epochs 150 --batch-size 128
"""

import sys
import os
import json
import random
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F_torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix,
)
from tqdm import tqdm

from nas_search_space import SearchSpace
from nas_model import NASMultiModalClassifier
from run_training_vit import (
    collect_video_ids,
    CSPMultiModalDatasetViT,
    ClassAccuracyTracker,
)


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(log_file: str = "train_best_arch.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [BEST-ARCH] %(levelname)s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )
    return logging.getLogger("BestArch")


# ============================================================================
# EARLY STOPPING
# ============================================================================

class EarlyStopping:
    def __init__(self, patience=10, delta=0.001, path="best_nas_model.pth"):
        self.patience, self.delta, self.path = patience, delta, path
        self.counter, self.best_score, self.early_stop = 0, None, False

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score, self.counter = score, 0
            self.save_checkpoint(model)

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)


# ============================================================================
# TRAINER
# ============================================================================

class BestArchTrainer:
    def __init__(self, model, train_loader, val_loader, arch, device, logger,
                 args, class_weights):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.arch = arch
        self.device = device
        self.logger = logger
        self.args = args

        # Optimizer from NAS config
        lr = arch["learning_rate"]
        wd = arch["weight_decay"]
        if arch["optimizer"] == "adamw":
            self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        else:
            self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                       weight_decay=wd)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=1e-7,
        )

        ls = arch["label_smoothing"]
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device), label_smoothing=ls,
        )

        self.use_amp = device.type == "cuda"
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.best_auc = 0.0
        self.best_raw_acc = 0.0

        ckpt_dir = Path(args.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.early_stopping = EarlyStopping(
            patience=12, path=str(ckpt_dir / "best_nas_model.pth"),
        )
        self.ckpt_dir = ckpt_dir

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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            tracker.update(logits, labels)
            total_loss += loss.item()

            if i % 10 == 0:
                r_acc, f_acc = tracker.get_stats()
                pbar.set_postfix({"loss": f"{loss.item():.3f}",
                                  "R%": f"{r_acc:.1f}", "F%": f"{f_acc:.1f}"})

        return total_loss / max(len(self.train_loader), 1)

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []

        for features, labels in tqdm(self.val_loader, desc="Val"):
            features = {k: v.to(self.device).float() for k, v in features.items()}
            labels = labels.to(self.device).long()
            logits, _ = self.model(features)
            probs = F_torch.softmax(logits, dim=1)
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

        prec, rec, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="binary", zero_division=0,
        )

        self.logger.info(
            f"Val Ep {epoch}: BalAcc={balanced_acc*100:.2f}%, "
            f"RealAcc={real_acc*100:.1f}%, FakeAcc={fake_acc*100:.1f}%, "
            f"AUC={auc:.4f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}"
        )
        return balanced_acc, raw_acc, auc

    def train(self):
        self.logger.info(f"Starting full training for {self.args.epochs} epochs")
        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.train_epoch(epoch)
            bal_acc, raw_acc, auc = self.validate(epoch)

            self.early_stopping(auc, self.model)
            self.scheduler.step()

            lr_now = self.optimizer.param_groups[0]["lr"]
            self.logger.info(f"  Ep {epoch} summary: train_loss={train_loss:.4f}, "
                             f"AUC={auc:.4f}, LR={lr_now:.2e}")

            if auc > self.best_auc:
                self.best_auc = auc
                self._save_full_checkpoint(epoch, "best_auc_nas_full.pth")

            if raw_acc > self.best_raw_acc:
                self.best_raw_acc = raw_acc
                self._save_full_checkpoint(epoch, "best_acc_nas_full.pth")

            if self.early_stopping.early_stop:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        self.logger.info(f"\nTraining complete — Best AUC: {self.best_auc:.4f}, "
                         f"Best Acc: {self.best_raw_acc:.4f}")

    def _save_full_checkpoint(self, epoch, name):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_auc": self.best_auc,
            "best_raw_acc": self.best_raw_acc,
            "architecture": self.arch,
        }
        path = self.ckpt_dir / name
        torch.save(ckpt, path)
        self.logger.info(f"  Saved checkpoint: {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train best NAS-discovered architecture from scratch",
    )
    parser.add_argument("--arch", type=str, required=True,
                        help="Path to best_architecture.json")
    parser.add_argument("--dataset", choices=["FR", "FR_10Mar"], default="FR")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_nas")
    args = parser.parse_args()

    logger = setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load architecture
    arch = SearchSpace.load(args.arch)
    logger.info(f"Loaded architecture from: {args.arch}")
    logger.info("Architecture config:")
    for k, v in arch.items():
        logger.info(f"  {k}: {v}")

    # Data
    if args.dataset == "FR_10Mar":
        base_dir = "/workspace/FR_10Mar/ProcessedDataset"
    else:
        base_dir = "/workspace/FR/ProcessedDataset"

    temporal_dir = f"{base_dir}/temporal_features_18feb"
    spatial_dir = f"{base_dir}/spatial_features_with_masks_updated"

    data_dict = collect_video_ids(base_dir)
    all_vids = list(data_dict.keys())
    labels = {vid: d["label"] for vid, d in data_dict.items()}

    logger.info(f"Total videos: {len(all_vids)}")

    train_ids, val_ids = train_test_split(
        all_vids, test_size=0.25,
        stratify=[labels[v] for v in all_vids], random_state=42,
    )

    # Balanced validation
    v_real = [v for v in val_ids if labels[v] == 0]
    v_fake = [v for v in val_ids if labels[v] == 1]
    min_v = min(len(v_real), len(v_fake))
    val_ids_balanced = random.sample(v_real, min_v) + random.sample(v_fake, min_v)

    train_ds = CSPMultiModalDatasetViT(
        train_ids, data_dict, labels, temporal_dir, spatial_dir,
    )
    val_ds = CSPMultiModalDatasetViT(
        val_ids_balanced, data_dict, labels, temporal_dir, spatial_dir,
    )

    # Class weights
    train_labels = [labels[vid] for vid in train_ids]
    counts = [train_labels.count(0), train_labels.count(1)]
    class_weights = torch.tensor([1.0 / c for c in counts], dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * 2.0

    # Weighted sampler
    sample_weights = torch.tensor([1.0 / counts[l] for l in train_labels])
    sampler = WeightedRandomSampler(sample_weights, len(train_labels), replacement=True)

    loader_cfg = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    train_loader = DataLoader(train_ds, sampler=sampler, **loader_cfg)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_cfg)

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} (balanced)")

    # Build model
    model = NASMultiModalClassifier(arch).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {total_params:,} total, {trainable_params:,} trainable")

    # Train
    trainer = BestArchTrainer(
        model, train_loader, val_loader, arch, device, logger, args, class_weights,
    )
    trainer.train()

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best AUC: {trainer.best_auc:.4f}")
    logger.info(f"Best Acc: {trainer.best_raw_acc:.4f}")
    logger.info(f"Checkpoints saved to: {args.checkpoint_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
