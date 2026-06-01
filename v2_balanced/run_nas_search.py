"""
Neural Architecture Search — Search Controller
=================================================
Two-phase pipeline:
  Phase 1: Train a supernet with random-path sampling.
  Phase 2: Evolutionary search to find the best sub-architecture.

Usage:
    python run_nas_search.py                          # full run
    python run_nas_search.py --supernet-epochs 1 \
           --search-generations 2 --population-size 4  # quick smoke-test
"""

import sys
import os
import copy
import json
import logging
import random
import argparse
import glob
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F_torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from tqdm import tqdm

from nas_search_space import SearchSpace, SEARCH_SPACE
from nas_model import NASMultiModalClassifier

# Reuse existing data utilities
from run_training_vit import (
    collect_video_ids,
    CSPMultiModalDatasetViT,
    ClassAccuracyTracker,
)


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(log_file: str = "nas_search.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [NAS] %(levelname)s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )
    return logging.getLogger("NAS")


# ============================================================================
# SUPERNET TRAINER  (Phase 1)
# ============================================================================

class SupernetTrainer:
    """
    Trains the supernet by sampling a random architecture at every iteration.
    Instead of a true weight-sharing supernet (which requires special
    parameter tables), we train N random architectures in rotation to
    warm-start the search — a practical approximation that is much simpler.
    """

    def __init__(self, search_space: SearchSpace, train_loader, val_loader,
                 device, logger, args):
        self.space = search_space
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = logger
        self.args = args

    def _build_and_train_one(self, arch, epochs, class_weights):
        """Build a model from arch, train for a few epochs, return val AUC."""
        model = NASMultiModalClassifier(arch).to(self.device)

        lr = arch["learning_rate"]
        wd = arch["weight_decay"]
        ls = arch["label_smoothing"]

        if arch["optimizer"] == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                  weight_decay=wd)

        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(self.device),
            label_smoothing=ls,
        )
        scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type == "cuda"))

        for ep in range(1, epochs + 1):
            model.train()
            for features, labels in self.train_loader:
                features = {k: v.to(self.device).float() for k, v in features.items()}
                labels = labels.to(self.device).long()

                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=(self.device.type == "cuda")):
                    logits, _ = model(features)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

        # Quick validation
        auc = self._evaluate(model)
        return auc

    @torch.no_grad()
    def _evaluate(self, model):
        model.eval()
        all_labels, all_probs = [], []
        for features, labels in self.val_loader:
            features = {k: v.to(self.device).float() for k, v in features.items()}
            labels = labels.to(self.device).long()
            logits, _ = model(features)
            probs = F_torch.softmax(logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
        try:
            return roc_auc_score(all_labels, all_probs)
        except ValueError:
            return 0.5

    def warmup(self, n_warmup: int, epochs_per_arch: int, class_weights):
        """Phase-1: warm-start by training n_warmup random architectures."""
        results = []
        for i in range(n_warmup):
            arch = self.space.random_architecture()
            self.logger.info(f"Warmup {i+1}/{n_warmup} — training for {epochs_per_arch} epochs ...")
            auc = self._build_and_train_one(arch, epochs_per_arch, class_weights)
            results.append((auc, arch))
            self.logger.info(f"  AUC = {auc:.4f}")
        return results


# ============================================================================
# EVOLUTIONARY SEARCH  (Phase 2)
# ============================================================================

class EvolutionarySearcher:
    """
    Evolutionary architecture search:
    1.  Initialise population (from warmup or random).
    2.  Evaluate each candidate on the val set via quick training.
    3.  Select top-K, mutate + crossover to form next generation.
    """

    def __init__(self, search_space: SearchSpace, train_loader, val_loader,
                 device, logger, args, class_weights):
        self.space = search_space
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = logger
        self.args = args
        self.class_weights = class_weights

    def _evaluate_arch(self, arch, epochs=3):
        """Train arch from scratch for a few epochs, return val AUC."""
        model = NASMultiModalClassifier(arch).to(self.device)
        lr = arch["learning_rate"]
        wd = arch["weight_decay"]
        ls = arch["label_smoothing"]

        if arch["optimizer"] == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                  weight_decay=wd)

        criterion = nn.CrossEntropyLoss(
            weight=self.class_weights.to(self.device),
            label_smoothing=ls,
        )
        scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type == "cuda"))

        for ep in range(1, epochs + 1):
            model.train()
            for features, labels in self.train_loader:
                features = {k: v.to(self.device).float() for k, v in features.items()}
                labels = labels.to(self.device).long()

                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=(self.device.type == "cuda")):
                    logits, _ = model(features)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

        return self._quick_val(model)

    @torch.no_grad()
    def _quick_val(self, model):
        model.eval()
        all_labels, all_probs = [], []
        for features, labels in self.val_loader:
            features = {k: v.to(self.device).float() for k, v in features.items()}
            labels = labels.to(self.device).long()
            logits, _ = model(features)
            probs = F_torch.softmax(logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
        try:
            return roc_auc_score(all_labels, all_probs)
        except ValueError:
            return 0.5

    def search(self, warmup_results, generations, pop_size, top_k,
               epochs_per_eval=3):
        """Run evolutionary search."""
        # Initialise population from warmup or random
        population = []
        for auc, arch in warmup_results:
            population.append((auc, arch))
        while len(population) < pop_size:
            arch = self.space.random_architecture()
            population.append((0.0, arch))  # will be evaluated below

        best_ever = (0.0, None)

        for gen in range(1, generations + 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Generation {gen}/{generations}  (pop={len(population)})")
            self.logger.info(f"{'='*60}")

            # Evaluate candidates that haven't been evaluated yet
            scored = []
            for i, (existing_auc, arch) in enumerate(population):
                if existing_auc > 0:
                    auc = existing_auc
                    self.logger.info(f"  [{i+1}/{len(population)}] reused AUC = {auc:.4f}")
                else:
                    auc = self._evaluate_arch(arch, epochs=epochs_per_eval)
                    self.logger.info(f"  [{i+1}/{len(population)}] AUC = {auc:.4f}")
                scored.append((auc, arch))

                if auc > best_ever[0]:
                    best_ever = (auc, copy.deepcopy(arch))
                    self.logger.info(f"  ★ New best AUC: {auc:.4f}")

            # Sort by AUC descending, keep top-K
            scored.sort(key=lambda x: x[0], reverse=True)
            parents = scored[:top_k]

            self.logger.info(f"\n  Top-{top_k} AUCs: {[f'{a:.4f}' for a, _ in parents]}")

            # Generate next generation via mutation + crossover
            next_gen = list(parents)  # elitism — keep parents
            while len(next_gen) < pop_size:
                if random.random() < 0.5:
                    # Mutation
                    parent_arch = random.choice(parents)[1]
                    child = self.space.mutate(parent_arch, prob=0.2)
                else:
                    # Crossover + mutation
                    p1 = random.choice(parents)[1]
                    p2 = random.choice(parents)[1]
                    child = self.space.crossover(p1, p2)
                    child = self.space.mutate(child, prob=0.1)
                next_gen.append((0.0, child))  # 0.0 = needs evaluation

            population = next_gen

        return best_ever


# ============================================================================
# DATA SETUP  (mirrors run_training_vit.py)
# ============================================================================

def build_data_loaders(args, logger):
    """Build train/val datasets and loaders, return loaders + class_weights."""
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
    if len(all_vids) == 0:
        logger.error("No videos found! Check BASE_DIR.")
        sys.exit(1)

    train_ids, val_ids = train_test_split(
        all_vids, test_size=0.25,
        stratify=[labels[v] for v in all_vids], random_state=42,
    )

    # Balanced validation set
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

    # Class weights for CE
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
    logger.info(f"Class counts (train): Real={counts[0]}, Fake={counts[1]}")

    return train_loader, val_loader, class_weights


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="NAS for Deepfake Detection")
    parser.add_argument("--dataset", choices=["FR", "FR_10Mar"], default="FR")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)

    # Phase 1 — Supernet warmup
    parser.add_argument("--warmup-archs", type=int, default=10,
                        help="Number of random architectures to pre-evaluate")
    parser.add_argument("--supernet-epochs", type=int, default=5,
                        help="Epochs per architecture during warmup")

    # Phase 2 — Evolutionary search
    parser.add_argument("--search-generations", type=int, default=20,
                        help="Number of evolutionary generations")
    parser.add_argument("--population-size", type=int, default=20,
                        help="Population size per generation")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Top-K parents to keep for next generation")
    parser.add_argument("--eval-epochs", type=int, default=5,
                        help="Training epochs per arch during search evaluation")

    # Output
    parser.add_argument("--output-dir", type=str, default="nas_results",
                        help="Directory to save search results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick smoke test with minimal compute")

    args = parser.parse_args()

    # Dry-run overrides
    if args.dry_run:
        args.warmup_archs = 2
        args.supernet_epochs = 1
        args.search_generations = 2
        args.population_size = 4
        args.top_k = 2
        args.eval_epochs = 1

    logger = setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Search space
    space = SearchSpace()
    logger.info(f"Total search space size: ~{space.total_combinations():.2e} combinations")

    # Data
    train_loader, val_loader, class_weights = build_data_loaders(args, logger)

    # ------------------------------------------------------------------
    # Phase 1: Warmup — evaluate random architectures
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: Supernet Warmup (random architecture evaluation)")
    logger.info("=" * 70)

    supernet_trainer = SupernetTrainer(
        space, train_loader, val_loader, device, logger, args,
    )
    warmup_results = supernet_trainer.warmup(
        n_warmup=args.warmup_archs,
        epochs_per_arch=args.supernet_epochs,
        class_weights=class_weights,
    )

    # Save warmup results
    warmup_log = [{"auc": auc, "arch": arch} for auc, arch in warmup_results]
    with open(os.path.join(args.output_dir, "warmup_results.json"), "w") as f:
        json.dump(warmup_log, f, indent=2)
    logger.info(f"Warmup complete — best AUC: {max(r[0] for r in warmup_results):.4f}")

    # ------------------------------------------------------------------
    # Phase 2: Evolutionary Search
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: Evolutionary Architecture Search")
    logger.info("=" * 70)

    searcher = EvolutionarySearcher(
        space, train_loader, val_loader, device, logger, args, class_weights,
    )
    best_auc, best_arch = searcher.search(
        warmup_results=warmup_results,
        generations=args.search_generations,
        pop_size=args.population_size,
        top_k=args.top_k,
        epochs_per_eval=args.eval_epochs,
    )

    # ------------------------------------------------------------------
    # Save best architecture
    # ------------------------------------------------------------------
    best_arch_path = os.path.join(args.output_dir, "best_architecture.json")
    SearchSpace.save(best_arch, best_arch_path)

    logger.info("\n" + "=" * 70)
    logger.info("NAS SEARCH COMPLETE")
    logger.info(f"Best AUC: {best_auc:.4f}")
    logger.info(f"Best architecture saved to: {best_arch_path}")
    logger.info("=" * 70)
    logger.info("\nBest architecture config:")
    for k, v in best_arch.items():
        logger.info(f"  {k}: {v}")

    logger.info(f"\nTo train the best architecture from scratch, run:")
    logger.info(f"  python run_train_best_arch.py --arch {best_arch_path}")


if __name__ == "__main__":
    main()
