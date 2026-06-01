"""
Cross-dataset evaluation for v2_balanced ViT model.

Example:
python evaluate_cross_dataset_vit.py \
  --checkpoint checkpoints/best_vit_model_full.pth \
  --datasets FR FR_10Mar
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_vit import Config, MultiModalFusionClassifierViT
from run_training_vit import CSPMultiModalDatasetViT, collect_video_ids


def resolve_dataset_paths(dataset_name: str):
    if dataset_name == "FR":
        base = "/workspace/FR/ProcessedDataset"
        temporal = "/workspace/FR/ProcessedDataset/temporal_features_18feb"
        spatial = "/workspace/FR/ProcessedDataset/spatial_features_with_masks_updated"
    elif dataset_name == "FR_10Mar":
        base = "/workspace/FR_10Mar/ProcessedDataset"
        temporal = "/workspace/FR_10Mar/ProcessedDataset/temporal_features_18feb"
        spatial = "/workspace/FR_10Mar/ProcessedDataset/spatial_features_with_masks_updated"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return base, temporal, spatial


@torch.no_grad()
def evaluate_single_dataset(model, device, dataset_name, batch_size, max_frames):
    base_dir, temporal_dir, spatial_dir = resolve_dataset_paths(dataset_name)
    data_dict = collect_video_ids(base_dir)
    if not data_dict:
        raise RuntimeError(f"No videos found in {base_dir}")

    labels = {vid: d["label"] for vid, d in data_dict.items()}
    video_ids = list(data_dict.keys())

    dataset = CSPMultiModalDatasetViT(
        video_ids=video_ids,
        data_dict=data_dict,
        labels=labels,
        temporal_dir=temporal_dir,
        spatial_dir=spatial_dir,
        max_frames=max_frames,
        freq_aggregate="mean",
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
    )

    all_preds, all_labels, all_probs = [], [], []
    for features, y in tqdm(loader, desc=f"Eval {dataset_name}"):
        features = {k: v.to(device).float() for k, v in features.items()}
        y = y.to(device).long()

        logits, _ = model(features)
        probs = F.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        all_labels.extend(y.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    real_acc = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fake_acc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    bal_acc = (real_acc + fake_acc) / 2.0
    raw_acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    return {
        "dataset": dataset_name,
        "num_videos": len(video_ids),
        "real_count": int(sum(1 for x in all_labels if x == 0)),
        "fake_count": int(sum(1 for x in all_labels if x == 1)),
        "raw_acc": raw_acc,
        "bal_acc": bal_acc,
        "real_acc": real_acc,
        "fake_acc": fake_acc,
        "auc": auc,
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset evaluation for ViT model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "Path to checkpoint (.pth): e.g. best_auc_model_full.pth "
            "or best_acc_model_full.pth"
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["FR", "FR_10Mar"],
        choices=["FR", "FR_10Mar"],
        help="Datasets to evaluate on",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=32)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("cross_eval")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()
    config.MAX_FRAMES = args.max_frames
    model = MultiModalFusionClassifierViT(config).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    logger.info(f"Loaded checkpoint: {args.checkpoint}")
    logger.info(f"Device: {device}")

    for ds in args.datasets:
        metrics = evaluate_single_dataset(
            model=model,
            device=device,
            dataset_name=ds,
            batch_size=args.batch_size,
            max_frames=args.max_frames,
        )
        logger.info(
            f"[{metrics['dataset']}] N={metrics['num_videos']} "
            f"(Real={metrics['real_count']}, Fake={metrics['fake_count']}) | "
            f"RawAcc={metrics['raw_acc']*100:.2f}% BalAcc={metrics['bal_acc']*100:.2f}% "
            f"RealAcc={metrics['real_acc']*100:.2f}% FakeAcc={metrics['fake_acc']*100:.2f}% "
            f"AUC={metrics['auc']:.4f}"
        )


if __name__ == "__main__":
    main()
