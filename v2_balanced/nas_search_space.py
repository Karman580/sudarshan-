"""
Neural Architecture Search — Search Space Definition
=====================================================
Defines all searchable dimensions for the multi-modal deepfake detector.
Each architecture is represented as a flat dict that can be JSON-serialised,
mutated, and crossed-over by the evolutionary controller.
"""

import copy
import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List


# ============================================================================
# SEARCH DIMENSIONS
# ============================================================================

SEARCH_SPACE = {
    # ---- Frequency Stream ----
    "freq_num_blocks":       [2, 3, 4],
    "freq_channels":         [[32, 64, 128], [64, 128, 256], [32, 64, 128, 256]],
    "freq_pool_type":        ["max", "avg"],
    "freq_use_se":           [True, False],
    "freq_hidden_dim":       [256, 384, 512],

    # ---- Temporal Stream ----
    "temp_num_blocks":       [3, 4, 5],
    "temp_first_kernel":     [3, 5, 7],
    "temp_channels":         [[64, 128, 256, 512], [32, 64, 128, 256], [64, 128, 256]],
    "temp_use_se":           [True, False],
    "temp_hidden_dim":       [256, 384, 512],

    # ---- Spatial Stream ----
    "spatial_act_blocks":    [1, 2, 3],
    "spatial_act_channels":  [[32, 64], [64, 128], [32, 64, 128]],
    "spatial_cls_proj_dim":  [256, 384, 512],
    "spatial_merge_mode":    ["concat", "gate", "bilinear"],
    "spatial_hidden_dim":    [256, 384, 512],

    # ---- Fusion Head ----
    "fusion_strategy":       ["concat", "attention", "gated"],
    "fusion_hidden_layers":  [[256], [512], [256, 128], [512, 256]],
    "fusion_use_bn":         [True, False],

    # ---- Shared / Training ----
    "dropout":               [0.2, 0.3, 0.4, 0.5],
    "learning_rate":         [1e-4, 5e-5, 3e-5, 1e-5, 1e-6],
    "weight_decay":          [1e-3, 1e-4, 5e-4],
    "label_smoothing":       [0.0, 0.05, 0.1],
    "optimizer":             ["adamw", "sgd"],
}


# ============================================================================
# SEARCH-SPACE HELPER CLASS
# ============================================================================

class SearchSpace:
    """Utility to sample, validate, mutate, and crossover architectures."""

    def __init__(self, space: Dict[str, List[Any]] | None = None):
        self.space = space or SEARCH_SPACE

    # ------ sampling ------
    def random_architecture(self) -> Dict[str, Any]:
        """Sample a fully random architecture config."""
        return {k: random.choice(v) for k, v in self.space.items()}

    # ------ validation ------
    def validate(self, arch: Dict[str, Any]) -> bool:
        """Check every key exists and the value is from the allowed list."""
        for k, allowed in self.space.items():
            if k not in arch:
                return False
            if arch[k] not in allowed:
                return False
        return True

    # ------ mutation ------
    def mutate(self, arch: Dict[str, Any], prob: float = 0.15) -> Dict[str, Any]:
        """Return a mutated copy — each gene flips with `prob`."""
        child = copy.deepcopy(arch)
        for k, allowed in self.space.items():
            if random.random() < prob:
                child[k] = random.choice(allowed)
        return child

    # ------ crossover ------
    def crossover(self, parent_a: Dict[str, Any],
                  parent_b: Dict[str, Any]) -> Dict[str, Any]:
        """Uniform crossover — pick each gene from a random parent."""
        child = {}
        for k in self.space:
            child[k] = parent_a[k] if random.random() < 0.5 else parent_b[k]
        return child

    # ------ IO ------
    @staticmethod
    def save(arch: Dict[str, Any], path: str):
        with open(path, "w") as f:
            json.dump(arch, f, indent=2)

    @staticmethod
    def load(path: str) -> Dict[str, Any]:
        with open(path) as f:
            return json.load(f)

    def total_combinations(self) -> int:
        n = 1
        for v in self.space.values():
            n *= len(v)
        return n
