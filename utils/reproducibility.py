"""Experiment reproducibility helpers."""

from __future__ import annotations

import os
import random
from typing import Sequence

import numpy as np
import torch


def set_seed(
    seed: int,
    allowed_seeds: Sequence[int],
    deterministic: bool,
    warn_only: bool,
) -> None:
    """Seed Python, NumPy, PyTorch, CUDA, and deterministic backend settings."""
    seed = int(seed)
    allowed = [int(value) for value in allowed_seeds]
    if seed not in allowed:
        raise ValueError(f"seed={seed} is not one of configured seeds {allowed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=bool(warn_only))


# Design note (paper Sec. 5.1): every run is keyed by one of seeds 0--4 so
# reported mean and standard deviation can be reproduced across all baselines.
