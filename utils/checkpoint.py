"""Recoverable PyTorch checkpoint IO."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Mapping[str, float],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": int(epoch),
            "metrics": dict(metrics),
        },
        destination,
    )
    return destination


def load_checkpoint(path: str | Path, model: nn.Module, device: torch.device) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    model.load_state_dict(payload["model_state"])
    return payload


# Design note: checkpoint selection is driven by validation Macro-F1 and freezes
# the exact parameters subsequently evaluated once on the held-out test runs.
