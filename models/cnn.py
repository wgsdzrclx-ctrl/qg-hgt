"""One-dimensional CNN temporal baseline with variables as input channels."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .common import activation, masked_window_standardize, raw_process_window


class CNN1D(nn.Module):
    def __init__(self, num_classes: int, config: Mapping[str, Any]) -> None:
        super().__init__()
        channels = [int(value) for value in config["channels"]]
        kernels = [int(value) for value in config["kernel_sizes"]]
        if len(channels) != len(kernels):
            raise ValueError("cnn.channels and kernel_sizes must have equal length.")
        layers: list[nn.Module] = []
        previous = int(config["input_channels"])
        for output, kernel in zip(channels, kernels):
            layers.extend(
                (
                    nn.Conv1d(previous, output, kernel, padding=kernel // 2),
                    nn.BatchNorm1d(output),
                    activation(str(config["activation"])),
                    nn.Dropout(float(config["dropout"])),
                )
            )
            previous = output
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(previous, num_classes)
        self.epsilon = float(config["normalization_epsilon"])

    def forward(self, x_dict: Mapping[str, Any], edge_index_dict: Mapping) -> Tensor:
        del edge_index_dict
        window, mask = raw_process_window(x_dict)
        encoded = self.encoder(masked_window_standardize(window, mask, self.epsilon))
        return self.classifier(self.pool(encoded).squeeze(-1))


# Design note (paper Sec. 5.3): XMEAS/XMV are the 52 Conv1d channels and the
# W samples remain the temporal axis; only within-window observable scaling is
# used, so the source windows and split protocol match every other baseline.
