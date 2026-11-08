"""Temporal convolutional-network baseline for TEP windows."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .common import activation, masked_window_standardize, raw_process_window


class CausalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int,
                 dropout: float, activation_name: str) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.padding = padding
        self.convolution = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, padding=padding
        )
        self.activation = activation(activation_name)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.residual(x)
        values = self.convolution(x)
        if self.padding:
            values = values[..., :-self.padding]
        return self.dropout(self.activation(values + residual))


class TCN(nn.Module):
    """Causal dilated TCN using the 52 process variables as input channels."""

    def __init__(self, num_classes: int, config: Mapping[str, Any]) -> None:
        super().__init__()
        channels = [int(value) for value in config["channels"]]
        dilations = [int(value) for value in config["dilations"]]
        if len(channels) != len(dilations) or not channels:
            raise ValueError("TCN channels and dilations must be non-empty and equally long.")
        blocks: list[nn.Module] = []
        previous = int(config["input_channels"])
        for width, dilation in zip(channels, dilations):
            blocks.append(
                CausalBlock(previous, width, int(config["kernel_size"]), dilation,
                            float(config["dropout"]), str(config["activation"]))
            )
            previous = width
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Linear(previous, num_classes)
        self.epsilon = float(config["normalization_epsilon"])

    def forward(
        self,
        x_dict: Mapping[str, Any],
        edge_index_dict: Mapping[tuple[str, str, str], Tensor],
    ) -> Tensor:
        del edge_index_dict
        window, mask = raw_process_window(x_dict)
        values = masked_window_standardize(window, mask, self.epsilon)
        return self.classifier(self.blocks(values)[..., -1])


# Design note (paper Sec. 5.3): causal dilated convolutions model multi-scale
# temporal dependence from the same W-step input; no future time point or
# degradation parameter is exposed to the classifier.
