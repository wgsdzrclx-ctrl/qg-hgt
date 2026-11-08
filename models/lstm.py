"""Single multivariate LSTM baseline over each complete process window."""

from __future__ import annotations

from typing import Any, Mapping

from torch import Tensor, nn

from .common import masked_window_standardize, raw_process_window


class LSTM(nn.Module):
    def __init__(self, num_classes: int, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=int(config["input_size"]),
            hidden_size=int(config["hidden_dim"]),
            num_layers=int(config["num_layers"]),
            dropout=float(config["dropout"]) if int(config["num_layers"]) > 1 else 0.0,
            batch_first=True,
            bidirectional=bool(config["bidirectional"]),
        )
        directions = 2 if bool(config["bidirectional"]) else 1
        self.classifier = nn.Linear(int(config["hidden_dim"]) * directions, num_classes)
        self.epsilon = float(config["normalization_epsilon"])

    def forward(self, x_dict: Mapping[str, Any], edge_index_dict: Mapping) -> Tensor:
        del edge_index_dict
        window, mask = raw_process_window(x_dict)
        sequence = masked_window_standardize(window, mask, self.epsilon).transpose(1, 2)
        _, (hidden, _) = self.encoder(sequence)
        directions = 2 if self.encoder.bidirectional else 1
        final = hidden[-directions:].transpose(0, 1).reshape(sequence.shape[0], -1)
        return self.classifier(final)


# Design note (paper Sec. 5.3): one LSTM jointly encodes all 52 variables at
# each time step, preserving cross-variable dynamics with fewer parameters than
# 52 independent LSTMs and using exactly the same W-step source window.
