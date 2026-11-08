"""MLP baseline over all 52 variable statistics."""

from __future__ import annotations

from typing import Any, Mapping

from torch import Tensor, nn

from .common import flattened_variable_statistics, make_mlp, normalization


class MLP(nn.Module):
    def __init__(self, num_classes: int, config: Mapping[str, Any]) -> None:
        super().__init__()
        input_dim = int(config["input_dim"])
        self.normalization = normalization(str(config["input_normalization"]), input_dim)
        self.classifier = make_mlp(
            input_dim,
            [int(value) for value in config["hidden_dims"]],
            num_classes,
            float(config["dropout"]),
            str(config["activation"]),
        )

    def forward(self, x_dict: Mapping[str, Any], edge_index_dict: Mapping) -> Tensor:
        del edge_index_dict
        features = flattened_variable_statistics(x_dict)
        return self.classifier(self.normalization(features))


# Design note (paper Sec. 5.3): all 52 variables' seven statistics are flattened
# and classified without graph edges, providing the direct non-graph control.
