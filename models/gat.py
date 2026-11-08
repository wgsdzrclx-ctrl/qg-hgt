"""Homogeneous graph-attention baseline."""

from __future__ import annotations

from typing import Any, Mapping

from torch import Tensor, nn
from torch_geometric.nn import GATConv

from .common import activation, homogeneous_graph, make_mlp, normalization, pool_homogeneous_by_type


class GAT(nn.Module):
    def __init__(self, num_classes: int, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden = int(config["hidden_dim"])
        layers = int(config["num_layers"])
        if layers < 1:
            raise ValueError("GAT num_layers must be positive.")
        heads = int(config["heads"])
        common = {"heads": heads, "concat": False, "dropout": float(config["attention_dropout"])}
        self.convolutions = nn.ModuleList(
            [GATConv(int(config["input_dim"]), hidden, **common)]
            + [GATConv(hidden, hidden, **common) for _ in range(layers - 1)]
        )
        self.activations = nn.ModuleList([activation(str(config["activation"])) for _ in range(layers)])
        self.dropout = nn.Dropout(float(config["dropout"]))
        self.input_normalization = normalization(str(config["input_normalization"]), int(config["input_dim"]))
        self.classifier = make_mlp(
            hidden * 3, [int(v) for v in config["classifier_hidden_dims"]], num_classes,
            float(config["dropout"]), str(config["activation"])
        )

    def forward(self, x_dict: Mapping[str, Any], edge_index_dict: Mapping[tuple[str, str, str], Tensor]) -> Tensor:
        x, edge_index, _, slices, _ = homogeneous_graph(x_dict, edge_index_dict)
        x = self.input_normalization(x)
        for convolution, nonlinearity in zip(self.convolutions, self.activations):
            x = self.dropout(nonlinearity(convolution(x, edge_index)))
        return self.classifier(pool_homogeneous_by_type(x, slices, x_dict))


# Design note (paper Sec. 5.3): GAT learns edge importance after relation types
# are removed, providing the homogeneous-attention control for QC-HGT.
