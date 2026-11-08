"""Native PyG heterogeneous graph-transformer baseline."""

from __future__ import annotations

from typing import Any, Mapping

from torch import Tensor, nn
from torch_geometric.nn import HGTConv

from data.tep_hetero_graph import EDGE_TYPES
from .common import NODE_TYPES, activation, make_mlp, normalization, typed_mean_pool


class HGT(nn.Module):
    def __init__(self, num_classes: int, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden = int(config["hidden_dim"])
        heads = int(config["heads"])
        if hidden % heads:
            raise ValueError("HGT hidden_dim must be divisible by heads.")
        layers = int(config["num_layers"])
        if layers < 1:
            raise ValueError("HGT num_layers must be positive.")
        self.input_projection = nn.ModuleDict(
            {node_type: nn.Linear(int(config["input_dim"]), hidden) for node_type in NODE_TYPES}
        )
        self.input_normalization = nn.ModuleDict(
            {node_type: normalization(str(config["input_normalization"]), int(config["input_dim"]))
             for node_type in NODE_TYPES}
        )
        metadata = (list(NODE_TYPES), list(EDGE_TYPES))
        self.layers = nn.ModuleList(
            [HGTConv(hidden, hidden, metadata, heads=heads)
             for _ in range(layers)]
        )
        self.nonlinearities = nn.ModuleList([activation(str(config["activation"])) for _ in range(layers)])
        self.dropout = nn.Dropout(float(config["dropout"]))
        self.classifier = make_mlp(
            hidden * 3, [int(v) for v in config["classifier_hidden_dims"]], num_classes,
            float(config["dropout"]), str(config["activation"])
        )

    def forward(self, x_dict: Mapping[str, Any], edge_index_dict: Mapping[tuple[str, str, str], Tensor]) -> Tensor:
        h_dict = {
            node_type: self.input_projection[node_type](self.input_normalization[node_type](x_dict[node_type]))
            for node_type in NODE_TYPES
        }
        for layer, nonlinearity in zip(self.layers, self.nonlinearities):
            updated = layer(h_dict, edge_index_dict)
            h_dict = {node_type: self.dropout(nonlinearity(updated[node_type])) for node_type in NODE_TYPES}
        return self.classifier(typed_mean_pool(h_dict, x_dict))


# Design note (paper Sec. 5.3): this is stock PyG HGT relation/type attention
# over the native graph, serving as the direct control before Eq. 4.7 injects
# observable quality and learned evidence consistency.
