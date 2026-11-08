"""Native heterogeneous relational-GCN baseline."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn

from data.tep_hetero_graph import EDGE_TYPES
from .common import NODE_TYPES, activation, make_mlp, normalization, typed_mean_pool


def _relation_key(edge_type: tuple[str, str, str]) -> str:
    return "__".join(edge_type)


class RelationalLayer(nn.Module):
    """R-GCN update with one transform per declared TEP relation."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.self_linears = nn.ModuleDict(
            {node_type: nn.Linear(input_dim, output_dim) for node_type in NODE_TYPES}
        )
        self.relation_linears = nn.ModuleDict(
            {_relation_key(edge_type): nn.Linear(input_dim, output_dim, bias=False)
             for edge_type in EDGE_TYPES}
        )

    def forward(self, h_dict: Mapping[str, Tensor], edge_index_dict: Mapping[tuple[str, str, str], Tensor]) -> dict[str, Tensor]:
        result = {node_type: self.self_linears[node_type](h_dict[node_type]) for node_type in NODE_TYPES}
        for edge_type in EDGE_TYPES:
            if edge_type not in edge_index_dict:
                continue
            source_type, _, target_type = edge_type
            source, target = edge_index_dict[edge_type]
            message = self.relation_linears[_relation_key(edge_type)](h_dict[source_type])[source]
            aggregate = message.new_zeros((h_dict[target_type].shape[0], message.shape[-1]))
            aggregate.index_add_(0, target, message)
            degree = torch.bincount(target, minlength=aggregate.shape[0]).to(message.dtype).clamp_min(1.0)
            result[target_type] = result[target_type] + aggregate / degree[:, None]
        return result


class RGCN(nn.Module):
    def __init__(self, num_classes: int, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden = int(config["hidden_dim"])
        layers = int(config["num_layers"])
        if layers < 1:
            raise ValueError("RGCN num_layers must be positive.")
        self.layers = nn.ModuleList(
            [RelationalLayer(int(config["input_dim"]), hidden)]
            + [RelationalLayer(hidden, hidden) for _ in range(layers - 1)]
        )
        self.nonlinearities = nn.ModuleList([activation(str(config["activation"])) for _ in range(layers)])
        self.dropout = nn.Dropout(float(config["dropout"]))
        self.input_normalization = nn.ModuleDict(
            {node_type: normalization(str(config["input_normalization"]), int(config["input_dim"]))
             for node_type in NODE_TYPES}
        )
        self.classifier = make_mlp(
            hidden * 3, [int(v) for v in config["classifier_hidden_dims"]], num_classes,
            float(config["dropout"]), str(config["activation"])
        )

    def forward(self, x_dict: Mapping[str, Any], edge_index_dict: Mapping[tuple[str, str, str], Tensor]) -> Tensor:
        h_dict = {node_type: self.input_normalization[node_type](x_dict[node_type]) for node_type in NODE_TYPES}
        for layer, nonlinearity in zip(self.layers, self.nonlinearities):
            h_dict = {key: self.dropout(nonlinearity(value)) for key, value in layer(h_dict, edge_index_dict).items()}
        return self.classifier(typed_mean_pool(h_dict, x_dict))


# Design note (paper Sec. 5.3): each of the six TEP relations owns W_r and is
# mean-normalized before typed pooling; unlike GCN/SAGE/GAT, relation identity
# is retained, but no quality or consistency signal is used.
