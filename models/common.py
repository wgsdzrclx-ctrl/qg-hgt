"""Shared input adapters and pooling helpers for uniformly-called baselines."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import global_mean_pool

from data.tep_hetero_graph import EDGE_TYPES, PROCESS_UNITS, XMEAS_TO_UNIT, XMV_TO_UNIT


NODE_TYPES = ("sensor", "actuator", "process_unit")
NODE_COUNTS = {
    "sensor": len(XMEAS_TO_UNIT),
    "actuator": len(XMV_TO_UNIT),
    "process_unit": len(PROCESS_UNITS),
}


def activation(name: str) -> torch.nn.Module:
    choices: dict[str, torch.nn.Module] = {
        "relu": torch.nn.ReLU(),
        "gelu": torch.nn.GELU(),
        "elu": torch.nn.ELU(),
    }
    if name not in choices:
        raise ValueError(f"Unsupported activation: {name}")
    return choices[name]


def normalization(name: str, dimension: int) -> torch.nn.Module:
    choices: dict[str, torch.nn.Module] = {
        "layer_norm": torch.nn.LayerNorm(dimension),
        "none": torch.nn.Identity(),
    }
    if name not in choices:
        raise ValueError(f"Unsupported normalization: {name}")
    return choices[name]


def make_mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    dropout: float,
    activation_name: str,
) -> torch.nn.Sequential:
    layers: list[torch.nn.Module] = []
    previous = input_dim
    for hidden in hidden_dims:
        layers.extend(
            (
                torch.nn.Linear(previous, int(hidden)),
                activation(activation_name),
                torch.nn.Dropout(dropout),
            )
        )
        previous = int(hidden)
    layers.append(torch.nn.Linear(previous, output_dim))
    return torch.nn.Sequential(*layers)


def batch_size_from_x_dict(x_dict: Mapping[str, Any]) -> int:
    sensor_count = x_dict["sensor"].shape[0]
    if sensor_count % NODE_COUNTS["sensor"]:
        raise ValueError("Sensor node count is not divisible by 41.")
    return sensor_count // NODE_COUNTS["sensor"]


def batch_dict_from_x_dict(x_dict: Mapping[str, Any]) -> dict[str, Tensor]:
    provided = x_dict.get("__batch_dict__")
    if provided is not None:
        return provided
    batch_size = batch_size_from_x_dict(x_dict)
    return {
        node_type: torch.arange(batch_size, device=x_dict[node_type].device).repeat_interleave(count)
        for node_type, count in NODE_COUNTS.items()
    }


def typed_mean_pool(h_dict: Mapping[str, Tensor], x_dict: Mapping[str, Any]) -> Tensor:
    batch_dict = batch_dict_from_x_dict(x_dict)
    return torch.cat(
        [
            global_mean_pool(h_dict[node_type], batch_dict[node_type])
            for node_type in ("sensor", "process_unit", "actuator")
        ],
        dim=-1,
    )


def flattened_variable_statistics(x_dict: Mapping[str, Any]) -> Tensor:
    batch_size = batch_size_from_x_dict(x_dict)
    sensor = x_dict["sensor"].reshape(batch_size, NODE_COUNTS["sensor"], -1)
    actuator = x_dict["actuator"].reshape(batch_size, NODE_COUNTS["actuator"], -1)
    return torch.cat((sensor, actuator), dim=1).flatten(start_dim=1)


def raw_process_window(x_dict: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
    if "__raw_window__" not in x_dict or "__observed_mask__" not in x_dict:
        raise KeyError("Temporal baselines require __raw_window__ and __observed_mask__.")
    return x_dict["__raw_window__"], x_dict["__observed_mask__"]


def masked_window_standardize(window: Tensor, mask: Tensor, epsilon: float) -> Tensor:
    valid = mask.bool() & torch.isfinite(window)
    count = valid.sum(dim=-1, keepdim=True).to(window.dtype).clamp_min(1.0)
    mean = torch.where(valid, window, torch.zeros_like(window)).sum(dim=-1, keepdim=True) / count
    variance = (
        torch.where(valid, window - mean, torch.zeros_like(window)).square().sum(dim=-1, keepdim=True)
        / count
    )
    normalized = (window - mean) / variance.sqrt().clamp_min(float(epsilon))
    return torch.where(valid, normalized, torch.zeros_like(normalized))


def homogeneous_graph(
    x_dict: Mapping[str, Any],
    edge_index_dict: Mapping[tuple[str, str, str], Tensor],
) -> tuple[Tensor, Tensor, Tensor, dict[str, slice], Tensor]:
    node_x = [x_dict[node_type] for node_type in NODE_TYPES]
    offsets: dict[str, int] = {}
    slices: dict[str, slice] = {}
    total = 0
    for node_type, values in zip(NODE_TYPES, node_x):
        offsets[node_type] = total
        slices[node_type] = slice(total, total + values.shape[0])
        total += values.shape[0]
    edges: list[Tensor] = []
    relation_ids: list[Tensor] = []
    for relation_id, edge_type in enumerate(EDGE_TYPES):
        edge_index = edge_index_dict[edge_type].clone()
        edge_index[0] += offsets[edge_type[0]]
        edge_index[1] += offsets[edge_type[2]]
        edges.append(edge_index)
        relation_ids.append(
            torch.full(
                (edge_index.shape[1],), relation_id, dtype=torch.long, device=edge_index.device
            )
        )
    batch_dict = batch_dict_from_x_dict(x_dict)
    homogeneous_batch = torch.cat([batch_dict[node_type] for node_type in NODE_TYPES])
    return (
        torch.cat(node_x, dim=0),
        torch.cat(edges, dim=1),
        torch.cat(relation_ids),
        slices,
        homogeneous_batch,
    )


def pool_homogeneous_by_type(
    node_embeddings: Tensor,
    node_slices: Mapping[str, slice],
    x_dict: Mapping[str, Any],
) -> Tensor:
    """Mean-pool each original node type, then concatenate the three pools."""
    batches = batch_dict_from_x_dict(x_dict)
    split = {
        node_type: node_embeddings[node_slices[node_type]] for node_type in NODE_TYPES
    }
    return typed_mean_pool(split, batches)


def extended_x_dict(data: HeteroData) -> dict[str, Any]:
    """Convert a PyG batch to the common forward(x_dict, edge_index_dict) contract."""
    result: dict[str, Any] = {node_type: data[node_type].x for node_type in NODE_TYPES}
    result["__batch_dict__"] = {
        node_type: data[node_type].batch
        if hasattr(data[node_type], "batch")
        else torch.zeros(data[node_type].x.shape[0], dtype=torch.long, device=data[node_type].x.device)
        for node_type in NODE_TYPES
    }
    batch_size = batch_size_from_x_dict(result)
    sensor_raw = data["sensor"].raw_window.reshape(batch_size, NODE_COUNTS["sensor"], -1)
    actuator_raw = data["actuator"].raw_window.reshape(batch_size, NODE_COUNTS["actuator"], -1)
    sensor_mask = data["sensor"].observed_mask.reshape(batch_size, NODE_COUNTS["sensor"], -1)
    actuator_mask = data["actuator"].observed_mask.reshape(batch_size, NODE_COUNTS["actuator"], -1)
    result["__raw_window__"] = torch.cat((sensor_raw, actuator_raw), dim=1)
    result["__observed_mask__"] = torch.cat((sensor_mask, actuator_mask), dim=1)
    result["__sensor_raw_window__"] = data["sensor"].raw_window
    result["__sensor_observed_mask__"] = data["sensor"].observed_mask
    result["__sensor_timestamp_delay__"] = data["sensor"].timestamp_delay
    if hasattr(data["sensor"], "quality_target"):
        result["__quality_target_dict__"] = {
            node_type: data[node_type].quality_target for node_type in NODE_TYPES
        }
    if all(hasattr(data[node_type], "temporal_negative_x") for node_type in NODE_TYPES):
        result["__temporal_negative_x_dict__"] = {
            node_type: data[node_type].temporal_negative_x for node_type in NODE_TYPES
        }
    if hasattr(data, "y"):
        result["__y__"] = data.y
    return result


def heterodata_from_inputs(
    x_dict: Mapping[str, Any],
    edge_index_dict: Mapping[tuple[str, str, str], Tensor],
) -> HeteroData:
    data = HeteroData()
    for node_type in NODE_TYPES:
        data[node_type].x = x_dict[node_type]
        batch = batch_dict_from_x_dict(x_dict)[node_type]
        data[node_type].batch = batch
        batch_size = int(batch.max().item()) + 1 if batch.numel() else 1
        counts = torch.bincount(batch, minlength=batch_size)
        data[node_type].ptr = torch.cat(
            (counts.new_zeros(1), counts.cumsum(dim=0)), dim=0
        )
    data["sensor"].raw_window = x_dict["__sensor_raw_window__"]
    data["sensor"].observed_mask = x_dict["__sensor_observed_mask__"]
    data["sensor"].timestamp_delay = x_dict["__sensor_timestamp_delay__"]
    for edge_type, edge_index in edge_index_dict.items():
        data[edge_type].edge_index = edge_index
    if "__quality_target_dict__" in x_dict:
        for node_type, target in x_dict["__quality_target_dict__"].items():
            data[node_type].quality_target = target
    if "__temporal_negative_x_dict__" in x_dict:
        for node_type, values in x_dict["__temporal_negative_x_dict__"].items():
            data[node_type].temporal_negative_x = values
    if "__y__" in x_dict:
        data.y = x_dict["__y__"]
    return data


# Design note (paper Sec. 5.3): all baselines receive one common graph batch;
# graph models use identical seven-dimensional node statistics, while temporal
# models read the same underlying W-step windows without extra labels or data.
