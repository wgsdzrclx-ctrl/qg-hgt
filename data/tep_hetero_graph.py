"""Reproducible TEP process graph construction for PyTorch Geometric.

Variable descriptions follow the public Downs--Vogel/Braatz simulator table in
``datasets/tep/source/tennessee-eastman-profBraatz/README.md``. Stream variables
are assigned to the unit that measures or actuates them; this assignment is an
explicit modelling choice, not an official unique node mapping.
"""

from __future__ import annotations

from itertools import permutations
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data, HeteroData


PROCESS_UNITS = (
    "feed_system",
    "reactor",
    "condenser",
    "product_separator",
    "recycle_compressor",
    "stripper",
)
STATISTIC_NAMES = ("mean", "std", "min", "max", "trend", "last", "delta")

# Complete XMEAS -> process-unit mapping. Keys are one-based TEP indices.
XMEAS_TO_UNIT: dict[int, str] = {
    1: "feed_system",          # A feed, stream 1
    2: "feed_system",          # D feed, stream 2
    3: "feed_system",          # E feed, stream 3
    4: "feed_system",          # A+C feed, stream 4
    5: "recycle_compressor",   # recycle flow, stream 8
    6: "reactor",             # combined reactor feed, stream 6
    7: "reactor",             # reactor pressure
    8: "reactor",             # reactor level
    9: "reactor",             # reactor temperature
    10: "product_separator",   # purge rate, stream 9
    11: "product_separator",   # product-separator temperature
    12: "product_separator",   # product-separator level
    13: "product_separator",   # product-separator pressure
    14: "product_separator",   # separator underflow, stream 10
    15: "stripper",            # stripper level
    16: "stripper",            # stripper pressure
    17: "stripper",            # stripper underflow/product, stream 11
    18: "stripper",            # stripper temperature
    19: "stripper",            # stripper steam flow
    20: "recycle_compressor",  # compressor work
    21: "reactor",             # reactor cooling-water outlet temperature
    22: "condenser",           # condenser/separator cooling-water outlet temperature
    23: "reactor",             # stream-6 reactor-feed analysis: component A
    24: "reactor",             # stream-6 reactor-feed analysis: component B
    25: "reactor",             # stream-6 reactor-feed analysis: component C
    26: "reactor",             # stream-6 reactor-feed analysis: component D
    27: "reactor",             # stream-6 reactor-feed analysis: component E
    28: "reactor",             # stream-6 reactor-feed analysis: component F
    29: "product_separator",   # stream-9 purge-gas analysis: component A
    30: "product_separator",   # stream-9 purge-gas analysis: component B
    31: "product_separator",   # stream-9 purge-gas analysis: component C
    32: "product_separator",   # stream-9 purge-gas analysis: component D
    33: "product_separator",   # stream-9 purge-gas analysis: component E
    34: "product_separator",   # stream-9 purge-gas analysis: component F
    35: "product_separator",   # stream-9 purge-gas analysis: component G
    36: "product_separator",   # stream-9 purge-gas analysis: component H
    37: "stripper",            # stream-11 product analysis: component D
    38: "stripper",            # stream-11 product analysis: component E
    39: "stripper",            # stream-11 product analysis: component F
    40: "stripper",            # stream-11 product analysis: component G
    41: "stripper",            # stream-11 product analysis: component H
}

# Complete XMV -> process-unit mapping. The dataset has XMV(1)--XMV(11);
# simulator-only XMV(12), agitator speed, is intentionally not fabricated.
XMV_TO_UNIT: dict[int, str] = {
    1: "feed_system",          # D-feed flow valve, stream 2
    2: "feed_system",          # E-feed flow valve, stream 3
    3: "feed_system",          # A-feed flow valve, stream 1
    4: "feed_system",          # A+C-feed flow valve, stream 4
    5: "recycle_compressor",   # compressor recycle valve
    6: "product_separator",   # purge valve, stream 9
    7: "product_separator",   # separator liquid-flow valve, stream 10
    8: "stripper",            # stripper product-flow valve, stream 11
    9: "stripper",            # stripper steam valve
    10: "reactor",            # reactor cooling-water valve
    11: "condenser",          # condenser cooling-water valve
}

# Directed material/energy-flow abstraction used by the paper graph.
PROCESS_FLOWS: tuple[tuple[str, str], ...] = (
    ("feed_system", "reactor"),
    ("recycle_compressor", "reactor"),
    ("reactor", "condenser"),
    ("condenser", "product_separator"),
    ("product_separator", "recycle_compressor"),
    ("product_separator", "stripper"),
)

EDGE_TYPES: tuple[tuple[str, str, str], ...] = (
    ("sensor", "monitors", "process_unit"),
    ("actuator", "controls", "process_unit"),
    ("process_unit", "flows_to", "process_unit"),
    ("sensor", "co_monitors", "sensor"),
    ("process_unit", "monitored_by", "sensor"),
    ("process_unit", "controlled_by", "actuator"),
)
EDGE_TYPE_TO_ID = {edge_type: index for index, edge_type in enumerate(EDGE_TYPES)}
NODE_TYPE_TO_ID = {name: index for index, name in enumerate(("sensor", "actuator", "process_unit"))}


def validate_tep_mapping() -> None:
    """Fail early if a variable or unit is missing from the declared mapping."""
    if set(XMEAS_TO_UNIT) != set(range(1, 42)):
        raise ValueError("XMEAS_TO_UNIT must contain exactly XMEAS(1)--XMEAS(41).")
    if set(XMV_TO_UNIT) != set(range(1, 12)):
        raise ValueError("XMV_TO_UNIT must contain exactly XMV(1)--XMV(11).")
    known = set(PROCESS_UNITS)
    mapped = set(XMEAS_TO_UNIT.values()) | set(XMV_TO_UNIT.values())
    if not mapped <= known:
        raise ValueError(f"Unknown mapped process units: {sorted(mapped - known)}")
    if any(source not in known or target not in known for source, target in PROCESS_FLOWS):
        raise ValueError("PROCESS_FLOWS contains an unknown process unit.")


def window_statistics(
    variable_first_window: Tensor,
    observed_mask: Tensor | None,
    statistics_config: Mapping[str, Any],
) -> Tensor:
    """Return [mean, std, min, max, slope, last, delta] for every variable."""
    x = torch.as_tensor(variable_first_window)
    if tuple(statistics_config["feature_names"]) != STATISTIC_NAMES:
        raise ValueError(f"statistics.feature_names must be exactly {list(STATISTIC_NAMES)}")
    if x.ndim != 2 or not x.is_floating_point():
        raise ValueError("Expected a floating [variables, time] window tensor.")
    if observed_mask is None:
        mask = torch.isfinite(x)
    else:
        mask = torch.as_tensor(observed_mask, device=x.device, dtype=torch.bool)
        if mask.shape != x.shape:
            raise ValueError("observed_mask must have the same shape as the window.")
        mask = mask & torch.isfinite(x)

    fill = float(statistics_config["missing_fill_value"])
    correction = int(statistics_config["std_correction"])
    safe = torch.where(mask, x, torch.zeros_like(x))
    count = mask.sum(dim=1).to(x.dtype)
    safe_count = count.clamp_min(1.0)
    mean = safe.sum(dim=1) / safe_count
    centered = torch.where(mask, x - mean[:, None], torch.zeros_like(x))
    variance_denominator = (count - correction).clamp_min(1.0)
    variance = centered.square().sum(dim=1) / variance_denominator
    variance = torch.where(count > correction, variance, torch.zeros_like(variance))
    std = variance.clamp_min(0.0).sqrt()

    positive_infinity = torch.full_like(x, float("inf"))
    negative_infinity = torch.full_like(x, float("-inf"))
    minimum = torch.where(mask, x, positive_infinity).min(dim=1).values
    maximum = torch.where(mask, x, negative_infinity).max(dim=1).values
    all_missing = count == 0
    minimum = torch.where(all_missing, torch.full_like(minimum, fill), minimum)
    maximum = torch.where(all_missing, torch.full_like(maximum, fill), maximum)

    time = torch.arange(x.shape[1], dtype=x.dtype, device=x.device)[None, :]
    time_mean = (time * mask).sum(dim=1) / safe_count
    time_centered = torch.where(mask, time - time_mean[:, None], torch.zeros_like(x))
    trend_denominator = time_centered.square().sum(dim=1)
    trend = (time_centered * centered).sum(dim=1) / trend_denominator.clamp_min(1.0)
    trend = torch.where(trend_denominator > 0.0, trend, torch.zeros_like(trend))

    fill_tensor = torch.full((x.shape[0],), fill, dtype=x.dtype, device=x.device)
    first = torch.where(mask[:, 0], x[:, 0], fill_tensor)
    last = torch.where(mask[:, -1], x[:, -1], fill_tensor)
    delta = last - first
    return torch.stack((mean, std, minimum, maximum, trend, last, delta), dim=1)


def _edge_index(pairs: Sequence[tuple[int, int]]) -> Tensor:
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def _process_edges() -> dict[tuple[str, str, str], Tensor]:
    unit_id = {name: index for index, name in enumerate(PROCESS_UNITS)}
    sensor_to_unit = [(index - 1, unit_id[unit]) for index, unit in XMEAS_TO_UNIT.items()]
    actuator_to_unit = [(index - 1, unit_id[unit]) for index, unit in XMV_TO_UNIT.items()]
    flow_edges = [(unit_id[source], unit_id[target]) for source, target in PROCESS_FLOWS]
    co_monitor_edges: list[tuple[int, int]] = []
    for unit in PROCESS_UNITS:
        members = [index - 1 for index, mapped_unit in XMEAS_TO_UNIT.items() if mapped_unit == unit]
        co_monitor_edges.extend(permutations(members, 2))
    return {
        EDGE_TYPES[0]: _edge_index(sensor_to_unit),
        EDGE_TYPES[1]: _edge_index(actuator_to_unit),
        EDGE_TYPES[2]: _edge_index(flow_edges),
        EDGE_TYPES[3]: _edge_index(co_monitor_edges),
        EDGE_TYPES[4]: _edge_index([(target, source) for source, target in sensor_to_unit]),
        EDGE_TYPES[5]: _edge_index([(target, source) for source, target in actuator_to_unit]),
    }


def _shift_window_for_negative(window: Tensor, mask: Tensor, steps: int) -> tuple[Tensor, Tensor]:
    """Create an edge-padded temporal mismatch used only for training negatives."""
    if steps <= 0 or steps >= window.shape[1]:
        raise ValueError("graph.temporal_negative_shift must be in [1, window_length-1].")
    prefix_x = window[:, :1].expand(-1, steps)
    prefix_mask = mask[:, :1].expand(-1, steps)
    return (
        torch.cat((prefix_x, window[:, :-steps]), dim=1),
        torch.cat((prefix_mask, mask[:, :-steps]), dim=1),
    )


def build_tep_hetero_graph(
    sample: Data,
    graph_config: Mapping[str, Any],
    homogeneous: bool | None = None,
) -> HeteroData | Data:
    """Convert one Stage-1 window sample into a TEP HeteroData or Data graph."""
    validate_tep_mapping()
    if sample.x.shape[0] != 52:
        raise ValueError(f"Expected 52 TEP variables, got shape {tuple(sample.x.shape)}")
    statistics = window_statistics(sample.x, sample.observed_mask, graph_config["statistics"])
    sensor_x = statistics[:41]
    actuator_x = statistics[41:]
    unit_features = []
    for unit in PROCESS_UNITS:
        sensor_ids = [index - 1 for index, value in XMEAS_TO_UNIT.items() if value == unit]
        actuator_ids = [index - 1 for index, value in XMV_TO_UNIT.items() if value == unit]
        attached = [sensor_x[sensor_ids], actuator_x[actuator_ids]]
        attached = [values for values in attached if values.numel()]
        unit_features.append(torch.cat(attached, dim=0).mean(dim=0))

    graph = HeteroData()
    graph["sensor"].x = sensor_x
    graph["sensor"].raw_window = sample.x[:41].clone()
    graph["sensor"].observed_mask = sample.observed_mask[:41].clone()
    if hasattr(sample, "timestamp_delay"):
        delay = sample.timestamp_delay.reshape(-1)[:41].to(sensor_x.dtype)
    else:
        delay = torch.zeros(41, dtype=sensor_x.dtype, device=sensor_x.device)
    graph["sensor"].timestamp_delay = delay[:, None]
    graph["sensor"].variable_index = torch.arange(1, 42, dtype=torch.long)
    graph["actuator"].x = actuator_x
    graph["actuator"].raw_window = sample.x[41:].clone()
    graph["actuator"].observed_mask = sample.observed_mask[41:].clone()
    graph["actuator"].variable_index = torch.arange(1, 12, dtype=torch.long)
    graph["process_unit"].x = torch.stack(unit_features)
    graph["process_unit"].unit_index = torch.arange(len(PROCESS_UNITS), dtype=torch.long)
    for edge_type, edge_index in _process_edges().items():
        relation_id = EDGE_TYPE_TO_ID[edge_type]
        graph[edge_type].edge_index = edge_index
        graph[edge_type].relation_id = torch.full(
            (edge_index.shape[1],), relation_id, dtype=torch.long
        )
    graph.y = sample.y.clone()
    graph.window_start = sample.window_start.clone()
    graph.window_end = sample.window_end.clone()
    graph.fault_scenario = sample.fault_scenario.clone()
    graph.edge_type_map = dict(EDGE_TYPE_TO_ID)
    graph.node_type_map = dict(NODE_TYPE_TO_ID)
    graph.process_unit_names = list(PROCESS_UNITS)
    if hasattr(sample, "run_uid"):
        graph.run_uid = sample.run_uid
    if hasattr(sample, "q_star"):
        graph["sensor"].quality_target = sample.q_star[:41, None]
        graph["actuator"].quality_target = sample.q_star[41:, None]
        unit_quality = []
        for unit in PROCESS_UNITS:
            sensor_ids = [index - 1 for index, value in XMEAS_TO_UNIT.items() if value == unit]
            actuator_ids = [index - 1 for index, value in XMV_TO_UNIT.items() if value == unit]
            targets = [sample.q_star[sensor_ids], sample.q_star[41 + torch.tensor(actuator_ids)]]
            targets = [values for values in targets if values.numel()]
            unit_quality.append(torch.cat(targets).mean())
        graph["process_unit"].quality_target = torch.stack(unit_quality)[:, None]
        shift = int(graph_config["temporal_negative_shift"])
        shifted_window, shifted_mask = _shift_window_for_negative(
            sample.x, sample.observed_mask, shift
        )
        shifted_statistics = window_statistics(
            shifted_window, shifted_mask, graph_config["statistics"]
        )
        graph["sensor"].temporal_negative_x = shifted_statistics[:41]
        graph["actuator"].temporal_negative_x = shifted_statistics[41:]
        shifted_units = []
        for unit in PROCESS_UNITS:
            sensor_ids = [index - 1 for index, value in XMEAS_TO_UNIT.items() if value == unit]
            actuator_ids = [index - 1 for index, value in XMV_TO_UNIT.items() if value == unit]
            attached = [
                shifted_statistics[:41][sensor_ids],
                shifted_statistics[41:][actuator_ids],
            ]
            attached = [values for values in attached if values.numel()]
            shifted_units.append(torch.cat(attached, dim=0).mean(dim=0))
        graph["process_unit"].temporal_negative_x = torch.stack(shifted_units)

    use_homogeneous = bool(graph_config["homogeneous_ablation"]) if homogeneous is None else homogeneous
    if not use_homogeneous:
        return graph
    node_attributes = ["x"]
    if hasattr(graph["sensor"], "quality_target"):
        node_attributes.append("quality_target")
    result = graph.to_homogeneous(node_attrs=node_attributes, edge_attrs=["relation_id"])
    result.y = graph.y
    result.edge_type_map = graph.edge_type_map
    result.node_type_map = graph.node_type_map
    if hasattr(graph, "run_uid"):
        result.run_uid = graph.run_uid
    result.window_start = graph.window_start
    result.window_end = graph.window_end
    result.fault_scenario = graph.fault_scenario
    return result


class TEPGraphDataset(Dataset[HeteroData | Data]):
    """Lazy graph view over a Stage-1 TEPWindowDataset."""

    def __init__(
        self,
        window_dataset: Dataset[Data],
        graph_config: Mapping[str, Any],
        homogeneous: bool | None = None,
    ) -> None:
        self.window_dataset = window_dataset
        self.graph_config = graph_config
        self.homogeneous = homogeneous

    def __len__(self) -> int:
        return len(self.window_dataset)

    def __getitem__(self, index: int) -> HeteroData | Data:
        return build_tep_hetero_graph(
            self.window_dataset[index], self.graph_config, self.homogeneous
        )

    def set_epoch(self, epoch: int) -> None:
        target = self.window_dataset
        if hasattr(target, "dataset"):
            target = target.dataset
        if hasattr(target, "set_epoch"):
            target.set_epoch(epoch)


# Design note (paper Sec. 4.2-4.6): variables become sensor/actuator nodes with
# seven statistics; unit nodes aggregate attached variables. Raw observable
# windows support quality features, while shifted copies are training negatives.
