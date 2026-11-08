"""Memory-bounded model evaluation over PyG graph batches."""

from __future__ import annotations

import random
from bisect import bisect_left
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from models.common import extended_x_dict
from .metrics import StreamingClassificationMetrics, StreamingProcessMonitor


def forward_graph_batch(
    model: nn.Module, batch: HeteroData
) -> tuple[Tensor, Tensor | None, Mapping[str, Any] | None]:
    output = model(extended_x_dict(batch), batch.edge_index_dict)
    if isinstance(output, Mapping):
        return output["logits"], output.get("loss"), output
    return output, None, None


class _Reservoir:
    def __init__(self, limit: int, seed: int, total: int) -> None:
        self.limit = int(limit)
        self.rng = random.Random(int(seed))
        count = min(self.limit, int(total))
        self.selected = sorted(self.rng.sample(range(int(total)), count))
        self.rows: list[dict[str, Any]] = []

    def positions(self, start: int, end: int) -> list[int]:
        left = bisect_left(self.selected, start)
        right = bisect_left(self.selected, end)
        return self.selected[left:right]

    def add(self, row: dict[str, Any]) -> None:
        self.rows.append(row)

    def arrays(self) -> dict[str, np.ndarray]:
        if not self.rows:
            return {}
        keys = self.rows[0]
        return {key: np.asarray([row[key] for row in self.rows]) for key in keys}


class _FaultTrace:
    """Keep one complete run that transitions from normal to a fault."""

    def __init__(self, normal_class: int) -> None:
        self.normal_class = int(normal_class)
        self.current_run: str | None = None
        self.current_rows: list[dict[str, Any]] = []
        self.selected: list[dict[str, Any]] = []

    def _commit(self) -> None:
        labels = {int(row["label"]) for row in self.current_rows}
        if self.normal_class in labels and any(value != self.normal_class for value in labels):
            self.selected = list(self.current_rows)

    def add(self, row: dict[str, Any]) -> None:
        run = str(row["run_uid"])
        if self.current_run is not None and run != self.current_run and not self.selected:
            self._commit()
        if run != self.current_run:
            self.current_run = run
            self.current_rows = []
        if not self.selected:
            self.current_rows.append(row)

    def arrays(self) -> dict[str, np.ndarray]:
        if not self.selected:
            self._commit()
        if not self.selected:
            return {}
        keys = self.selected[0]
        return {f"trace_{key}": np.asarray([row[key] for row in self.selected]) for key in keys}


def _batch_run_ids(batch: HeteroData, size: int) -> list[str]:
    value = getattr(batch, "run_uid", None)
    if value is None:
        return [f"unknown_{index}" for index in range(size)]
    if isinstance(value, str):
        return [value] * size
    return [str(item) for item in value]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    criterion: nn.Module | None = None,
    *,
    metric_config: Mapping[str, Any] | None = None,
    artifact_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate once, streaming full metrics and optionally sampling artifacts."""
    model.eval()
    if metric_config is None:
        raise ValueError("metric_config is required; define metric hyperparameters in YAML.")
    accumulator = StreamingClassificationMetrics(
        int(num_classes), int(metric_config["ece_bins"]), float(metric_config["probability_epsilon"])
    )
    process = StreamingProcessMonitor(int(metric_config["normal_class"]))
    collector = None
    trace = None
    if artifact_config and bool(artifact_config["enabled"]):
        collector = _Reservoir(
            int(artifact_config["max_samples"]), int(artifact_config["seed"]), len(loader.dataset)
        )
        trace = _FaultTrace(int(metric_config["normal_class"]))
    total_loss = 0.0
    total_examples = 0
    reliability_sum = 0.0
    reliability_count = 0
    artifact_offset = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        target = batch.y.reshape(-1).long()
        logits, _, model_output = forward_graph_batch(model, batch)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        if criterion is not None:
            total_loss += float(criterion(logits, target).item()) * target.shape[0]
        total_examples += target.shape[0]
        truth_np = target.cpu().numpy()
        probability_np = probabilities.cpu().numpy()
        accumulator.update(truth_np, probability_np)
        run_ids = _batch_run_ids(batch, target.shape[0])
        if hasattr(batch, "window_end"):
            window_times = batch.window_end.reshape(-1).cpu().tolist()
        else:
            window_times = list(range(total_examples - target.shape[0], total_examples))
        process.update(truth_np, predictions.cpu().tolist(), run_ids, window_times)

        representation = logits
        reliability = None
        if model_output is not None:
            representation = model_output.get("graph_embedding", logits)
            reliability = model_output.get("reliability")
        if reliability is not None:
            reliability = reliability.reshape(target.shape[0], -1)
            reliability_sum += float(reliability.sum().item())
            reliability_count += reliability.numel()
        if collector is not None:
            representation_np = representation.detach().cpu().numpy()
            reliability_np = (
                reliability.detach().cpu().numpy().mean(axis=1)
                if reliability is not None
                else np.full(target.shape[0], np.nan)
            )
            batch_end = artifact_offset + target.shape[0]
            selected_local = {
                position - artifact_offset for position in collector.positions(artifact_offset, batch_end)
            }
            trace_local = range(target.shape[0]) if not trace.selected else ()
            candidate_local = sorted(selected_local.union(trace_local))
            for index in candidate_local:
                row = {
                        "representation": representation_np[index],
                        "probabilities": probability_np[index],
                        "label": int(truth_np[index]),
                        "prediction": int(predictions[index].item()),
                        "run_uid": run_ids[index],
                        "window_end": int(window_times[index]),
                        "mean_reliability": float(reliability_np[index]),
                    }
                if index in selected_local:
                    collector.add(row)
                if not trace.selected:
                    trace.add(row)
            artifact_offset = batch_end
    if total_examples == 0:
        raise ValueError("Cannot evaluate an empty DataLoader.")
    result = accumulator.compute()
    result.update(process.compute(bool(metric_config["store_per_run_delays"])))
    if criterion is not None:
        result["loss"] = total_loss / total_examples
    result["mean_sensor_reliability"] = (
        reliability_sum / reliability_count if reliability_count else None
    )
    if collector is not None:
        result["_artifacts"] = {**collector.arrays(), **trace.arrays()}
    return result


# Design note (paper Sec. 5.4): evaluation reads logits and observable model
# outputs only; streaming avoids materializing millions of test probabilities.
