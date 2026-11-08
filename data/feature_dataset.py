"""Feature views and deterministic sampling for non-graph baselines."""

from __future__ import annotations

import random
from bisect import bisect_right
from collections import defaultdict
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler, Subset

from .tep_hetero_graph import window_statistics


class StatisticalFeatureDataset(Dataset[tuple[Tensor, Tensor]]):
    """Flatten the seven statistics of all 52 variables for an MLP."""

    def __init__(self, window_dataset: Dataset, graph_config: Mapping[str, Any]) -> None:
        self.window_dataset = window_dataset
        self.graph_config = graph_config

    def __len__(self) -> int:
        return len(self.window_dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sample = self.window_dataset[index]
        features = window_statistics(
            sample.x, sample.observed_mask, self.graph_config["statistics"]
        )
        return features.flatten(), sample.y.reshape(()).long()

    def set_epoch(self, epoch: int) -> None:
        target = self.window_dataset
        if isinstance(target, Subset):
            target = target.dataset
        if hasattr(target, "set_epoch"):
            target.set_epoch(epoch)


def _run_balanced_indices(
    dataset: Dataset,
    maximum: int,
    seed: int,
    windows_per_run: int,
) -> list[int]:
    if not hasattr(dataset, "runs") or not hasattr(dataset, "_cumulative"):
        raise TypeError("run_balanced sampling requires a TEPWindowDataset-like dataset.")
    if int(windows_per_run) <= 0:
        raise ValueError("sampling.windows_per_run must be positive.")
    grouped: dict[int, list[int]] = defaultdict(list)
    for run_index, run in enumerate(dataset.runs):
        grouped[int(run.fault_id)].append(run_index)
    if not grouped:
        raise ValueError("Cannot run-balance an empty run collection.")
    rng = random.Random(int(seed))
    classes = sorted(grouped)
    base, remainder = divmod(int(maximum), len(classes))
    selected: list[int] = []
    for class_position, fault_id in enumerate(classes):
        quota = base + int(class_position < remainder)
        run_order = list(grouped[fault_id])
        rng.shuffle(run_order)
        class_indices: list[int] = []
        for run_index in run_order:
            preceding = 0 if run_index == 0 else int(dataset._cumulative[run_index - 1])
            count = int(dataset._cumulative[run_index]) - preceding
            take = min(int(windows_per_run), count, quota - len(class_indices))
            if take > 0:
                local = rng.sample(range(count), take)
                class_indices.extend(preceding + value for value in local)
            if len(class_indices) == quota:
                break
        if len(class_indices) != quota:
            raise ValueError(
                f"Fault {fault_id} provides {len(class_indices)} sampled windows, below quota {quota}."
            )
        selected.extend(class_indices)
    return sorted(selected)


def deterministic_subset(
    dataset: Dataset,
    maximum: int | None,
    seed: int,
    strategy: str = "uniform_windows",
    windows_per_run: int | None = None,
) -> Dataset:
    """Select a deterministic model-shared window budget after run splitting."""
    if maximum is None or int(maximum) >= len(dataset):
        return dataset
    if int(maximum) <= 0:
        raise ValueError("A configured sample limit must be positive or null.")
    if strategy == "uniform_windows":
        indices = sorted(random.Random(int(seed)).sample(range(len(dataset)), int(maximum)))
    elif strategy == "run_balanced":
        if windows_per_run is None:
            raise ValueError("run_balanced sampling requires sampling.windows_per_run.")
        indices = _run_balanced_indices(dataset, int(maximum), int(seed), int(windows_per_run))
    else:
        raise ValueError("sampling.strategy must be uniform_windows or run_balanced.")
    return Subset(dataset, indices)


class RunGroupedSampler(Sampler[int]):
    """Shuffle sampled run blocks while keeping each run's windows cache-local."""

    def __init__(
        self,
        graph_dataset: Dataset,
        generator: torch.Generator,
        chunk_size: int,
    ) -> None:
        if int(chunk_size) <= 0:
            raise ValueError("sampling.run_group_chunk_size must be positive.")
        target = getattr(graph_dataset, "window_dataset", graph_dataset)
        if not isinstance(target, Subset) or not hasattr(target.dataset, "_cumulative"):
            raise TypeError("RunGroupedSampler requires a Subset of TEPWindowDataset.")
        run_blocks: dict[int, list[int]] = defaultdict(list)
        for subset_position, global_index in enumerate(target.indices):
            run_index = bisect_right(target.dataset._cumulative, int(global_index))
            run_blocks[run_index].append(subset_position)
        self.blocks_by_fault: dict[int, list[list[int]]] = defaultdict(list)
        for run_index, block in run_blocks.items():
            fault_id = int(target.dataset.runs[run_index].fault_id)
            self.blocks_by_fault[fault_id].append(block)
        self.generator = generator
        self.chunk_size = int(chunk_size)
        self.length = len(target)

    def __iter__(self):
        ordered: dict[int, list[list[int]]] = {}
        for fault_id, blocks in self.blocks_by_fault.items():
            permutation = torch.randperm(len(blocks), generator=self.generator).tolist()
            ordered[fault_id] = [blocks[index] for index in permutation]
        max_runs = max(len(blocks) for blocks in ordered.values())
        for run_position in range(max_runs):
            active = [
                blocks[run_position]
                for blocks in ordered.values()
                if run_position < len(blocks)
            ]
            max_chunks = max((len(block) + self.chunk_size - 1) // self.chunk_size for block in active)
            for chunk_index in range(max_chunks):
                available = [
                    block for block in active if chunk_index * self.chunk_size < len(block)
                ]
                order = torch.randperm(len(available), generator=self.generator).tolist()
                for index in order:
                    start = chunk_index * self.chunk_size
                    chunk = available[index][start : start + self.chunk_size]
                    within = torch.randperm(len(chunk), generator=self.generator).tolist()
                    for position in within:
                        yield chunk[position]

    def __len__(self) -> int:
        return self.length


# Design note (paper Sec. 4.2, 5.1): sampling occurs only after run splitting;
# fault-balanced run blocks share indices across models and reduce Parquet I/O.
