"""Determinism and coverage tests for the Stage-5 run-balanced sampler."""

from __future__ import annotations

from dataclasses import dataclass
import unittest

import torch
from torch.utils.data import Dataset, Subset

from data.feature_dataset import RunGroupedSampler, deterministic_subset


@dataclass
class _Run:
    fault_id: int


class _Windows(Dataset):
    def __init__(self) -> None:
        self.runs = [_Run(fault) for fault in range(3) for _ in range(4)]
        self._cumulative = [(index + 1) * 10 for index in range(len(self.runs))]

    def __len__(self) -> int:
        return self._cumulative[-1]

    def __getitem__(self, index: int) -> int:
        return index


class _GraphView:
    def __init__(self, window_dataset: Subset) -> None:
        self.window_dataset = window_dataset


class SamplingTests(unittest.TestCase):
    def test_run_balanced_subset_is_deterministic_and_balanced(self) -> None:
        dataset = _Windows()
        left = deterministic_subset(dataset, 60, 7, "run_balanced", 5)
        right = deterministic_subset(dataset, 60, 7, "run_balanced", 5)
        self.assertEqual(left.indices, right.indices)
        counts = [0, 0, 0]
        for index in left.indices:
            counts[index // 40] += 1
        self.assertEqual(counts, [20, 20, 20])

    def test_grouped_sampler_visits_each_subset_position_once(self) -> None:
        subset = deterministic_subset(_Windows(), 60, 7, "run_balanced", 5)
        sampler = RunGroupedSampler(
            _GraphView(subset), torch.Generator().manual_seed(0), chunk_size=2
        )
        order = list(sampler)
        self.assertEqual(len(order), len(subset))
        self.assertEqual(sorted(order), list(range(len(subset))))


if __name__ == "__main__":
    unittest.main()


# Design note (paper Sec. 5.1): tests enforce fixed, fault-balanced sampling
# and exact one-pass coverage without changing the preceding run-level split.
