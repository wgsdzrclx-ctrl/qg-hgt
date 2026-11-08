"""Metric-based early stopping state."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float
    mode: str
    best: float | None = None
    bad_epochs: int = 0

    def update(self, value: float) -> tuple[bool, bool]:
        """Return (improved, should_stop)."""
        if self.mode not in {"min", "max"}:
            raise ValueError("EarlyStopping mode must be 'min' or 'max'.")
        improved = self.best is None
        if self.best is not None:
            improved = value < self.best - self.min_delta if self.mode == "min" else value > self.best + self.min_delta
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience


# Design note: early stopping selects optimization time from validation data
# only and never accesses the test split.
