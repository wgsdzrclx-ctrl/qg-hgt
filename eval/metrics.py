"""Classification, robustness, calibration, and process-monitoring metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Iterable, Sequence

import numpy as np


def _as_labels(values: Sequence[int] | np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.int64).reshape(-1)
    if result.size == 0:
        raise ValueError("Metric inputs must not be empty.")
    return result


def _as_probabilities(values: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0:
        raise ValueError("Probabilities must have shape [samples, classes].")
    if not np.isfinite(result).all() or (result < 0.0).any():
        raise ValueError("Probabilities must be finite and non-negative.")
    row_sums = result.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1.0e-6):
        raise ValueError("Each probability row must sum to one.")
    return result


def metrics_from_confusion(matrix: Sequence[Sequence[int]] | np.ndarray) -> dict[str, Any]:
    """Derive Accuracy, Macro/Weighted-F1 and per-class P/R/F1 from a matrix."""
    cm = np.asarray(matrix, dtype=np.int64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or cm.shape[0] == 0:
        raise ValueError("Confusion matrix must be a non-empty square matrix.")
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    true_positive = np.diag(cm).astype(np.float64)
    precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support > 0)
    denominator = precision + recall
    f1 = np.divide(2.0 * precision * recall, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    total = float(support.sum())
    weighted_f1 = float(np.dot(f1, support) / total) if total else 0.0
    per_class = [
        {
            "class_id": int(index),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index in range(cm.shape[0])
    ]
    return {
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "macro_f1": float(f1.mean()),
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "num_examples": int(total),
    }


def classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    num_classes: int,
) -> dict[str, Any]:
    """Compute the paper Sec. 5.4 classification metrics."""
    truth = _as_labels(y_true)
    prediction = _as_labels(y_pred)
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have equal lengths.")
    classes = int(num_classes)
    if classes <= 1:
        raise ValueError("num_classes must be greater than one.")
    if (truth < 0).any() or (truth >= classes).any() or (prediction < 0).any() or (prediction >= classes).any():
        raise ValueError("Class labels must lie in [0, num_classes).")
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (truth, prediction), 1)
    return metrics_from_confusion(matrix)


def calibration_metrics(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    num_bins: int,
    epsilon: float,
) -> dict[str, float]:
    """Compute top-label ECE, multiclass Brier score, and NLL."""
    truth = _as_labels(y_true)
    probability = _as_probabilities(probabilities)
    if len(truth) != len(probability):
        raise ValueError("Labels and probabilities must have equal sample counts.")
    if int(num_bins) <= 0 or not 0.0 < float(epsilon) < 1.0:
        raise ValueError("num_bins must be positive and epsilon must lie in (0, 1).")
    confidence = probability.max(axis=1)
    prediction = probability.argmax(axis=1)
    correct = (prediction == truth).astype(np.float64)
    edges = np.linspace(0.0, 1.0, int(num_bins) + 1)
    bin_index = np.minimum(np.searchsorted(edges, confidence, side="right") - 1, int(num_bins) - 1)
    ece = 0.0
    for index in range(int(num_bins)):
        selected = bin_index == index
        if selected.any():
            ece += selected.mean() * abs(float(correct[selected].mean() - confidence[selected].mean()))
    one_hot = np.eye(probability.shape[1], dtype=np.float64)[truth]
    brier = np.square(probability - one_hot).sum(axis=1).mean()
    nll = -np.log(np.clip(probability[np.arange(len(truth)), truth], float(epsilon), 1.0)).mean()
    return {"ece": float(ece), "brier_score": float(brier), "nll": float(nll)}


def delta_f1(clean_macro_f1: float, corrupted_macro_f1: float) -> float:
    """Return the robustness drop defined in paper Eq. 5.4."""
    return float(clean_macro_f1) - float(corrupted_macro_f1)


def performance_curve(
    strengths: Sequence[float], macro_f1_values: Sequence[float]
) -> list[dict[str, float]]:
    """Return a strength-sorted Macro-F1 curve."""
    if len(strengths) != len(macro_f1_values) or not strengths:
        raise ValueError("Strength and Macro-F1 sequences must be equally sized and non-empty.")
    pairs = sorted((float(x), float(y)) for x, y in zip(strengths, macro_f1_values))
    return [{"strength": x, "macro_f1": y} for x, y in pairs]


def multi_strength_summary(
    strengths: Sequence[float], macro_f1_values: Sequence[float]
) -> dict[str, Any]:
    """Compute the mean and range-normalized trapezoidal AUC over strengths."""
    curve = performance_curve(strengths, macro_f1_values)
    x = np.asarray([point["strength"] for point in curve], dtype=np.float64)
    y = np.asarray([point["macro_f1"] for point in curve], dtype=np.float64)
    if len(x) == 1 or float(x[-1] - x[0]) == 0.0:
        normalized_auc = float(y.mean())
    else:
        area = np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5)
        normalized_auc = float(area / (x[-1] - x[0]))
    return {"curve": curve, "mean_macro_f1": float(y.mean()), "normalized_auc": normalized_auc}


def process_monitoring_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    run_ids: Sequence[Hashable],
    window_times: Sequence[int] | np.ndarray,
    normal_class: int,
) -> dict[str, Any]:
    """Compute window FDR/FAR/MAR and per-run first-detection delay."""
    truth = _as_labels(y_true)
    prediction = _as_labels(y_pred)
    times = np.asarray(window_times, dtype=np.int64).reshape(-1)
    if not (len(truth) == len(prediction) == len(run_ids) == len(times)):
        raise ValueError("Process-monitoring inputs must have equal lengths.")
    actual_fault = truth != int(normal_class)
    predicted_fault = prediction != int(normal_class)
    fault_total = int(actual_fault.sum())
    normal_total = int((~actual_fault).sum())
    detected = int((actual_fault & predicted_fault).sum())
    false_alarms = int(((~actual_fault) & predicted_fault).sum())
    fdr = detected / fault_total if fault_total else 0.0
    far = false_alarms / normal_total if normal_total else 0.0

    delays: dict[str, int | None] = {}
    ordered_runs = dict.fromkeys(run_ids)
    run_array = np.asarray([str(value) for value in run_ids], dtype=object)
    for raw_run in ordered_runs:
        run = str(raw_run)
        selected = run_array == run
        run_fault = selected & actual_fault
        if not run_fault.any():
            continue
        onset = int(times[run_fault].min())
        # A detection must fall on an actual fault window.  This prevents a
        # normal-window false alarm after a finite fault episode from being
        # counted as a delayed detection.
        post_onset_detection = selected & actual_fault & predicted_fault & (times >= onset)
        delays[run] = int(times[post_onset_detection].min() - onset) if post_onset_detection.any() else None
    finite_delays = [value for value in delays.values() if value is not None]
    return {
        "fdr": float(fdr),
        "far": float(far),
        "mar": float(1.0 - fdr),
        "detection_delay_mean": float(np.mean(finite_delays)) if finite_delays else None,
        "detection_delay_by_run": delays,
        "detected_fault_runs": len(finite_delays),
        "fault_runs": len(delays),
    }


@dataclass
class StreamingClassificationMetrics:
    """Memory-bounded accumulator used by full extended500 evaluation."""

    num_classes: int
    num_bins: int
    epsilon: float
    confusion: np.ndarray = field(init=False)
    bin_count: np.ndarray = field(init=False)
    bin_confidence: np.ndarray = field(init=False)
    bin_correct: np.ndarray = field(init=False)
    brier_sum: float = 0.0
    nll_sum: float = 0.0
    count: int = 0

    def __post_init__(self) -> None:
        if self.num_classes <= 1 or self.num_bins <= 0 or not 0.0 < self.epsilon < 1.0:
            raise ValueError("Invalid streaming metric configuration.")
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.bin_count = np.zeros(self.num_bins, dtype=np.int64)
        self.bin_confidence = np.zeros(self.num_bins, dtype=np.float64)
        self.bin_correct = np.zeros(self.num_bins, dtype=np.float64)

    def update(self, y_true: np.ndarray, probabilities: np.ndarray) -> None:
        truth = _as_labels(y_true)
        probability = _as_probabilities(probabilities)
        if len(truth) != len(probability) or probability.shape[1] != self.num_classes:
            raise ValueError("Streaming batch shapes do not match the configured class count.")
        prediction = probability.argmax(axis=1)
        np.add.at(self.confusion, (truth, prediction), 1)
        confidence = probability.max(axis=1)
        correct = (prediction == truth).astype(np.float64)
        indices = np.minimum((confidence * self.num_bins).astype(np.int64), self.num_bins - 1)
        np.add.at(self.bin_count, indices, 1)
        np.add.at(self.bin_confidence, indices, confidence)
        np.add.at(self.bin_correct, indices, correct)
        self.brier_sum += float(np.square(probability).sum() - 2.0 * probability[np.arange(len(truth)), truth].sum() + len(truth))
        self.nll_sum += float(-np.log(np.clip(probability[np.arange(len(truth)), truth], self.epsilon, 1.0)).sum())
        self.count += len(truth)

    def compute(self) -> dict[str, Any]:
        result = metrics_from_confusion(self.confusion)
        if self.count == 0:
            raise ValueError("Cannot compute metrics before any update.")
        nonempty = self.bin_count > 0
        accuracy = np.divide(self.bin_correct, self.bin_count, out=np.zeros_like(self.bin_correct), where=nonempty)
        confidence = np.divide(self.bin_confidence, self.bin_count, out=np.zeros_like(self.bin_confidence), where=nonempty)
        result.update(
            ece=float((self.bin_count * np.abs(accuracy - confidence)).sum() / self.count),
            brier_score=float(self.brier_sum / self.count),
            nll=float(self.nll_sum / self.count),
        )
        return result


@dataclass
class StreamingProcessMonitor:
    """Streaming window-level FDR/FAR/MAR and run-level delay accumulator."""

    normal_class: int
    fault_windows: int = 0
    detected_fault_windows: int = 0
    normal_windows: int = 0
    false_alarm_windows: int = 0
    onset_by_run: dict[str, int] = field(default_factory=dict)
    first_detection_by_run: dict[str, int] = field(default_factory=dict)

    def update(
        self,
        y_true: Iterable[int],
        y_pred: Iterable[int],
        run_ids: Iterable[Hashable],
        window_times: Iterable[int],
    ) -> None:
        for truth, prediction, run_id, time in zip(y_true, y_pred, run_ids, window_times):
            actual_fault = int(truth) != self.normal_class
            predicted_fault = int(prediction) != self.normal_class
            if actual_fault:
                self.fault_windows += 1
                self.detected_fault_windows += int(predicted_fault)
                key = str(run_id)
                self.onset_by_run[key] = min(self.onset_by_run.get(key, int(time)), int(time))
                if predicted_fault:
                    self.first_detection_by_run[key] = min(
                        self.first_detection_by_run.get(key, int(time)), int(time)
                    )
            else:
                self.normal_windows += 1
                self.false_alarm_windows += int(predicted_fault)

    def compute(self, include_per_run: bool = True) -> dict[str, Any]:
        fdr = self.detected_fault_windows / self.fault_windows if self.fault_windows else 0.0
        far = self.false_alarm_windows / self.normal_windows if self.normal_windows else 0.0
        delays = {
            run: self.first_detection_by_run[run] - onset if run in self.first_detection_by_run else None
            for run, onset in self.onset_by_run.items()
        }
        finite = [value for value in delays.values() if value is not None]
        result = {
            "fdr": float(fdr),
            "far": float(far),
            "mar": float(1.0 - fdr),
            "detection_delay_mean": float(np.mean(finite)) if finite else None,
            "detected_fault_runs": len(finite),
            "fault_runs": len(delays),
        }
        if include_per_run:
            result["detection_delay_by_run"] = delays
        return result


# Design note (paper Sec. 5.4): Macro-F1 is primary; the same predictions also
# produce class-wise, robustness, calibration, and window-level monitoring metrics.
