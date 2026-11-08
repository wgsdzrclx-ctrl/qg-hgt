"""Range and identity tests for Stage-5 metrics."""

from __future__ import annotations

import unittest

import numpy as np

from eval.metrics import (
    StreamingClassificationMetrics,
    StreamingProcessMonitor,
    calibration_metrics,
    classification_metrics,
    delta_f1,
    multi_strength_summary,
    performance_curve,
    process_monitoring_metrics,
)


class MetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(0)
        self.classes = 5
        self.labels = self.rng.integers(0, self.classes, size=200)
        raw = self.rng.random((200, self.classes))
        self.probabilities = raw / raw.sum(axis=1, keepdims=True)
        self.predictions = self.probabilities.argmax(axis=1)

    def test_classification_ranges(self) -> None:
        result = classification_metrics(self.labels, self.predictions, self.classes)
        for name in ("accuracy", "macro_f1", "weighted_f1"):
            self.assertGreaterEqual(result[name], 0.0)
            self.assertLessEqual(result[name], 1.0)
        self.assertEqual(np.asarray(result["confusion_matrix"]).shape, (self.classes, self.classes))
        for per_class in result["per_class"]:
            for name in ("precision", "recall", "f1"):
                self.assertTrue(0.0 <= per_class[name] <= 1.0)

    def test_calibration_ranges(self) -> None:
        result = calibration_metrics(self.labels, self.probabilities, num_bins=10, epsilon=1.0e-12)
        self.assertTrue(0.0 <= result["ece"] <= 1.0)
        self.assertTrue(0.0 <= result["brier_score"] <= 2.0)
        self.assertGreaterEqual(result["nll"], 0.0)

    def test_streaming_matches_batch(self) -> None:
        stream = StreamingClassificationMetrics(self.classes, num_bins=10, epsilon=1.0e-12)
        stream.update(self.labels[:80], self.probabilities[:80])
        stream.update(self.labels[80:], self.probabilities[80:])
        result = stream.compute()
        batch = classification_metrics(self.labels, self.predictions, self.classes)
        calibration = calibration_metrics(self.labels, self.probabilities, 10, 1.0e-12)
        self.assertAlmostEqual(result["macro_f1"], batch["macro_f1"])
        self.assertAlmostEqual(result["brier_score"], calibration["brier_score"])
        self.assertAlmostEqual(result["nll"], calibration["nll"])

    def test_robustness_metrics(self) -> None:
        self.assertAlmostEqual(delta_f1(0.8, 0.6), 0.2)
        result = multi_strength_summary([0.1, 0.2, 0.4], [0.8, 0.7, 0.5])
        curve = performance_curve([0.4, 0.1, 0.2], [0.5, 0.8, 0.7])
        self.assertEqual([point["strength"] for point in curve], [0.1, 0.2, 0.4])
        self.assertTrue(0.0 <= result["mean_macro_f1"] <= 1.0)
        self.assertTrue(0.0 <= result["normalized_auc"] <= 1.0)

    def test_process_monitoring_ranges(self) -> None:
        truth = [0, 0, 1, 1, 0, 2, 2]
        prediction = [0, 1, 0, 1, 0, 0, 2]
        result = process_monitoring_metrics(
            truth, prediction, ["a"] * 4 + ["b"] * 3, list(range(4)) + list(range(3)), 0
        )
        for name in ("fdr", "far", "mar"):
            self.assertTrue(0.0 <= result[name] <= 1.0)
        self.assertGreaterEqual(result["detection_delay_mean"], 0.0)
        stream = StreamingProcessMonitor(normal_class=0)
        stream.update(truth[:4], prediction[:4], ["a"] * 4, range(4))
        stream.update(truth[4:], prediction[4:], ["b"] * 3, range(3))
        streamed = stream.compute()
        for name in ("fdr", "far", "mar"):
            self.assertAlmostEqual(result[name], streamed[name])

    def test_post_fault_false_alarm_is_not_detection(self) -> None:
        result = process_monitoring_metrics(
            [0, 1, 1, 0], [0, 0, 0, 1], ["finite"] * 4, [0, 1, 2, 3], 0
        )
        self.assertEqual(result["fdr"], 0.0)
        self.assertEqual(result["mar"], 1.0)
        self.assertIsNone(result["detection_delay_mean"])
        self.assertIsNone(result["detection_delay_by_run"]["finite"])


if __name__ == "__main__":
    unittest.main()


# Design note (paper Sec. 5.4): deterministic random predictions verify metric
# domains and streaming/batch equivalence without using experimental data.
