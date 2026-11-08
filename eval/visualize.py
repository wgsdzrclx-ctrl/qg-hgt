"""Generate Stage-5 paper figures from JSON summaries and sampled NPZ artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import yaml


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.summarize import load_results  # noqa: E402


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _projection(features: np.ndarray, method: str, config: dict[str, Any]) -> np.ndarray:
    if len(features) < 3:
        raise ValueError("At least three artifact samples are required for projection.")
    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = min(float(config["tsne_perplexity"]), float(len(features) - 1))
        return TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=int(config["random_state"]),
            init="pca",
            learning_rate="auto",
        ).fit_transform(features)
    if method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise RuntimeError("UMAP visualization requires the optional umap-learn package.") from exc
        return umap.UMAP(
            n_components=2,
            n_neighbors=int(config["umap_neighbors"]),
            min_dist=float(config["umap_min_dist"]),
            random_state=int(config["random_state"]),
        ).fit_transform(features)
    raise ValueError("visualization.method must be tsne or umap.")


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _artifact_path(root: Path, protocol: str, model: str, seed: int, scenario: str) -> Path:
    safe = "".join(character if character.isalnum() or character in "_.-" else "_" for character in scenario)
    return root / protocol / "main" / f"{model}_{seed}_{safe}.npz"


def _embedding_figures(
    artifact_root: Path, output: Path, protocol: str, model: str, seed: int, config: dict[str, Any]
) -> None:
    scenarios = {"clean": "clean", "mixed": "protocol_group=unseen_combination"}
    for display, scenario in scenarios.items():
        path = _artifact_path(artifact_root, protocol, model, seed, scenario)
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as archive:
            coordinates = _projection(archive["representation"], str(config["method"]), config)
            labels = archive["label"]
        fig, axis = plt.subplots(figsize=tuple(config["figure_size"]))
        scatter = axis.scatter(coordinates[:, 0], coordinates[:, 1], c=labels, s=9, alpha=0.7, cmap=config["colormap"])
        axis.set_title(f"{model}: {display} ({config['method'].upper()})")
        axis.set_xlabel("Dimension 1")
        axis.set_ylabel("Dimension 2")
        fig.colorbar(scatter, ax=axis, label="Fault class")
        _save(fig, output / f"embedding_{display}_{config['method']}.png", config["dpi"])


def _confusion_figure(document: dict[str, Any], output: Path, config: dict[str, Any]) -> None:
    scenarios = [("Clean", "clean"), ("Mixed degradation", "protocol_group=unseen_combination")]
    available = [(title, key) for title, key in scenarios if key in document["evaluations"]]
    if not available:
        return
    fig, axes = plt.subplots(1, len(available), figsize=(config["figure_size"][0] * len(available), config["figure_size"][1]))
    axes = np.atleast_1d(axes)
    for axis, (title, key) in zip(axes, available):
        matrix = np.asarray(document["evaluations"][key]["confusion_matrix"], dtype=np.float64)
        row_sum = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, row_sum, out=np.zeros_like(matrix), where=row_sum > 0)
        image = axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
        axis.set_title(title)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        fig.colorbar(image, ax=axis, fraction=0.046)
    _save(fig, output / "confusion_clean_vs_degraded.png", config["dpi"])


def _reliability_figure(
    documents: list[dict[str, Any]], output: Path, protocol: str, model: str, config: dict[str, Any]
) -> None:
    selected = [doc for doc in documents if doc["protocol"] == protocol and doc["model"] == model]
    grouped: dict[str, dict[float, list[float]]] = {}
    for document in selected:
        for scenario, metrics in document["evaluations"].items():
            if "=" not in scenario or scenario.startswith("protocol_group="):
                continue
            value = metrics.get("mean_sensor_reliability")
            if value is None:
                continue
            corruption, strength = scenario.split("=", 1)
            grouped.setdefault(corruption, {}).setdefault(float(strength), []).append(float(value))
    if not grouped:
        return
    fig, axis = plt.subplots(figsize=tuple(config["figure_size"]))
    for corruption, points in sorted(grouped.items()):
        strengths = sorted(points)
        means = [statistics.fmean(points[value]) for value in strengths]
        axis.plot(strengths, means, marker="o", label=corruption)
    axis.set_xlabel("Degradation strength")
    axis.set_ylabel("Mean sensor reliability")
    axis.set_title(f"Observable reliability response: {model}")
    axis.legend()
    axis.grid(alpha=0.25)
    _save(fig, output / "sensor_reliability_vs_strength.png", config["dpi"])


def _missing_curve(
    documents: list[dict[str, Any]], output: Path, protocol: str, models: list[str], config: dict[str, Any]
) -> None:
    fig, axis = plt.subplots(figsize=tuple(config["figure_size"]))
    plotted = False
    for model in models:
        selected = [doc for doc in documents if doc["protocol"] == protocol and doc["model"] == model]
        grouped: dict[float, list[float]] = {}
        for document in selected:
            for scenario, metrics in document["evaluations"].items():
                if scenario.startswith("random_missing="):
                    strength = float(scenario.split("=", 1)[1])
                    grouped.setdefault(strength, []).append(float(metrics["macro_f1"]))
        if grouped:
            strengths = sorted(grouped)
            axis.plot(strengths, [statistics.fmean(grouped[x]) for x in strengths], marker="o", label=model)
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    axis.set_xlabel("Random missing rate")
    axis.set_ylabel("Macro-F1")
    axis.set_title(f"Missing-rate robustness ({protocol})")
    axis.legend(fontsize="small", ncol=2)
    axis.grid(alpha=0.25)
    _save(fig, output / "missing_rate_macro_f1_all_models.png", config["dpi"])


def _fault_trace(artifact: Path, output: Path, normal_class: int, config: dict[str, Any]) -> None:
    if not artifact.exists():
        return
    with np.load(artifact, allow_pickle=False) as archive:
        required = {"trace_probabilities", "trace_label", "trace_prediction", "trace_window_end"}
        if not required.issubset(archive.files):
            return
        probabilities = archive["trace_probabilities"]
        labels = archive["trace_label"].astype(int)
        predictions = archive["trace_prediction"].astype(int)
        times = archive["trace_window_end"].astype(int)
    fault_labels = labels[labels != int(normal_class)]
    if not len(fault_labels):
        return
    fault_class = int(np.bincount(fault_labels).argmax())
    onset = int(times[labels != int(normal_class)].min())
    detections = times[(times >= onset) & (predictions != int(normal_class))]
    fig, axis = plt.subplots(figsize=tuple(config["figure_size"]))
    order = np.argsort(times)
    axis.plot(times[order], probabilities[order, fault_class], label=f"P(class={fault_class})")
    axis.axvline(onset, color="black", linestyle="--", label="Fault onset")
    if len(detections):
        first = int(detections.min())
        axis.axvline(first, color="red", linestyle=":", label=f"Detection, delay={first-onset}")
    axis.set_xlabel("Window end index")
    axis.set_ylabel("Predicted probability")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Fault probability and window-level detection delay")
    axis.legend()
    axis.grid(alpha=0.25)
    _save(fig, output / "fault_probability_detection_delay.png", config["dpi"])


def visualize(config_path: Path, protocol: str, seed: int) -> Path:
    stage5 = _load_yaml(config_path)
    results_root = _resolve(config_path.parent, str(stage5["output"]["results_directory"]))
    output = results_root / str(stage5["output"]["figure_subdirectory"]) / protocol
    output.mkdir(parents=True, exist_ok=True)
    artifacts_root = results_root / str(stage5["output"]["artifact_subdirectory"])
    documents = load_results(results_root)
    config = stage5["visualization"]
    model = str(config["reliability_model"])
    selected = next(
        (
            doc for doc in documents
            if doc["protocol"] == protocol and doc["model"] == model and int(doc["seed"]) == int(seed)
        ),
        None,
    )
    if bool(config.get("embedding_enabled", True)):
        _embedding_figures(artifacts_root, output, protocol, model, seed, config)
    if selected is not None:
        _confusion_figure(selected, output, config)
    _reliability_figure(documents, output, protocol, model, config)
    _missing_curve(documents, output, protocol, [str(v) for v in stage5["models"]["main"]], config)
    trace_model = str(config["fault_trace_model"])
    trace_path = _artifact_path(artifacts_root, protocol, trace_model, seed, "protocol_group=unseen_combination")
    _fault_trace(trace_path, output, int(stage5["metrics"]["normal_class"]), config)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "stage5.yaml")
    parser.add_argument("--protocol", default="protocol_2", choices=("protocol_1", "protocol_2"))
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    destination = visualize(arguments.config.expanduser().resolve(), arguments.protocol, arguments.seed)
    print(f"Stage-5 figures written to: {destination}")


if __name__ == "__main__":
    main()


# Design note (paper Sec. 5.8): figures expose representation separation,
# confusion, learned reliability, missing-data robustness, and temporal detection.
