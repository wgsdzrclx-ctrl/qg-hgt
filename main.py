"""End-to-end TEP experiment orchestration for the shared Stage-4 interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch_geometric.loader import DataLoader

from data.feature_dataset import deterministic_subset
from data.tep_dataset import build_tep_datasets_from_config, load_data_config
from data.tep_hetero_graph import TEPGraphDataset
from eval import evaluate
from models import build_model
from train import build_parser, fit
from utils import create_logger, set_seed


def _load_experiment_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_dir"] = str(path.parent)
    return config


def _resolve(config: dict[str, Any], value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["_config_dir"]) / path
    return path.resolve()


def _select_device(configured: str) -> torch.device:
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(configured)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def _loader(dataset: TEPGraphDataset, split: str, seed: int, config: dict[str, Any]) -> DataLoader:
    training = config["training"]
    workers = int(training["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=bool(training["shuffle"][split]),
        num_workers=workers,
        pin_memory=bool(training["pin_memory"]),
        persistent_workers=bool(training["persistent_workers"]) and workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


def run_experiment(model_name: str, config_path: str | Path, seed: int) -> dict[str, Any]:
    if model_name == "hgt_fixed_quality":
        raise RuntimeError(
            "hgt_fixed_quality is an explicitly labelled oracle diagnostic and cannot run the "
            "normal validation/test pipeline because q_star is derived from injected degradation "
            "parameters. Use hgt_learnable_quality for a deployable comparison."
        )
    experiment_path = Path(config_path).expanduser().resolve()
    config = _load_experiment_config(experiment_path)
    data_config_path = _resolve(config, str(config["data"]["config_path"]))
    data_config = load_data_config(data_config_path)
    variant_override = config["data"]["variant_override"]
    if variant_override is not None:
        data_config["dataset"]["variant"] = str(variant_override)
    set_seed(
        seed, data_config["experiment"]["seeds"], bool(config["runtime"]["deterministic"]),
        bool(config["runtime"]["deterministic_warn_only"]),
    )
    torch.set_num_threads(int(config["runtime"]["cpu_threads"]))
    device = _select_device(str(config["runtime"]["device"]))

    output_dir = _resolve(config, str(config["output"]["directory"]))
    result_path = output_dir / f"{model_name}_{seed}.json"
    checkpoint_path = output_dir / str(config["output"]["checkpoint_subdirectory"]) / f"{model_name}_{seed}.pt"
    log_path = output_dir / str(config["output"]["log_subdirectory"]) / f"{model_name}_{seed}.log"
    logger = create_logger(f"tep.{model_name}.{seed}", log_path, str(config["logging"]["level"]))
    logger.info("model=%s seed=%d device=%s config=%s", model_name, seed, device, experiment_path)

    window_datasets = build_tep_datasets_from_config(
        data_config, seed=seed, test_scenario=str(config["data"]["test_scenario"]),
        print_leakage_report=bool(config["data"]["print_leakage_report"]),
    )
    graph_datasets: dict[str, TEPGraphDataset] = {}
    for split, dataset in window_datasets.items():
        sampling = config["sampling"]
        subset = deterministic_subset(
            dataset, sampling["max_samples_per_split"][split],
            int(sampling["base_seed"]) + int(sampling["seed_offsets"][split]),
        )
        graph_datasets[split] = TEPGraphDataset(subset, config["graph"], homogeneous=False)
    loaders = {split: _loader(dataset, split, seed, config) for split, dataset in graph_datasets.items()}

    variant = str(data_config["dataset"]["variant"])
    layout = data_config["dataset"]["variants"][variant]
    input_dim = int(data_config["dataset"]["num_variables"]) * len(config["graph"]["statistics"]["feature_names"])
    num_classes = int(layout["num_classes"])
    model = build_model(model_name, input_dim, num_classes, config)
    training_result = fit(
        model, loaders["train"], loaders["val"], device, num_classes, config, checkpoint_path, logger
    )
    test_metrics = evaluate(
        model,
        loaders["test"],
        device,
        num_classes,
        metric_config=config["evaluation"]["metrics"],
    )
    result = {
        "model": model_name,
        "seed": seed,
        "dataset_variant": variant,
        "split_seed": int(layout.get("split", data_config["dataset"]["split"])["seed"]),
        "sampling_seed": int(config["sampling"]["base_seed"]),
        "num_classes": num_classes,
        "split_samples": {name: len(dataset) for name, dataset in graph_datasets.items()},
        "best_epoch": training_result["best_epoch"],
        "validation": training_result["best_validation"],
        "test": test_metrics,
        "accuracy": test_metrics["accuracy"],
        "macro_f1": test_metrics["macro_f1"],
        "checkpoint": str(checkpoint_path),
        "config": str(experiment_path),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=int(config["output"]["json_indent"]))
    logger.info("test_accuracy=%.6f test_macro_f1=%.6f result=%s", result["accuracy"], result["macro_f1"], result_path)
    return result


def cli() -> None:
    arguments = build_parser().parse_args()
    run_experiment(arguments.model, arguments.config, arguments.seed)


if __name__ == "__main__":
    cli()


# Design note (paper Sec. 5.1-5.4): windows are created only after run-first
# splitting; every baseline receives the same graph batch and held-out test set,
# and each seed writes Accuracy/Macro-F1 for later mean +/- standard deviation.
