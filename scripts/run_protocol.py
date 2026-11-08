"""Train each seed once and evaluate every Stage-5 degradation scenario."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch import nn
from torch_geometric.loader import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.feature_dataset import RunGroupedSampler, deterministic_subset  # noqa: E402
from data.tep_dataset import (  # noqa: E402
    build_tep_datasets_from_config,
    build_tep_protocol_test_sets_from_config,
    load_data_config,
)
from data.tep_hetero_graph import TEPGraphDataset  # noqa: E402
from eval import evaluate  # noqa: E402
from eval.metrics import delta_f1, multi_strength_summary  # noqa: E402
from models import build_model  # noqa: E402
from train import fit  # noqa: E402
from utils import create_logger, load_checkpoint, set_seed  # noqa: E402


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _deep_update(target: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)
    return target


def _device(configured: str) -> torch.device:
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(configured)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return result


def _loader(dataset: TEPGraphDataset, split: str, seed: int, experiment: Mapping[str, Any]) -> DataLoader:
    training = experiment["training"]
    workers = int(training["num_workers"])
    shuffle = bool(training["shuffle"][split])
    sampler = None
    if split == "train" and shuffle and experiment["sampling"].get("train_order") == "run_grouped":
        sampler = RunGroupedSampler(
            dataset,
            torch.Generator().manual_seed(int(seed)),
            int(experiment["sampling"]["run_group_chunk_size"]),
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=bool(training["pin_memory"]),
        persistent_workers=bool(training["persistent_workers"]) and workers > 0,
        generator=torch.Generator().manual_seed(int(seed)),
    )


def _graph_dataset(dataset: Any, split: str, experiment: Mapping[str, Any]) -> TEPGraphDataset:
    sampling = experiment["sampling"]
    subset = deterministic_subset(
        dataset,
        sampling["max_samples_per_split"][split],
        int(sampling["base_seed"]) + int(sampling["seed_offsets"][split]),
        str(sampling.get("strategy", "uniform_windows")),
        sampling.get("windows_per_run"),
    )
    return TEPGraphDataset(subset, experiment["graph"], homogeneous=False)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _write_json(path: Path, payload: Mapping[str, Any], indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=int(indent))
    temporary.replace(path)


def _save_artifacts(path: Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {key: np.asarray(value) for key, value in arrays.items()}
    serializable["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
    np.savez_compressed(path, **serializable)


def _robustness(evaluations: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    clean = float(evaluations["clean"]["macro_f1"])
    result: dict[str, Any] = {
        "delta_f1": {
            name: delta_f1(clean, float(metrics["macro_f1"]))
            for name, metrics in evaluations.items()
            if name != "clean"
        },
        "by_corruption": {},
    }
    grouped: dict[str, list[tuple[float, float]]] = {}
    for name, metrics in evaluations.items():
        if "=" not in name or name.startswith("protocol_group="):
            continue
        corruption, raw_strength = name.split("=", 1)
        grouped.setdefault(corruption, []).append((float(raw_strength), float(metrics["macro_f1"])))
    for corruption, values in grouped.items():
        result["by_corruption"][corruption] = multi_strength_summary(
            [item[0] for item in values], [item[1] for item in values]
        )
    return result


def _experiment_and_data(stage5_path: Path, protocol: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    stage5 = _load_yaml(stage5_path)
    base_path = _resolve(stage5_path.parent, str(stage5["base_experiment_config"]))
    experiment = _load_yaml(base_path)
    experiment["_config_dir"] = str(base_path.parent)
    _deep_update(experiment, stage5["formal_overrides"])
    experiment["evaluation"] = {"metrics": deepcopy(stage5["metrics"])}
    data_path = _resolve(stage5_path.parent, str(stage5["data_config"]))
    data_config = load_data_config(data_path)
    active = str(stage5["protocols"][protocol]["data_protocol"])
    data_config["protocols"]["active"] = active
    return stage5, experiment, data_config


def _model_config(experiment: dict[str, Any], spec: Mapping[str, Any] | None) -> tuple[str, dict[str, Any]]:
    configured = deepcopy(experiment)
    if spec is None:
        raise ValueError("A model specification is required.")
    base_model = str(spec["base_model"])
    if base_model == "qc_hgt" and "switches" in spec:
        configured["models"]["qc_hgt"]["switches"].update(deepcopy(spec["switches"]))
    return base_model, configured


def run_one(
    stage5_path: Path,
    protocol: str,
    run_id: str,
    base_model: str,
    seed: int,
    experiment_type: str,
    model_spec: Mapping[str, Any] | None = None,
    sensitivity: Mapping[str, Any] | None = None,
) -> Path:
    stage5, experiment, data_config = _experiment_and_data(stage5_path, protocol)
    if model_spec is not None:
        base_model, experiment = _model_config(experiment, model_spec)
    if sensitivity:
        parameter = str(sensitivity["parameter"])
        value = sensitivity["value"]
        reference = stage5["sensitivity"]["fixed_reference"]
        data_config["window"]["length"] = int(
            value if parameter == "window_length" else reference["window_length"]
        )
        experiment["models"]["qc_hgt"]["loss_weights"]["quality"] = float(
            value if parameter == "lambda_q" else reference["lambda_q"]
        )
        experiment["models"]["qc_hgt"]["loss_weights"]["consistency"] = float(
            value if parameter == "lambda_c" else reference["lambda_c"]
        )

    results_root = _resolve(stage5_path.parent, str(stage5["output"]["results_directory"]))
    run_directory = results_root / protocol / experiment_type
    result_path = run_directory / f"{run_id}_{seed}.json"
    existing_result: dict[str, Any] | None = None
    if bool(stage5["run"]["resume"]) and result_path.exists():
        with result_path.open("r", encoding="utf-8") as handle:
            existing_result = json.load(handle)
        if bool(existing_result.get("complete")):
            print(f"[resume] {result_path}")
            return result_path

    allowed_seeds = data_config["experiment"]["seeds"]
    set_seed(
        seed,
        allowed_seeds,
        bool(experiment["runtime"]["deterministic"]),
        bool(experiment["runtime"]["deterministic_warn_only"]),
    )
    torch.set_num_threads(int(experiment["runtime"]["cpu_threads"]))
    device = _device(str(experiment["runtime"]["device"]))
    checkpoint = results_root / str(stage5["output"]["checkpoint_subdirectory"]) / protocol / experiment_type / f"{run_id}_{seed}.pt"
    log_path = results_root / str(stage5["output"]["log_subdirectory"]) / protocol / experiment_type / f"{run_id}_{seed}.log"
    logger = create_logger(f"stage5.{protocol}.{run_id}.{seed}", log_path, str(experiment["logging"]["level"]))

    datasets = build_tep_datasets_from_config(
        data_config, seed=seed, print_leakage_report=True
    )
    train_graph = _graph_dataset(datasets["train"], "train", experiment)
    validation_graph = _graph_dataset(datasets["val"], "val", experiment)
    variant = str(data_config["dataset"]["variant"])
    num_classes = int(data_config["dataset"]["variants"][variant]["num_classes"])
    model = build_model(base_model, 0, num_classes, experiment)
    # A checkpoint can be written before training finishes.  Only skip fit when
    # the seed-level result document exists, which is created after fit returns.
    if bool(stage5["run"]["resume"]) and existing_result is not None and checkpoint.exists():
        best = load_checkpoint(checkpoint, model, device)
        training_result = {"best_epoch": int(best["epoch"]), "best_validation": best["metrics"]}
        model.to(device)
        logger.info("resume_checkpoint=%s", checkpoint)
    else:
        training_result = fit(
            model,
            _loader(train_graph, "train", seed, experiment),
            _loader(validation_graph, "val", seed, experiment),
            device,
            num_classes,
            experiment,
            checkpoint,
            logger,
        )

    result: dict[str, Any] = existing_result or {
        "schema_version": "stage5.v1",
        "experiment_type": experiment_type,
        "protocol": protocol,
        "data_protocol": data_config["protocols"]["active"],
        "model": run_id,
        "base_model": base_model,
        "seed": int(seed),
        "dataset_variant": variant,
        "best_epoch": training_result["best_epoch"],
        "validation": training_result["best_validation"],
        "evaluations": {},
        "complete": False,
        "config": str(stage5_path),
    }
    if sensitivity:
        result["sensitivity_parameter"] = sensitivity["parameter"]
        result["sensitivity_value"] = sensitivity["value"]
    _write_json(result_path, result, int(stage5["output"]["json_indent"]))

    test_sets = build_tep_protocol_test_sets_from_config(data_config, seed)
    selected = None
    if experiment_type == "sensitivity":
        selected = {"clean", str(stage5["sensitivity"]["evaluation_scenario"])}
    for scenario, dataset in test_sets.items():
        if selected is not None and scenario not in selected:
            continue
        if bool(stage5["run"]["resume"]) and scenario in result["evaluations"]:
            logger.info("resume_scenario=%s", scenario)
            continue
        graph = _graph_dataset(dataset, "test", experiment)
        artifact_enabled = scenario in {str(value) for value in stage5["artifacts"]["scenarios"]}
        artifact_config = deepcopy(stage5["artifacts"])
        artifact_config["enabled"] = bool(artifact_config["enabled"]) and artifact_enabled
        metrics = evaluate(
            model,
            _loader(graph, "test", seed, experiment),
            device,
            num_classes,
            metric_config=stage5["metrics"],
            artifact_config=artifact_config,
        )
        arrays = metrics.pop("_artifacts", None)
        result["evaluations"][scenario] = metrics
        if arrays:
            artifact_path = (
                results_root
                / str(stage5["output"]["artifact_subdirectory"])
                / protocol
                / experiment_type
                / f"{run_id}_{seed}_{_safe_name(scenario)}.npz"
            )
            _save_artifacts(
                artifact_path,
                arrays,
                {"protocol": protocol, "model": run_id, "seed": seed, "scenario": scenario},
            )
        _write_json(result_path, result, int(stage5["output"]["json_indent"]))
        logger.info("scenario=%s macro_f1=%.6f", scenario, metrics["macro_f1"])
    result["robustness"] = _robustness(result["evaluations"])
    result["complete"] = True
    _write_json(result_path, result, int(stage5["output"]["json_indent"]))
    return result_path


def _run_matrix(stage5_path: Path, protocol: str, requested_model: str | None, requested_seed: int | None) -> None:
    stage5, _, data_config = _experiment_and_data(stage5_path, protocol)
    seeds = [int(requested_seed)] if requested_seed is not None else [int(v) for v in data_config["experiment"]["seeds"]]
    protocol_config = stage5["protocols"][protocol]
    main_models = [
        str(value)
        for value in protocol_config.get("main_models", stage5["models"]["main"])
    ]
    for model in main_models:
        if requested_model is not None and requested_model != model:
            continue
        for seed in seeds:
            run_one(stage5_path, protocol, model, model, seed, "main")
    if bool(stage5["run"]["include_ablations"]):
        selected_ablations = {
            str(value)
            for value in protocol_config.get(
                "ablation_models", stage5["models"]["ablations"].keys()
            )
        }
        for run_id, spec in stage5["models"]["ablations"].items():
            if run_id not in selected_ablations:
                continue
            if requested_model is not None and requested_model != run_id:
                continue
            for seed in seeds:
                run_one(
                    stage5_path, protocol, str(run_id), str(spec["base_model"]), seed, "ablation", spec
                )


def _run_sensitivity(stage5_path: Path, protocol: str, requested_seed: int | None) -> None:
    stage5, _, data_config = _experiment_and_data(stage5_path, protocol)
    if not bool(stage5["sensitivity"].get("enabled", True)):
        return
    if protocol != str(stage5["sensitivity"]["protocol"]):
        return
    seeds = [int(requested_seed)] if requested_seed is not None else [int(v) for v in data_config["experiment"]["seeds"]]
    for parameter in ("window_length", "lambda_q", "lambda_c"):
        for value in stage5["sensitivity"][parameter]:
            text_value = str(value).replace(".", "p")
            run_id = f"sensitivity_{parameter}_{text_value}"
            for seed in seeds:
                run_one(
                    stage5_path,
                    protocol,
                    run_id,
                    str(stage5["sensitivity"]["model"]),
                    seed,
                    "sensitivity",
                    sensitivity={"parameter": parameter, "value": value},
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "stage5.yaml")
    parser.add_argument("--protocol", required=True, choices=("protocol_1", "protocol_2"))
    parser.add_argument("--model")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--include-sensitivity", action="store_true")
    arguments = parser.parse_args()
    config_path = arguments.config.expanduser().resolve()
    _run_matrix(config_path, arguments.protocol, arguments.model, arguments.seed)
    stage5 = _load_yaml(config_path)
    configured = bool(stage5["run"]["include_sensitivity_in_protocol2"])
    if arguments.include_sensitivity or (configured and arguments.protocol == "protocol_2" and arguments.model is None):
        _run_sensitivity(config_path, arguments.protocol, arguments.seed)


if __name__ == "__main__":
    main()


# Design note (paper Sec. 5.1--5.7): every run trains once on a run-disjoint
# split, reuses that checkpoint across paired degradation tests, and writes an
# auditable seed-level JSON before five-seed aggregation.
