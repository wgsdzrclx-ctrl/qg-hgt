"""Run the configured five seeds and write mean +/- standard-deviation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.tep_dataset import load_data_config  # noqa: E402
from main import run_experiment  # noqa: E402
from train import MODEL_CHOICES  # noqa: E402


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def run_all(model: str, config_path: Path) -> dict[str, Any]:
    experiment_path = config_path.expanduser().resolve()
    with experiment_path.open("r", encoding="utf-8") as handle:
        experiment = yaml.safe_load(handle)
    data_path = _resolve(experiment_path.parent, str(experiment["data"]["config_path"]))
    data_config = load_data_config(data_path)
    seeds = [int(value) for value in data_config["experiment"]["seeds"]]
    if seeds != [0, 1, 2, 3, 4]:
        raise ValueError("The five-seed runner requires exactly seeds 0,1,2,3,4.")
    results = [run_experiment(model, experiment_path, seed) for seed in seeds]
    ddof = int(experiment["reporting"]["std_ddof"])
    if ddof not in {0, 1}:
        raise ValueError("reporting.std_ddof must be 0 or 1.")
    spread = statistics.pstdev if ddof == 0 else statistics.stdev
    summary: dict[str, Any] = {"model": model, "seeds": seeds, "std_ddof": ddof}
    for metric in ("accuracy", "macro_f1"):
        values = [float(result[metric]) for result in results]
        summary[metric] = {
            "values": values,
            "mean": statistics.fmean(values),
            "std": spread(values),
            "mean_plus_minus_std": f"{statistics.fmean(values):.6f} +/- {spread(values):.6f}",
        }
    output_dir = _resolve(experiment_path.parent, str(experiment["output"]["directory"]))
    destination = output_dir / f"{model}_summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=int(experiment["output"]["json_indent"]))
    print(json.dumps(summary, ensure_ascii=False, indent=int(experiment["output"]["json_indent"])))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES)
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args()
    run_all(arguments.model, arguments.config)


if __name__ == "__main__":
    main()


# Design note (paper Sec. 5.1): this runner enforces seeds 0--4 and reports the
# sample standard deviation configured in YAML; individual seed JSON files are
# retained for auditability.
