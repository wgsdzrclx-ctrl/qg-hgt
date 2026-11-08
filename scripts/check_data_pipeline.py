"""Visual and numerical self-check for the Stage-1 TEP data pipeline.

Usage:
    python scripts/check_data_pipeline.py --config configs/data.yaml
    python scripts/check_data_pipeline.py --config configs/data.yaml --check-dataset
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

import matplotlib.pyplot as plt
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.corruptions import (  # noqa: E402
    CorruptionResult,
    continuous_missing,
    gaussian_noise,
    random_missing,
    sensor_drift,
    time_shift,
)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def _synthetic_window(config: Mapping[str, Any]) -> torch.Tensor:
    check = config["self_check"]
    time_points = int(check["time_points"])
    width = int(config["dataset"]["num_variables"])
    cycles = float(check["sine_cycles"])
    slope = float(check["trend_slope"])
    phase = torch.linspace(0.0, cycles * 2.0 * torch.pi, time_points)
    base = torch.sin(phase) + slope * torch.arange(time_points, dtype=torch.float32)
    return base[:, None].repeat(1, width)


def _apply(
    name: str,
    window: torch.Tensor,
    variable: int,
    strength: float | int,
    config: Mapping[str, Any],
    seed: int,
) -> CorruptionResult:
    selected = [variable]
    quality = config["quality"]
    corruption = config["corruptions"][name]
    generator = _generator(seed)
    if name == "gaussian_noise":
        return gaussian_noise(window, selected, float(strength), quality, generator)
    if name == "drift":
        return sensor_drift(
            window, selected, float(strength), corruption["time_origin"], quality, generator
        )
    if name == "random_missing":
        return random_missing(
            window, selected, float(strength), float(corruption["fill_value"]), quality, generator
        )
    if name == "continuous_missing":
        return continuous_missing(
            window, selected, int(strength), float(corruption["fill_value"]), quality, generator
        )
    if name == "time_shift":
        return time_shift(window, selected, int(strength), str(corruption["padding"]), quality, generator)
    raise KeyError(name)


def _assert_nonincreasing(values: list[float], tolerance: float, name: str) -> None:
    if any(right > left + tolerance for left, right in zip(values, values[1:])):
        raise AssertionError(f"q* is not monotonically non-increasing for {name}: {values}")


def run_corruption_check(config_path: Path) -> Path:
    config = _load_config(config_path)
    check = config["self_check"]
    seed = int(check["seed"])
    if seed not in [int(value) for value in config["experiment"]["seeds"]]:
        raise ValueError("self_check.seed must be one of experiment.seeds.")
    torch.manual_seed(seed)
    window = _synthetic_window(config)
    variable = int(check["variable_index"])
    tolerance = float(check["monotonic_tolerance"])
    names = ["gaussian_noise", "drift", "random_missing", "continuous_missing", "time_shift"]
    figure_size = tuple(float(value) for value in check["figure_size"])
    figure, axes = plt.subplots(len(names), 1, figsize=figure_size, sharex=True)

    print("[Corruption q* monotonicity self-check]")
    for axis, name in zip(axes, names):
        levels = config["corruptions"][name]["levels"]
        q_values: list[float] = []
        strongest: CorruptionResult | None = None
        for level in levels:
            result = _apply(name, window, variable, level, config, seed)
            q_values.append(float(result.q_star[variable]))
            strongest = result
        _assert_nonincreasing(q_values, tolerance, name)
        print(f"  {name:>18}: levels={levels}, q*={[round(value, 6) for value in q_values]}  PASS")
        assert strongest is not None
        time = torch.arange(window.shape[0])
        line_width = float(check["line_width"])
        axis.plot(time, window[:, variable], label="clean", linewidth=line_width)
        axis.plot(time, strongest.x[:, variable], label=f"strongest={levels[-1]}", linewidth=line_width)
        missing = ~strongest.observed_mask[:, variable]
        if bool(missing.any()):
            axis.scatter(time[missing], strongest.x[missing, variable], marker="x", label="masked")
        axis.set_ylabel(name)
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    axes[-1].set_xlabel("time in window")
    figure.suptitle("TEP controlled corruption self-check")
    figure.tight_layout()

    output = Path(str(check["output_figure"]))
    if not output.is_absolute():
        output = (config_path.parent / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=int(check["dpi"]), bbox_inches="tight")
    plt.close(figure)
    print(f"Figure written to: {output}")
    return output


def run_dataset_check(config_path: Path, check_all_variants: bool = False) -> None:
    """Exercise real files, split invariants, labels, PyG fields, and fixed grids."""
    from data.tep_dataset import (
        build_tep_datasets_from_config,
        build_tep_evaluation_grid,
        load_data_config,
    )

    base_config = load_data_config(config_path)
    active_variant = str(base_config["dataset"]["variant"])
    variants = (
        list(base_config["dataset"]["variants"])
        if check_all_variants
        else [active_variant]
    )
    seed = int(base_config["self_check"]["seed"])
    for variant in variants:
        config = copy.deepcopy(base_config)
        config["dataset"]["variant"] = variant
        datasets = build_tep_datasets_from_config(config, seed, print_leakage_report=True)
        for split, dataset in datasets.items():
            if not dataset.runs or not len(dataset):
                raise AssertionError(f"variant={variant}, split={split} is empty")
            first = dataset[0]
            expected_shape = (
                int(config["dataset"]["num_variables"]),
                int(config["window"]["length"]),
            )
            if tuple(first.x.shape) != expected_shape:
                raise AssertionError(
                    f"variant={variant}, split={split}: {tuple(first.x.shape)} != {expected_shape}"
                )
            if tuple(first.observed_mask.shape) != expected_shape:
                raise AssertionError("Observation mask and node window shapes differ.")
            run_index = int(first.run_index.item())
            end = int(first.window_end.item()) + 1
            _, labels = dataset._run_tensors(run_index)
            if int(first.y.item()) != int(labels[end - 1].item()):
                raise AssertionError("Window label is not the final time-step label.")
            expected_q = bool(config["quality"]["attach_target"][split])
            if hasattr(first, "q_star") != expected_q:
                raise AssertionError(f"Unexpected q_star visibility on split={split}.")
            repeated = dataset[0]
            if not torch.equal(first.x, repeated.x):
                raise AssertionError("Per-window corruption is not deterministic for a fixed seed/epoch.")
        print(f"[Dataset smoke-check] variant={variant}: PASS")

    if active_variant == str(base_config["dataset"]["paper_main_variant"]):
        grid = build_tep_evaluation_grid(config_path, seed)
        expected_count = 1 + sum(
            len(levels)
            for levels in base_config["evaluation"]["fixed_strength_grid"]["levels"].values()
        )
        if len(grid) != expected_count or len({len(dataset) for dataset in grid.values()}) != 1:
            raise AssertionError("Fixed-strength grid is incomplete or its datasets are not aligned.")
        clean = grid["clean"][0]
        for corruption_name, levels in base_config["evaluation"]["fixed_strength_grid"]["levels"].items():
            selection_masks = []
            for level in levels:
                sample = grid[f"{corruption_name}={level}"][0]
                if hasattr(sample, "q_star"):
                    raise AssertionError("q_star leaked into a fixed-strength test sample.")
                selection_masks.append((sample.x != clean.x).any(dim=1))
            if not all(torch.equal(selection_masks[0], mask) for mask in selection_masks[1:]):
                raise AssertionError(
                    f"Target variables changed across fixed strengths for {corruption_name}."
                )
        print(f"[Fixed-strength evaluation grid] datasets={len(grid)}: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "data.yaml")
    parser.add_argument(
        "--check-dataset",
        action="store_true",
        help="Also load the configured TEP files and print the run-leakage report.",
    )
    parser.add_argument(
        "--check-all-variants",
        action="store_true",
        help="Exercise both the paper-main and compatibility dataset variants.",
    )
    arguments = parser.parse_args()
    config_path = arguments.config.expanduser().resolve()
    run_corruption_check(config_path)
    if arguments.check_dataset or arguments.check_all_variants:
        run_dataset_check(config_path, check_all_variants=arguments.check_all_variants)


if __name__ == "__main__":
    main()


# Design note (paper Sec. 4.1, 4.4, 5.1): plots check controlled operators and
# q* monotonicity; dataset checks enforce run-first splits, last-step labels,
# fixed-strength pairing, deterministic seeds, and inference-time q* hiding.
