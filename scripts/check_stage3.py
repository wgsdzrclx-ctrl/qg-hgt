"""Forward/backward and ablation self-checks for QC-HGT."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import torch
from torch_geometric.loader import DataLoader
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.tep_dataset import build_tep_datasets  # noqa: E402
from data.tep_hetero_graph import TEPGraphDataset  # noqa: E402
from models.qc_hgt import QCHGT  # noqa: E402


def _model_config(base: dict, **switch_updates: bool) -> dict:
    config = copy.deepcopy(base)
    config["switches"].update(switch_updates)
    return config


def run(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as handle:
        experiment = yaml.safe_load(handle)
    data_path = (config_path.parent / experiment["data"]["config_path"]).resolve()
    with data_path.open("r", encoding="utf-8") as handle:
        data_config = yaml.safe_load(handle)
    seed = int(data_config["self_check"]["seed"])
    datasets = build_tep_datasets(
        data_path,
        seed=seed,
        test_scenario=str(experiment["data"]["test_scenario"]),
        print_leakage_report=True,
    )
    train_graphs = TEPGraphDataset(datasets["train"], experiment["graph"])
    train_batch = next(iter(DataLoader(train_graphs, batch_size=2, shuffle=False)))
    test_graph = TEPGraphDataset(datasets["test"], experiment["graph"])[0]
    base = experiment["models"]["qc_hgt"]
    num_classes = int(
        data_config["dataset"]["variants"][data_config["dataset"]["variant"]]["num_classes"]
    )

    model = QCHGT(num_classes, copy.deepcopy(base))
    output = model(train_batch)
    assert output["logits"].shape == (2, num_classes)
    assert output["quality_features"].shape == (82, 6)
    assert torch.allclose(
        output["probabilities"].sum(dim=1), torch.ones(2), atol=1.0e-6
    )
    assert all(
        torch.isfinite(output[name])
        for name in ("loss", "classification_loss", "quality_loss", "consistency_loss")
    )
    output["loss"].backward()
    if not any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("QC-HGT backward pass produced no gradients.")

    model.eval()
    with torch.no_grad():
        learned_test = model(test_graph)
    assert learned_test["logits"].shape == (1, num_classes)
    assert not hasattr(test_graph["sensor"], "quality_target")

    variants = {
        "without_quality": _model_config(
            base, quality_attention=False, quality_loss=False
        ),
        "without_consistency": _model_config(
            base, consistency_attention=False, consistency_loss=False
        ),
        "product_fusion": _model_config(base, product_fusion=True),
        "without_quality_loss": _model_config(base, quality_loss=False),
        "without_consistency_loss": _model_config(base, consistency_loss=False),
        "fixed_observable_quality": _model_config(
            base, fixed_observable_quality=True, fixed_quality=False, quality_loss=False
        ),
    }
    for name, variant in variants.items():
        result = QCHGT(num_classes, variant)(train_batch)
        if not torch.isfinite(result["loss"]):
            raise AssertionError(f"Non-finite loss for ablation: {name}")
        if name == "fixed_observable_quality":
            deployable = QCHGT(num_classes, variant)(test_graph)
            if not torch.isfinite(deployable["reliability"]).all():
                raise AssertionError("Fixed observable quality failed without q_star.")

    fixed_config = _model_config(base, fixed_quality=True)
    fixed_model = QCHGT(num_classes, fixed_config)
    fixed_output = fixed_model(train_batch)
    if not torch.allclose(
        fixed_output["reliability"], train_batch["sensor"].quality_target.reshape(-1)
    ):
        raise AssertionError("Fixed Quality did not use the configured q_star target.")
    try:
        fixed_model(test_graph)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Fixed Quality bypassed the inference-time q_star guard.")
    print("[Stage-3 QC-HGT forward/backward and ablations] PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "stage2.yaml")
    arguments = parser.parse_args()
    run(arguments.config.expanduser().resolve())


if __name__ == "__main__":
    main()


# Design note (paper Sec. 4.5--4.8): this check covers observable quality,
# positive/negative consistency, collaborative propagation, joint loss, every
# requested ablation, and the inference-time prohibition on q_star.
