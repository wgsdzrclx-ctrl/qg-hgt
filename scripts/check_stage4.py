"""Forward/backward and leakage-guard checks for every Stage-4 baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.tep_dataset import build_tep_datasets  # noqa: E402
from data.tep_hetero_graph import TEPGraphDataset  # noqa: E402
from models import build_model  # noqa: E402
from models.common import extended_x_dict  # noqa: E402


def _forward(model: torch.nn.Module, batch: object) -> tuple[torch.Tensor, torch.Tensor]:
    output = model(extended_x_dict(batch), batch.edge_index_dict)
    if isinstance(output, dict):
        return output["logits"], output["loss"]
    return output, F.cross_entropy(output, batch.y.reshape(-1).long())


def run(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as handle:
        experiment = yaml.safe_load(handle)
    data_path = (config_path.parent / experiment["data"]["config_path"]).resolve()
    with data_path.open("r", encoding="utf-8") as handle:
        data_config = yaml.safe_load(handle)
    seed = int(data_config["self_check"]["seed"])
    datasets = build_tep_datasets(
        data_path, seed=seed, test_scenario=str(experiment["data"]["test_scenario"]),
        print_leakage_report=True,
    )
    train_graphs = TEPGraphDataset(datasets["train"], experiment["graph"], homogeneous=False)
    train_batch = next(iter(DataLoader(train_graphs, batch_size=2, shuffle=False)))
    test_batch = next(iter(DataLoader(
        TEPGraphDataset(datasets["test"], experiment["graph"], homogeneous=False),
        batch_size=1, shuffle=False,
    )))
    variant = data_config["dataset"]["variant"]
    num_classes = int(data_config["dataset"]["variants"][variant]["num_classes"])
    names = (
        "mlp", "cnn", "tcn", "lstm", "gcn", "sage", "gat", "rgcn", "hgt",
        "hgt_learnable_quality", "hgt_consistency", "qc_hgt", "hgt_fixed_quality",
    )
    for name in names:
        model = build_model(name, 52 * 7, num_classes, experiment)
        logits, loss = _forward(model, train_batch)
        assert logits.shape == (2, num_classes), (name, logits.shape)
        assert torch.isfinite(logits).all() and torch.isfinite(loss), name
        loss.backward()
        assert any(parameter.grad is not None for parameter in model.parameters()), name
        if name != "hgt_fixed_quality":
            with torch.no_grad():
                test_logits, _ = _forward(model.eval(), test_batch)
            assert test_logits.shape == (1, num_classes), name

    fixed = build_model("hgt_fixed_quality", 52 * 7, num_classes, experiment)
    try:
        _forward(fixed.eval(), test_batch)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Fixed Quality bypassed the inference-time q_star guard.")
    print("[Stage-4 all baselines forward/backward and Fixed-Quality guard] PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "stage2.yaml")
    arguments = parser.parse_args()
    run(arguments.config.expanduser().resolve())


if __name__ == "__main__":
    main()


# Design note (paper Sec. 5.3): all baselines are checked on the same two graph
# windows and class space; Fixed Quality must pass on labelled training data and
# fail on held-out inference data that intentionally omit q_star.
