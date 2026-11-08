"""Structural and numerical self-check for the Stage-2 process graph."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import yaml
from torch_geometric.data import Data, HeteroData


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.tep_dataset import build_tep_datasets  # noqa: E402
from data.tep_hetero_graph import (  # noqa: E402
    EDGE_TYPES,
    PROCESS_UNITS,
    XMEAS_TO_UNIT,
    XMV_TO_UNIT,
    build_tep_hetero_graph,
    validate_tep_mapping,
    window_statistics,
)


def run(config_path: Path, check_dataset: bool) -> None:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data_config_path = (config_path.parent / config["data"]["config_path"]).resolve()
    with data_config_path.open("r", encoding="utf-8") as handle:
        data_config = yaml.safe_load(handle)
    validate_tep_mapping()
    time_points = int(data_config["window"]["length"])
    width = int(data_config["dataset"]["num_variables"])
    feature_count = len(config["graph"]["statistics"]["feature_names"])
    time = torch.arange(time_points, dtype=torch.float32)
    window = torch.stack([(index + 1) * time + index for index in range(width)])
    mask = torch.ones_like(window, dtype=torch.bool)
    statistics = window_statistics(window, mask, config["graph"]["statistics"])
    if not torch.allclose(statistics[:, 4], torch.arange(1, width + 1, dtype=torch.float32)):
        raise AssertionError("Linear-trend feature is incorrect.")
    sample = Data(x=window, observed_mask=mask, y=torch.tensor([0]))
    hetero = build_tep_hetero_graph(sample, config["graph"], homogeneous=False)
    homogeneous = build_tep_hetero_graph(sample, config["graph"], homogeneous=True)
    assert isinstance(hetero, HeteroData)
    assert len(XMEAS_TO_UNIT) == 41 and len(XMV_TO_UNIT) == 11
    assert hetero["sensor"].x.shape == (41, feature_count)
    assert hetero["actuator"].x.shape == (11, feature_count)
    assert hetero["process_unit"].x.shape == (len(PROCESS_UNITS), feature_count)
    assert set(hetero.edge_types) == set(EDGE_TYPES)
    assert homogeneous.x.shape == (58, feature_count)
    assert homogeneous.edge_index.shape == (2, 480)
    assert not hasattr(hetero["sensor"], "quality_target")
    print("[Stage-2 synthetic graph] PASS")

    if check_dataset:
        datasets = build_tep_datasets(
            data_config_path,
            seed=int(data_config["self_check"]["seed"]),
            test_scenario=str(config["data"]["test_scenario"]),
            print_leakage_report=True,
        )
        train_graph = build_tep_hetero_graph(datasets["train"][0], config["graph"])
        test_graph = build_tep_hetero_graph(datasets["test"][0], config["graph"])
        assert hasattr(train_graph["sensor"], "quality_target")
        assert not hasattr(test_graph["sensor"], "quality_target")
        print("[Stage-2 real-data graph and q* visibility] PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "stage2.yaml")
    parser.add_argument("--check-dataset", action="store_true")
    arguments = parser.parse_args()
    run(arguments.config.expanduser().resolve(), arguments.check_dataset)


if __name__ == "__main__":
    main()


# Design note (paper Sec. 4.2-4.4): this check validates statistics, node/edge
# cardinalities, homogeneous conversion, and training-only q* visibility.
