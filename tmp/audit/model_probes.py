"""Read-only numerical probes of the current QC-HGT information bottlenecks.

Run from the repository root. No model, data, or checkpoint is modified.
"""
from copy import deepcopy
import json
from pathlib import Path

import torch
import yaml
from torch_geometric.data import Data

from data.tep_hetero_graph import _process_edges, build_tep_hetero_graph, window_statistics
from models.qc_hgt import QCHGT


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    with open("configs/stage2.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    raw = torch.tensor([10.0, 100.0])[:, None].repeat(1, 30)
    statistics = window_statistics(raw, torch.ones_like(raw, dtype=torch.bool), config["graph"]["statistics"])
    normalized = torch.nn.LayerNorm(7)(statistics)
    report = {
        "constant_levels_10_vs_100_layer_norm_max_absolute_difference": float(
            (normalized[0] - normalized[1]).abs().max().detach()
        )
    }
    raw = torch.randn(52, 30) + torch.arange(52)[:, None] + 10
    sample = Data(
        x=raw,
        observed_mask=torch.ones_like(raw, dtype=torch.bool),
        y=torch.tensor([1]),
        window_start=torch.tensor([0]),
        window_end=torch.tensor([29]),
        fault_scenario=torch.tensor([1]),
    )
    original = build_tep_hetero_graph(sample, config["graph"])
    swapped_sample = sample.clone()
    swapped_sample.x[[0, 1]] = swapped_sample.x[[1, 0]]
    swapped = build_tep_hetero_graph(swapped_sample, config["graph"])
    model = QCHGT(21, deepcopy(config["models"]["qc_hgt"])).eval()
    with torch.no_grad():
        original_output = model(original)
        swapped_output = model(swapped)
    report["swap_XMEAS1_XMEAS2_logits_max_absolute_difference"] = float(
        (original_output["logits"] - swapped_output["logits"]).abs().max()
    )
    report["timestamp_delay_all_zero_without_observed_timestamps"] = bool(
        (original["sensor"].timestamp_delay == 0).all()
    )
    counts = {"sensor": 41, "actuator": 11, "process_unit": 6}
    report["relations"] = []
    for edge_type, edge_index in _process_edges().items():
        degrees = torch.bincount(edge_index[1], minlength=counts[edge_type[2]])
        source_degrees = torch.bincount(edge_index[0], minlength=counts[edge_type[0]])[edge_index[0]]
        # Uniform non-neighbor selection from a batch of 128 disjoint identical graphs.
        probability_cross_graph = (counts[edge_type[2]] * 127) / (
            counts[edge_type[2]] * 128 - source_degrees.float()
        )
        report["relations"].append({
            "edge_type": list(edge_type),
            "edges_per_graph": int(edge_index.shape[1]),
            "single_incoming_targets": int((degrees == 1).sum()),
            "active_targets": int((degrees > 0).sum()),
            "wrong_negative_expected_cross_graph_fraction_batch128": float(probability_cross_graph.mean()),
        })
    output = Path("tmp/audit/model_probe_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
