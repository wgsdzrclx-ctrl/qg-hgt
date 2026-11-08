"""Stage-4 baseline registry with one uniform forward contract."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from torch import nn

from .cnn import CNN1D
from .gat import GAT
from .gcn import GCN
from .hgt import HGT
from .hgt_consistency import HGTConsistency
from .hgt_fixed_quality import HGTFixedQuality
from .hgt_learnable_quality import HGTLearnableQuality
from .lstm import LSTM
from .mlp import MLP
from .rgcn import RGCN
from .qc_hgt_full import QCHGTModel
from .sage import GraphSAGE
from .tcn import TCN


MODEL_CLASSES: dict[str, type[nn.Module]] = {
    "mlp": MLP,
    "cnn": CNN1D,
    "tcn": TCN,
    "lstm": LSTM,
    "gcn": GCN,
    "sage": GraphSAGE,
    "gat": GAT,
    "rgcn": RGCN,
    "hgt": HGT,
    "hgt_fixed_quality": HGTFixedQuality,
    "hgt_learnable_quality": HGTLearnableQuality,
    "hgt_consistency": HGTConsistency,
    "qc_hgt": QCHGTModel,
}


def build_model(name: str, input_dim: int, num_classes: int, config: Mapping[str, Any]) -> nn.Module:
    """Build any Stage-4 model; input_dim is retained for CLI compatibility."""
    del input_dim
    if name not in MODEL_CLASSES:
        raise KeyError(f"Unknown model: {name}")
    if name == "qc_hgt":
        model_config = deepcopy(dict(config["models"]["qc_hgt"]))
    elif name.startswith("hgt_"):
        model_config = deepcopy(dict(config["models"]["qc_hgt"]))
        model_config["switches"] = deepcopy(dict(model_config["switches"]))
        model_config["switches"].update(config["models"][name]["switches"])
    else:
        model_config = config["models"][name]
    return MODEL_CLASSES[name](num_classes, model_config)


# Design note (paper Sec. 5.3): one registry guarantees every comparator uses
# forward(x_dict, edge_index_dict), the same class count and YAML-only settings.
