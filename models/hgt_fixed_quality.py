"""Oracle HGT + Fixed Quality diagnostic variant."""

from __future__ import annotations

from typing import Any, Mapping

from torch import Tensor, nn

from .common import heterodata_from_inputs
from .qc_hgt import QCHGT


class HGTFixedQuality(nn.Module):
    def __init__(self, num_classes: int, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.model = QCHGT(num_classes, config)

    def forward(self, x_dict: Mapping[str, Any], edge_index_dict: Mapping[tuple[str, str, str], Tensor]) -> dict[str, Any]:
        return self.model(heterodata_from_inputs(x_dict, edge_index_dict))


# Design note (paper Sec. 4.7 ablation): r_j is replaced by q*_j in the log-r
# attention term. The QC-HGT guard deliberately rejects normal test inference,
# because q* derives from injected degradation parameters and is oracle-only.
