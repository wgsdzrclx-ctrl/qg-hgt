"""HGT + observable learnable-quality variant without consistency."""

from __future__ import annotations

from typing import Any, Mapping

from torch import Tensor, nn

from .common import heterodata_from_inputs
from .qc_hgt import QCHGT


class HGTLearnableQuality(nn.Module):
    def __init__(self, num_classes: int, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.model = QCHGT(num_classes, config)

    def forward(self, x_dict: Mapping[str, Any], edge_index_dict: Mapping[tuple[str, str, str], Tensor]) -> dict[str, Any]:
        return self.model(heterodata_from_inputs(x_dict, edge_index_dict))


# Design note (paper Sec. 4.5 and 4.7): MLP_q learns r_j from observable a_j;
# lambda_r log(r_j+epsilon) adjusts attention, while c_ij and L_c are disabled.
