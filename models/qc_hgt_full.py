"""Uniform Stage-4 adapter for the complete QC-HGT model."""

from __future__ import annotations

from typing import Any, Mapping

from torch import Tensor, nn

from .common import heterodata_from_inputs
from .qc_hgt import QCHGT


class QCHGTModel(nn.Module):
    def __init__(self, num_classes: int, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.model = QCHGT(num_classes, config)

    def forward(
        self,
        x_dict: Mapping[str, Any],
        edge_index_dict: Mapping[tuple[str, str, str], Tensor],
    ) -> dict[str, Any]:
        return self.model(heterodata_from_inputs(x_dict, edge_index_dict))


# Design note (paper Sec. 4.5--4.8): this adapter exposes the complete quality
# and consistency model through the same interface used by every comparator.
