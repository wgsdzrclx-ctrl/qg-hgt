"""Numerical checks for memory-bounded consistency scoring."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

import torch
import yaml

from models.qc_hgt import QCHGTLayer


class ConsistencyChunkingTests(unittest.TestCase):
    def test_chunked_logits_match_direct_logits_and_gradients(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "stage2.yaml"
        with config_path.open("r", encoding="utf-8") as handle:
            config = deepcopy(yaml.safe_load(handle)["models"]["qc_hgt"])
        config["dropout"] = 0.0
        config["consistency"]["dropout"] = 0.0
        config["consistency"]["memory"]["logit_chunk_size"] = 3
        config["consistency"]["memory"]["gradient_checkpointing"] = True
        layer = QCHGTLayer(config).train()

        generator = torch.Generator().manual_seed(0)
        source = torch.randn(7, config["hidden_dim"], generator=generator, requires_grad=True)
        targets = torch.randn(11, config["hidden_dim"], generator=generator, requires_grad=True)
        indices = torch.tensor([0, 4, 8, 2, 9, 3, 6])
        relation = torch.randn(7, config["relation_dim"], generator=generator, requires_grad=True)

        direct = layer._consistency_logits(source, targets[indices], relation)
        chunked = layer._indexed_consistency_logits(source, targets, indices, relation)
        self.assertTrue(torch.allclose(chunked, direct, atol=1.0e-7, rtol=1.0e-6))

        direct_gradients = torch.autograd.grad(direct.sum(), (source, targets, relation), retain_graph=True)
        chunked_gradients = torch.autograd.grad(chunked.sum(), (source, targets, relation))
        for chunked_gradient, direct_gradient in zip(chunked_gradients, direct_gradients):
            self.assertTrue(
                torch.allclose(chunked_gradient, direct_gradient, atol=1.0e-7, rtol=1.0e-6)
            )


if __name__ == "__main__":
    unittest.main()


# Design note (paper Sec. 4.6): chunking changes only execution memory;
# logits, gradients, negative sampling, and the consistency loss are unchanged.
