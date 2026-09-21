"""Small structural tests for the attention-TCN implementation."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch


MODULE_PATH = Path(__file__).with_name("09_attention_tcn_loso.py")
SPEC = importlib.util.spec_from_file_location("attention_tcn_loso", MODULE_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class AttentionTCNTests(unittest.TestCase):
    def test_contexts_never_cross_subject_boundaries(self) -> None:
        groups = np.array(["a", "a", "a", "b", "b"])
        contexts = module.build_contexts(groups, window=3)
        self.assertEqual(
            contexts.tolist(),
            [[-1, -1, 0], [-1, 0, 1], [0, 1, 2], [-1, -1, 3], [-1, 3, 4]],
        )

    def test_model_returns_interval_logits_and_normalized_attention(self) -> None:
        model = module.AttentionTCN(24, 15, channels=16, levels=2, kernel_size=3, dropout=0.0)
        features = torch.randn(4, 8, 24)
        mask = torch.ones(4, 8, dtype=torch.bool)
        mask[0, :3] = False
        logits, attention = model(features, mask)
        self.assertEqual(logits.shape, (4, 15))
        self.assertEqual(attention.shape, (4, 8))
        self.assertTrue(torch.allclose(attention.sum(dim=1), torch.ones(4), atol=1e-6))
        self.assertTrue(bool(torch.all(attention[0, :3] == 0)))


if __name__ == "__main__":
    unittest.main()
