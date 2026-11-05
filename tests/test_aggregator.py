import unittest

import torch

from models.aggregator import Aggregator


class AggregatorTest(unittest.TestCase):
    def test_isab_pma_ignores_padded_tokens(self):
        torch.manual_seed(7)
        model = Aggregator(8, mode="isab_pma", num_heads=2, num_inds=4, num_seeds=2)
        model.eval()
        tokens = torch.randn(2, 5, 8)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
        perturbed = tokens.clone()
        perturbed[~mask] = 1_000.0
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(tokens, mask), model(perturbed, mask), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
