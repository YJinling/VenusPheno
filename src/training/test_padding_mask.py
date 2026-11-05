"""PBS smoke check: padded and unpadded ISAB/PMA inference must agree."""
import torch

from models.aggregator import Aggregator


def main():
    torch.manual_seed(7)
    model = Aggregator(8, mode="isab_pma", num_heads=2, num_inds=4, num_seeds=2).eval()
    short = torch.randn(3, 8)
    long = torch.randn(6, 8)
    padded = torch.zeros(2, 6, 8)
    padded[0, :3], padded[1] = short, long
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    with torch.no_grad():
        batch = model(padded, mask)
        alone = model(short.unsqueeze(0), torch.ones(1, 3, dtype=torch.bool))
        _, attn = model(padded, mask, return_attn=True)
    assert torch.allclose(batch[0], alone[0], atol=1e-6, rtol=1e-5)
    assert torch.count_nonzero(attn["pma"][0, :, :, 3:]) == 0
    print("padding-mask smoke check passed")


if __name__ == "__main__":
    main()
