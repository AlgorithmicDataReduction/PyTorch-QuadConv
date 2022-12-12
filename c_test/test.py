
import torch
from core.torch_quadconv.utils.agglomeration import agglomerate

if __name__=="__main__":
    input = torch.randn(10, 2)

    output = agglomerate(torch, None, 5)

    assert torch.allclose(input[:5], output)
