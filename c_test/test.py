
import torch
from core.torch_quadconv.utils.agglomeration import agglomerate

if __name__=="__main__":
    input = torch.randn(10, 2)

    output = agglomerate(input, None, 5)

    print(input[:5, :])
    print(output[:, :])

    assert torch.allclose(input[:5,:], output)
