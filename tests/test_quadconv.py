'''
'''

import pytest
import torch

from torch_quadconv import QuadConv, Grid

'''
Would be better to combine these in the future.
'''

def test_forward():

    test_grid = Grid((20,20))

    test_data = torch.randn(8,16,20*20)

    QC = QuadConv(domain= test_grid, range=Grid((5,5)), in_channels=16, out_channels=32)

    output = QC.forward(test_data)

    assert output.shape == torch.Size([8,32,25])

    return

def test_forward_cuda():

    test_grid = Grid((20,20)).cuda()

    test_data = torch.randn(8,16,20*20).cuda()

    QC = QuadConv(domain= test_grid, range=Grid((5,5)), in_channels=16, out_channels=32).cuda()

    output = QC.forward(test_data)

    assert output.shape == torch.Size([8,32,25])

    return

def test_forward_output_same():

    test_grid = Grid((20,20))

    test_data = torch.randn(8,2,20*20)

    QC = QuadConv(domain= test_grid, range=Grid((5,5)), in_channels=2, out_channels=4, output_same=True)

    output = QC.forward(test_data)

    assert output.shape == torch.Size([8,4,400])

    return
