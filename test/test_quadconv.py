'''
'''

import pytest
import torch

from torch_quadconv import QuadConv, MeshHandler

'''
Would be better to combine these in the future.
'''

def test_forward():
    input_nodes, input_weights = torch.randn(25, 2), torch.randn(25)

    Mesh = MeshHandler(input_nodes, input_weights)

    Mesh.cache([25, 16])

    QC = QuadConv(spatial_dim = 2,
                    in_points = 25,
                    out_points = 16,
                    in_channels = 1,
                    out_channels = 2,
                    filter_seq = [4,4])

    input = torch.randn(1, 1, 25)

    output = QC.forward(Mesh, input)

    assert output.shape == torch.Size([1,2,16])
    assert QC.cached == True

    return

def test_forward_cuda():
    input_nodes, input_weights = torch.randn(25, 2), torch.randn(25)

    Mesh = MeshHandler(input_nodes, input_weights)

    Mesh = Mesh.cache([25, 16]).cuda()

    QC = QuadConv(spatial_dim = 2,
                    in_points = 25,
                    out_points = 16,
                    in_channels = 1,
                    out_channels = 2,
                    filter_seq = [4,4]).cuda()

    input = torch.randn(1, 1, 25).cuda()

    output = QC.forward(Mesh, input)

    return

def test_forward_output_same():
    input_nodes, input_weights = torch.randn(25, 2), torch.randn(25)

    Mesh = MeshHandler(input_nodes, input_weights)

    Mesh = Mesh.cache([25, 16]).cuda()

    QC = QuadConv(spatial_dim = 2,
                    in_points = 25,
                    out_points = 25,
                    in_channels = 1,
                    out_channels = 2,
                    filter_seq = [4,4],
                    output_same = True).cuda()

    input = torch.randn(1, 1, 25).cuda()

    output = QC.forward(Mesh, input)

    return
