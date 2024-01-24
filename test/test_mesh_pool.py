'''
'''

import pytest
import torch

import matplotlib.pyplot as plt

from torch_quadconv import Mesh_MaxPool, MeshHandler


def test_forward():

    input_weights = torch.ones(25)

    input_nodes = torch.sort(torch.randn(25, 2),stable=True, dim=0)[0]

    Mesh = MeshHandler(input_nodes, input_weights, quad_map= 'mfnus')

    Mesh.cache([25, 16])

    MMP = Mesh_MaxPool(adjoint=False)

    input = torch.arange(25).reshape(1,1,25).float()

    output = MMP.forward(Mesh, input)

    plt.scatter(input_nodes[:, 0], input_nodes[:, 1], s=input*50, marker='o')
    plt.scatter(Mesh.output_points[:, 0], Mesh.output_points[:, 1], s=output*50, marker='x')
    plt.show()

    print(input_nodes); print(input)

    print('\n')

    print(Mesh.output_points); print(output)

    print('\n')

    print(Mesh.get_downsample_map(Mesh.output_points.shape[0]).copy())

    return