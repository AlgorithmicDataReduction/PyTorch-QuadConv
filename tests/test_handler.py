'''
'''

import pytest
import torch

from torch_quadconv import MeshHandler

def test_init_simple():
    input_nodes = torch.randn(10, 2)

    Mesh = MeshHandler(input_nodes)

    assert Mesh._spatial_dim == 2
    assert Mesh._weights[0].requires_grad == True

    return

def test_init_nodes_weights():
    input_nodes = torch.randn(10, 2)
    input_weights = torch.randn(10)

    Mesh = MeshHandler(input_nodes, input_weights)

    assert Mesh._weights[0].requires_grad == False

    return

def test_step():
    input_nodes = torch.randn(16, 2)

    Mesh = MeshHandler(input_nodes)
    Mesh.cache([16, 9], mirror=True)

    Mesh.step()

    assert Mesh._current_index == 1

    Mesh.step()

    assert Mesh._current_index == 0

def test_index():
    input_nodes = torch.randn(16, 2)

    Mesh = MeshHandler(input_nodes)
    Mesh.cache([16, 9], mirror=True)

    assert Mesh._radix == 1

    Mesh.step()

    assert Mesh._get_index() == 1
    assert Mesh._get_index(1) == 0

def test_cache():
    input_nodes = torch.randn(25, 2)
    input_weights = torch.randn(25)

    Mesh = MeshHandler(input_nodes, input_weights)

    node_seq = [25, 16, 4]

    Mesh.cache(node_seq, mirror=True)

    assert Mesh._num_meshes == 3
    assert Mesh._num_stages == 4

    for i in range(3):
        assert Mesh._weights[i].requires_grad == False
        assert Mesh._weights[i].shape[0] == node_seq[i]
