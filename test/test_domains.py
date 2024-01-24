from torch_quadconv import Grid, Mesh
import torch
import pytest



def test_grid_downsample():

    test_grid = Grid((20,20))

    output = test_grid.downsample(factor=2)

    assert output.shape == (100,2)

    return



def test_grid_pool_map():

    test_grid = Grid((20,20))

    output = test_grid.pool_map(kernel_size=2)

    assert type(output) == dict

    for key in output.keys():
        assert len(output[key]) == 4

    return


def test_mesh_downsample():

    #set random seed
    torch.manual_seed(0)

    points = torch.rand(100,3)

    test_mesh = Mesh(points = points)

    smaller_mesh = test_mesh.downsample(factor=2)

    assert smaller_mesh.points.shape[0] <= test_mesh.points.shape[0]

    return

def test_mesh_pool_map():
    
    #set random seed
    torch.manual_seed(0)

    points = torch.rand(100,3)

    test_mesh = Mesh(points = points)

    index_map = test_mesh.pool_map(factor=2)

    assert type(index_map) == dict

    return     