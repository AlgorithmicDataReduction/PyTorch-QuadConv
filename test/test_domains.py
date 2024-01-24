from torch_quadconv import Grid
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
    pass

def test_mesh_pool_map():
    pass    