
import torch
import torch.nn as nn

from torch_cluster import grid_cluster
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_scatter import scatter

'''
NOTE: Using this as a reference https://torch-points3d.readthedocs.io/en/latest/_modules/torch_points3d/core/data_transform/grid_transform.html#GridSampling3D
'''
class Voxelizer(nn.Module):

    '''
    Input:
        points: NxD tensor of points
    '''
    def __init__(self, points, voxel_size):
        super().__init__()

        #extract some info and check
        self._voxel_size = torch.tensor(voxel_size)
        point_dim = points.shape[1]

        assert point_dim == self._voxel_size.numel(), f"{point_dim}, {self._voxel_size.numel()}"

        #
        coords = torch.empty_like(points)
        coords = torch.trunc(points / (self._voxel_size + 1e-6))

        #
        self._grid_shape = torch.Size((torch.max(coords, dim=0)[0] - torch.min(coords, dim=0)[0]).int() + 1)

        #
        cluster = grid_cluster(coords, torch.tensor([1 for i in range(point_dim)]))
        self._num_voxels = torch.max(cluster)+1

        assert self._num_voxels == self._grid_shape.numel(), f"{self._num_voxels}, {self._grid_shape.numel()}"

        #
        cluster, _ = consecutive_cluster(cluster)
        self._indices = nn.Parameter(cluster, requires_grad=False)

        return

    '''
    Convert point-cloud features to a voxel grid.

    Input:
        features: BxCxN tensor of features
    '''
    def voxelize(self, features):

        batch_size = features.shape[0]
        channels = features.shape[1]

        voxels = scatter(features, self._indices, dim=2, dim_size=self._num_voxels, reduce="mean")

        return voxels.reshape(torch.Size([batch_size, channels])+self._grid_shape)

    '''
    Conver a voxel grid to a point-cloud.

    Input:
        voxels: N1xN2x...NDxC tensor of voxels
    '''
    def devoxelize(self, voxels):

        batch_size = voxels.shape[0]
        channels = voxels.shape[1]

        return voxels.reshape(batch_size, channels, -1)[:,:,self._indices]
