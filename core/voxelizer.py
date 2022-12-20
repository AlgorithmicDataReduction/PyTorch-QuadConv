
import torch

from torch_cluster import grid_cluster
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_scatter import segment_coo as scatter

'''
NOTE: Using this as a reference https://torch-points3d.readthedocs.io/en/latest/_modules/torch_points3d/core/data_transform/grid_transform.html#GridSampling3D
'''
class Voxelizer():

    '''
    Input:
        points: NxD tensor of points
    '''
    def __init__(self, points, voxel_size):

        self._voxel_size = voxel_size

        point_dim = points.shape[1]

        coords = torch.round(points / self._voxel_size)
        cluster = grid_cluster(coords, torch.tensor([1 for i in range(point_dim)]))

        cluster, _ = consecutive_cluster(cluster)

        self._num_voxels = torch.max(cluster)+1
        self._indices = cluster
        self._grid_shape = [int(self._num_voxels**(1/point_dim))]*point_dim

        return

    '''
    Convert point-cloud features to a voxel grid.

    Input:
        features: BxCxN tensor of features
    '''
    def voxelize(self, features):

        batch_size = features.shape[0]
        channels = features.shape[1]

        voxels = torch.zeros(batch_size, channels, self._num_voxels)
        scatter(features, self._indices.expand(batch_size, channels, -1), voxels, reduce="mean")

        return voxels.view(tuple([batch_size, channels]+self._grid_shape))

    '''
    Conver a voxel grid to a point-cloud.

    Input:
        voxels: N1xN2x...NDxC tensor of voxels
    '''
    def devoxelize(self, voxels):

        batch_size = voxels.shape[0]
        channels = voxels.shape[1]

        return voxels.view(batch_size, channels, -1)[:,:,self._indices]
