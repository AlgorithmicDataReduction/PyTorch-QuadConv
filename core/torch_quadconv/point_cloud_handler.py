'''
'''

import torch
import torch.nn as nn

from .utilities import newton_cotes_quad

'''
Point and Quadrature data handler.

Input:
    input_points: input points
    point_seq: sequence of number of nodes (e.g. [100, 50, 10])
    point_map: map from number of points to point locations
'''
class PointCloudHandler(nn.Module):

    def __init__(self,
            input_points,
            point_seq,
            point_map = lambda x : newton_cotes_quad(x[0], x[1])[0]
        ):
        super().__init__()

        ##attributes
        self._points = input_points
        self._weights = nn.ParameterList()

        self._point_seq = point_seq

        self._current_index = 0

        self._spatial_dim = input_points.shape[1]

        self._point_map = point_map

        return

    def step(self):
        self._current_index = (self._current_index + 1)%(len(self._point_seq))

        return

    @property
    def input_nodes(self):
        return self._nodes

    @property
    def output_nodes(self):
        pass

    @property
    def weights(self):
        return self._weights[self._get_index()]
