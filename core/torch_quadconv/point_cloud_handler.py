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

        assert input_points.shape[0] == point_seq[0]

        #points
        self._input_points = input_points
        self._output_points = point_map(input_points, point_seq[1])

        #weights
        self._weights = nn.ParameterList()
        for num_points in self._point_seq:
            self._weights.append(nn.Parameter(torch.ones(num_nodes), requires_grad=True))

        #other attributes
        self._point_seq = point_seq
        self._current_index = 0
        self._point_map = point_map

        return

    def step(self):
        self._current_index = (self._current_index + 1)%(len(self._point_seq))

        self._input_points = self._output_points
        self._output_points = self.point_map(self._input_points, self._point_seq[self._current_index])

        return

    @property
    def input_points(self):
        return self._points

    @property
    def output_points(self):
        return self._output_points

    @property
    def weights(self):
        return self._weights[self._current_index]
