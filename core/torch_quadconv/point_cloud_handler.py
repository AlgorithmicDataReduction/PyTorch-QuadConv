'''
'''

import torch
import torch.nn as nn

from .utilities import random_downsample

'''
Point and Quadrature data handler.

Input:
    input_points: input points
    point_seq: sequence of number of nodes (e.g. [100, 50, 10])
    point_map: map from number of points to point locations
'''
class PointCloudHandler(nn.Module):

    def __init__(self,
            point_seq,
            point_map = random_downsample
        ):
        super().__init__()

        #points
        self._input_points = None
        self._output_points = None

        #weights
        self._weights = nn.ParameterList()
        for num_points in point_seq:
            self._weights.append(nn.Parameter(torch.ones(num_points), requires_grad=True))

        #other attributes
        self._point_seq = point_seq
        self._current_index = 0
        self._num_stages = len(point_seq)-1
        self._point_map = point_map

        return

    def step(self):
        self._current_index = (self._current_index + 1)%(self._num_stages)

        if self._current_index != 0:
            self._input_points = self._output_points
            self._output_points = self._point_map(self._input_points, self._point_seq[self._current_index+1])
        else:
            self._input_points, self._output_points = None, None

        return

    @property
    def input_points(self):
        return self._input_points

    @property
    def output_points(self):
        return self._output_points

    @property
    def weights(self):
        return self._weights[self._current_index]

    def cache(self, input_points):

        assert input_points.shape[1] == self._point_seq[0]
        assert self._current_index == 0

        self._input_points = input_points
        self._output_points = self._point_map(input_points, self._point_seq[1])

        return
