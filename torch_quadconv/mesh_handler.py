'''
'''

import torch
import torch.nn as nn

from .utils import quadrature

'''
Point and Quadrature data handler.

Input:
    input_points: input mesh nodes
    input_weights: input mesh quadrature weights
    input_adjacency: input mesh adjacency structure
    quad_map: map from number of points to quadrature (points, weights)
'''
class MeshHandler(nn.Module):

    def __init__(self,
            input_points,
            input_weights = None,
            input_adjacency = None,
            quad_map = 'newton_cotes_quad',
            weight_activation = nn.Identity,
            normalize_weights = False,
        ):
        super().__init__()

        if input_weights is not None:
            assert input_points.shape[0] == input_weights.shape[0]

        #points
        self._points = nn.ParameterList([nn.Parameter(input_points, requires_grad=False)])

        self.normalize_weights = normalize_weights

        #weights
        if input_weights is None:
            input_weights = torch.zeros(input_points.shape[0])
            input_weights = torch.nn.init.uniform_(input_weights, a= 0, b = 1)
            self.weight_activation = getattr(nn, weight_activation)()
            req_grad = True
        else:
            req_grad = False
            self.weight_activation = getattr(nn, weight_activation)()

        self._weights = nn.ParameterList([nn.Parameter(input_weights, requires_grad=req_grad)])


        #adjacency
        if input_adjacency != None:
            self._adjacency = nn.ParameterList([input_adjacency])
        else:
            self._adjacency = nn.ParameterList([None])

        #other attributes
        self._spatial_dim = input_points.shape[1]
        self._current_index = 0
        self._quad_map = getattr(quadrature, quad_map)

        return

    def step(self):
        self._current_index = (self._current_index + 1)%(self._num_stages)

        return

    def _get_index(self, offset=0):
        return self._radix - abs(self._radix - (self._current_index + offset))

    @property
    def input_points(self):
        return self._points[self._get_index()]

    @property
    def output_points(self):
        return self._points[self._get_index(1)]

    @property
    def weights(self):
        if self.normalize_weights:
            with torch.no_grad():
                self._weights[self._get_index()] = self._weights[self._get_index()] / torch.sum(self.weight_activation(self._weights[self._get_index()]))

        return self.weight_activation(self._weights[self._get_index()])

    @property
    def adjacency(self):
        return self._adjacency[self._get_index()]

    '''
    Cache mesh stages.

    Input:
        point_seq: sequence of number of points (e.g. [100, 50, 10])
        mirror: whether or not to mirror the point_seq, i.e. append reversed point_seq
    '''
    def cache(self, point_seq, mirror=False):

        #make sure input_points and point_seq align
        assert point_seq[0] == self._points[0].shape[0]

        #set number of meshes and mesh stages
        self._num_meshes = len(point_seq)
        self._num_stages = len(point_seq)-1
        self._radix = len(point_seq)-1

        #construct other point sets
        for i, num_points in enumerate(point_seq[1:]):
            points, weights = self._quad_map(self._points[i-1], num_points)

            if weights is None:
                weights = torch.ones(points.shape[0])
                #weights = torch.nn.init.uniform_(weights, a= 0, b = 1)
                weights /= torch.sum(weights)

                req_grad = True
            else:
                req_grad = False

            self._points.append(nn.Parameter(points, requires_grad=False))
            self._weights.append(nn.Parameter(weights, requires_grad=req_grad))

        #mirror the sequence, but reuse underlying data
        if mirror:
            self._num_stages *= 2

        return self
