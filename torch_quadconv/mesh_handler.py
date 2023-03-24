'''
'''

import torch
import torch.nn as nn

from .utils import quadrature
from .utils.agglomeration import agglomerate

class Elements(nn.Module):

     def __init__(self, element_pos, element_ind, bd_point_ind=None):
         super().__init__()

         self.element_pos = element_pos
         self.element_ind = element_ind
         self.bd_point_ind = bd_point_ind

         return

'''
Point and Quadrature data handler.

Input:
    input_points: input mesh nodes
    input_weights: input mesh quadrature weights
    input_elements: input mesh element structure
    quad_map: map from number of points to quadrature (points, weights)
'''
class MeshHandler(nn.Module):

    def __init__(self,
            input_points,
            input_weights = None,
            input_elements = None,
            weight_activation = "Sigmoid",
            normalize_weights = False
        ):
        super().__init__()

        if input_weights is not None:
            assert input_points.shape[0] == input_weights.shape[0]

        #points
        self._points = nn.ParameterList([nn.Parameter(input_points, requires_grad=False)])

        #weights
        if input_weights is None:
            input_weights = torch.nn.init.uniform_(torch.empty(input_points.shape[0]), a=0, b=1)
            req_grad = True
        else:
            req_grad = False

        self._weights = nn.ParameterList([nn.Parameter(input_weights, requires_grad=req_grad)])

        self._weight_activation = getattr(nn, weight_activation)()
        self._normalize_weights = normalize_weights

        #adjacency
        if input_elements != None:
            self._elements = nn.ModuleList([input_elements])
        else:
            self._elements = nn.ModuleList([None])

        #other attributes
        self._spatial_dim = input_points.shape[1]
        self._current_index = 0

        return

    def reset(self):
        self._current_index = 0

        return

    def step(self):
        self._current_index = (self._current_index + 1)%(self._num_levels)

        return

    def _get_index(self, offset=0):
        return self._radix - abs(self._radix - (self._current_index + offset))

    def input_points(self):
        return self._points[self._get_index()]

    def output_points(self):
        return self._points[self._get_index(1)]

    def weights(self):
        if self._normalize_weights:
            with torch.no_grad():
                self._weights[self._get_index()] = self._weights[self._get_index()] / torch.sum(self._weight_activation(self._weights[self._get_index()]))

        w = self._weight_activation(self._weights[self._get_index()])

        return w

    def elements(self):
        return self._elements[self._get_index()]

    '''
    Construct mesh levels via quadrature map.

    Input:
        point_seq: sequence of number of points (e.g. [100, 50, 10])
        mirror: whether or not to mirror the point_seq, i.e. append reversed point_seq
    '''
    def construct(self, point_seq, mirror=False, quad_map='newton_cotes_quad', quad_args={}):

        #get quad map
        quad_map = getattr(quadrature, quad_map)

        #make sure input_points and point_seq align
        assert point_seq[0] == self._points[0].shape[0]

        #set number of meshes and mesh stages
        self._num_meshes = len(point_seq)
        self._num_levels = len(point_seq)-1
        self._radix = len(point_seq)-1

        #construct other point sets
        for i, num_points in enumerate(point_seq[1:]):
            points, weights = quad_map(self._points[i-1], num_points, **quad_args)

            if weights is None:
                weights = torch.nn.init.uniform_(torch.empty(points.shape[0]), a=0, b=1)
                req_grad = True
            else:
                req_grad = False

            self._points.append(nn.Parameter(points, requires_grad=False))
            self._weights.append(nn.Parameter(weights, requires_grad=req_grad))
            self._elements.append(None)

        #mirror the sequence, but reuse underlying data
        if mirror:
            self._num_levels *= 2

        return self

    '''
    Construct mesh levels via agglomeration
    '''
    def agglomerate(self, levels, factor, mirror=False):

        #set number of meshes and mesh stages
        self._num_meshes = levels+1
        self._num_levels = levels
        self._radix = levels

        #agglomerate
        activity = agglomerate(self._points[0].data.numpy(), self._elements[0], levels, factor)

        for i in range(levels):
            sub_points = self._points[0][activity[:,i]]
            weights = torch.nn.init.uniform_(torch.empty(sub_points.shape[0]), a=0, b=1)

            self._points.append(nn.Parameter(sub_points, requires_grad=False))
            self._weights.append(nn.Parameter(weights, requires_grad=True))
            self._elements.append(None)

        #mirror the sequence, but reuse underlying data
        if mirror:
            self._num_levels *= 2

        return self
