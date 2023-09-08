'''
'''

import torch
import torch.nn as nn
from scipy.spatial import Delaunay

from .utils import quadrature

'''
Point and Quadrature data handler.

Input:
    input_points: input mesh nodes
    input_weights: input mesh quadrature weights
    input_adjacency: input adjacency structure
    weight_activation: activation function for weights
    normalize_weights: whether to normalize weights
'''
class MeshHandler(nn.Module):

    def __init__(self,
            input_points,
            input_weights = None,
            input_adjacency = None,
            weight_activation = 'Sigmoid',
            normalize_weights = False,
        ):
        super().__init__()

        if input_weights is not None:
            assert input_points.shape[0] == input_weights.shape[0]

        #points
        self._points = nn.ParameterList([nn.Parameter(input_points, requires_grad=False)])

        self.point_seq = None

        self._downsample_map = []

        #weights
        if input_weights is None:
            self.build_weight_map()
        else:
            req_grad = False

        self._weight_activation = getattr(nn, weight_activation)() if req_grad else nn.Identity()
        self._normalize_weights = normalize_weights

        #adjacency
        if input_adjacency != None:
            self._adjacency = nn.ModuleList([input_adjacency])
        else:
            self._adjacency = nn.ParameterList([nn.Parameter(torch.from_numpy(Delaunay(input_points).simplices).long(), requires_grad=False)])

        #other attributes
        self._spatial_dim = input_points.shape[1]
        self._current_index = 0

        return

    def reset(self, mirror=False):
        self._current_index = 0 + mirror*self._radix

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
        return self.eval_weight_map(self._points[self._get_index()])

    def adjacency(self):
        return self._adjacency[self._get_index()]
    
    def get_downsample_map(self, key_num):
        output = list(dsm for dsm in self._downsample_map if len(dsm) == key_num)[0]

        return output
    
    def build_weight_map(self, element_size=3, dimension=2):

        self.weight_map = nn.Sequential(
            nn.Linear(element_size * dimension, 8),
            nn.Sigmoid(),
            nn.Linear(8, 8),
            nn.Sigmoid(),
            nn.Linear(8, 8),
            nn.Sigmoid(),
            nn.Linear(8, element_size),
            nn.Sigmoid()
        )

        return
    
    def eval_weight_map(self, points, element_size=3, dimension=2):

        element_list = self.adjacency

        el_points = points[element_list].reshape(-1, element_size * dimension)

        el_weights =  self.weight_map(el_points)

        weights = torch.zeros(points.shape[0], device=points.device)

        weights.scatter_add_(0, element_list.reshape(-1), el_weights.reshape(-1))

        return weights
    
    '''
    Construct mesh levels via quadrature map.

    Input:
        point_seq: sequence of number of points (e.g. [100, 50, 10])
        mirror: whether or not to mirror the point_seq, i.e. append reversed point_seq
        quad_map: 
    '''
    def construct(self, point_seq, mirror=False, quad_map='param_quad', quad_args={}):

        #get quad map
        quad_map = getattr(quadrature, quad_map)

        #make sure input_points and point_seq align
        assert point_seq[0] == self._points[0].shape[0]

        #set number of meshes and mesh stages
        self._num_meshes = len(point_seq)
        self._num_levels = len(point_seq)-1
        self._radix = len(point_seq)-1

        self.point_seq = point_seq.copy()

        #construct other point sets
        for i, num_points in enumerate(point_seq[1:]):
            points, weights, *elim_map = quad_map(self._points[i].clone(), num_points)

            if weights is None:
                weights = torch.ones(points.shape[0])
                weights /= torch.sum(weights)
                req_grad = True
            else:
                req_grad = False

            self._points.append(nn.Parameter(torch.from_numpy(points), requires_grad=False))

            if num_points != points.shape[0]:
                self.point_seq[i+1] = points.shape[0]
                Warning('Number of points in the user defined sequence is changing from {num_points} to {points.shape[0]}')

            #self._weights.append(nn.Parameter(weights, requires_grad=req_grad))
            self._adjacency.append(nn.Parameter(torch.from_numpy(Delaunay(points).simplices).long(), requires_grad=False))

            if len(elim_map) > 0:
                self._downsample_map.append(elim_map[0])

        #mirror the sequence, but reuse underlying data
        if mirror:
            self._num_levels *= 2

        return self
