'''
'''

import torch
import torch.nn as nn

from .utils import quadrature

from scipy.spatial import Delaunay

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
            multilevel_map = 'mfnus',
            weight_activation = 'Identity',
            normalize_weights = False,
        ):
        super().__init__()

        if input_weights is not None:
            assert input_points.shape[0] == input_weights.shape[0]

        #points
        self._points = nn.ParameterList([nn.Parameter(input_points, requires_grad=False)])

        self.normalize_weights = normalize_weights

        self.point_seq = None

        #weights
        if input_weights is None:
            self.build_weight_map()
        else:
            req_grad = False
            self.weight_activation = getattr(nn, weight_activation)()

        #self._quad_map = getattr(quadrature, quad_map)

        self._elim_map = []

        self.multilevel_map = getattr(quadrature, multilevel_map)

        #adjacency
        if input_adjacency != None:
            self._adjacency = nn.ParameterList([input_adjacency])
        else:
            self._adjacency = nn.ParameterList([nn.Parameter(torch.from_numpy(Delaunay(input_points).simplices).long(), requires_grad=False)])

        #other attributes
        self._spatial_dim = input_points.shape[1]
        self._current_index = 0

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
        return self.eval_weight_map(self._points[self._get_index()])

    @property
    def adjacency(self):
        return self._adjacency[self._get_index()]
    
    def get_downsample_map(self, key_num):

        output = list(dsm for dsm in self._elim_map if len(dsm) == key_num)[0]

        return output
    


    def build_weight_map(self, element_size = 3, dimension = 2):

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
    

    def eval_weight_map(self, points, element_size = 3, dimension = 2):

        element_list = self.adjacency

        el_points = points[element_list].reshape(-1, element_size * dimension)

        el_weights =  self.weight_map(el_points)

        weights = torch.zeros(points.shape[0], device=points.device)

        weights.scatter_add_(0, element_list.reshape(-1), el_weights.reshape(-1))

        return weights
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

        self.point_seq = point_seq.copy()

        #construct other point sets
        for i, num_points in enumerate(point_seq[1:]):
            points, weights, *elim_map = self._quad_map(self._points[i].clone(), num_points)

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
            self._num_stages *= 2

        return self


    def build_seq(self, stages, mirror=False):

        #make sure input_points and point_seq align

        #set number of meshes and mesh stages
        self._num_meshes = stages + 1
        self._num_stages = stages
        self._radix = stages

        self.point_seq = []

        self.point_seq.append(self._points[0].shape[0])

        #construct other point sets
        for i in range(stages):
            points, elim_map = self.multilevel_map(self._points[i].clone())

            self._points.append(nn.Parameter(torch.from_numpy(points), requires_grad=False))

            self.point_seq.append(points.shape[0])

            self._adjacency.append(nn.Parameter(torch.from_numpy(Delaunay(points).simplices).long(), requires_grad=False))

            self._elim_map.append(elim_map)

        #mirror the sequence, but reuse underlying data
        if mirror:
            self._num_stages *= 2

        return self