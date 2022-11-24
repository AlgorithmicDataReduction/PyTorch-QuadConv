'''
'''

import torch
import torch.nn as nn

from .utilities import newton_cotes_quad

'''
Point and Quadrature data handler.

Input:
    input_nodes: input mesh nodes
    input_weights: input mesh quadrature weights
    input_adjacency: input mesh adjacency structure
    quad_map: map from number of points to quadrature (nodes, weights)
'''
class MeshHandler(nn.Module):

    def __init__(self,
            input_nodes,
            input_weights = None,
            input_adjacency = None,
            quad_map = newton_cotes_quad
        ):
        super().__init__()

        ##attributes
        self._current_index = 0

        self._spatial_dim = input_nodes.shape[1]

        self._quad_map = quad_map

        #nodes
        self._nodes = nn.ParameterList([nn.Parameter(input_nodes, requires_grad=False)])

        #weights
        if input_weights is None:
            input_weights = torch.ones(input_nodes.shape[0])
            req_grad = True
        else:
            req_grad = False

        self._weights = nn.ParameterList([nn.Parameter(input_weights, requires_grad=req_grad)])

        #adjacency
        if input_adjacency != None:
            self._adjacency = nn.ParameterList([input_adjacency])
        else:
            self._adjacency = nn.ParameterList()

        return

    def step(self):
        self._current_index = (self._current_index + 1)%(self._num_stages)

        return

    def _get_index(self, offset=0):
        return self._radix - abs(self._radix - (self._current_index + offset))

    @property
    def input_nodes(self):
        return self._nodes[self._get_index()]

    @property
    def output_nodes(self):
        return self._nodes[self._get_index(1)]

    @property
    def weights(self):
        return self._weights[self._get_index()]

    @property
    def adjacency(self):
        return self._adjacency[self._get_index()]

    '''
    Cache mesh stages.

    Input:
        node_seq: sequence of number of nodes (e.g. [100, 50, 10])
        mirror: whether or not to mirror the node_seq, i.e. append reversed node_seq
    '''
    def cache(self, node_seq, mirror=False):

        #make sure input_nodes and node_seq align
        assert node_seq[0] == self._nodes[0].shape[0]

        #set number of meshes and mesh stages
        self._num_meshes = len(node_seq)
        self._num_stages = len(node_seq)-1
        self._radix = len(node_seq)-1

        #construct other node sets
        for num_nodes in node_seq[1:]:
            nodes, weights = self._quad_map(self._spatial_dim, num_nodes)

            if weights is None:
                weights = torch.ones(nodes.shape[0])
                req_grad = True
            else:
                req_grad = False

            self._nodes.append(nn.Parameter(nodes, requires_grad=False))
            self._weights.append(nn.Parameter(weights, requires_grad=req_grad))

        #mirror the sequence, but reuse underlying data
        if mirror:
            self._num_stages *= 2

        return self
