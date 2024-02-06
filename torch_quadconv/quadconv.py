'''
'''

import torch
import torch.nn as nn
import numpy as np

from .utils.misc import Sin
from .utils.mlp import Siren

'''
Quadrature convolution operator.

Input:
    domain: the domain of the operator (as a domains object)
    range: the range of the operator (as a domains object)
    in_channels: input feature channels
    out_channels: output feature channels
    filter_seq: number of features at each filter stage
    filter_mode: type of point to filter operation
    bias: whether or not to use bias
    output_same: whether or not to use the input points as the output points
    cache: whether or not to cache the evaluation indices
'''
class QuadConv(nn.Module):

    def __init__(self,
            domain,
            range,
            in_channels,
            out_channels,
            filter_seq = [16, 16, 16, 16, 16],
            filter_mode = 'single',
            knn = 9,
            omega_0 = 1.0, 
            bias = False,
            output_same = False,
            cache = True,
            verbose = False
        ):
        super().__init__()

        #validate spatial dim
        assert domain.shape[1] == range.shape[1], "Domain and range must have the same spatial dimension"

        #set attributes
        self.spatial_dim = domain.shape[1]

        self.add_module('domain', domain)
        self.add_module('range', range)

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.output_same = output_same

        self.cache = cache
        self.cached = False
        self.verbose = verbose
        self.knn = knn
        self.omega_0 = omega_0

        #initialize filter
        self._init_filter(filter_seq, filter_mode)

        self._build_weight_map()

        #bias
        if bias:
            bias = torch.empty(1, self.out_channels, self.range.shape[0])
            self.bias = nn.Parameter(nn.init.xavier_uniform_(bias, gain=2), requires_grad=True)
        else:
            self.bias = None

        return

    '''
    Initialize the layer filter.

    Input:
        filter_seq: mlp feature sequence
        filter_mode: type of filter operation
    '''
    def _init_filter(self, filter_seq, filter_mode):

        #single mlp
        if filter_mode == 'single':

            mlp_spec = (self.spatial_dim, *filter_seq, self.in_channels*self.out_channels)

            mlp = self._create_mlp(mlp_spec)
            self.H = lambda z: self.mlp(z).reshape(-1, self.in_channels, self.out_channels)

        #mlp for each output channel
        elif filter_mode == 'share_in':

            mlp_spec = (self.spatial_dim, *filter_seq, self.in_channels)

            self.filter = nn.ModuleList()
            for j in range(self.out_channels):
                self.filter.append(self._create_mlp(mlp_spec))

            self.H = lambda z: torch.cat([module(z) for module in self.filter]).reshape(-1, self.channels_in, self.channels_out)

        #mlp for each input and output channel pair
        elif filter_mode == 'nested':

            mlp_spec = (self.spatial_dim, *filter_seq, 1)

            self.filter = nn.ModuleList()
            for i in range(self.in_channels):
                for j in range(self.out_channels):
                    self.filter.append(self._create_mlp(mlp_spec))

            self.H = lambda z: torch.cat([module(z) for module in self.filter]).reshape(-1, self.in_channels, self.out_channels)

        else:
            raise ValueError(f'core::modules::quadconv: Filter mode {filter_mode} is not supported.')

        #NOTE: Unnecessary, but leaving this in for backwards compatability
        self.G = self.H

        return

    '''
    Build an mlp.

    Input:
        mlp_channels: sequence of channels
    '''
    def _create_mlp(self, mlp_channels):

        self.register_module('mlp', Siren(mlp_channels, outermost_linear=True, first_omega_0=self.omega_0, hidden_omega_0=self.omega_0))

        return self.mlp

    '''
    Compute indices associated with non-zero filters.

    Input:
        self.domain / self.range: 
    '''
    def _compute_eval_indices(self):

        idx = self.domain.query(self.range, k=self.knn)[1].reshape(-1, self.knn)

        eval_indices = torch.tensor([ [i,idx[i,j]] for i in range(idx.shape[0]) for j in range(idx.shape[1]) ]).to(self.domain.points.device, dtype=torch.long)

        if self.cache:
            self.register_buffer('eval_indices', eval_indices, persistent=False)
            self.cached = True

        if self.verbose:
            print(f"\nQuadConv eval_indices: {eval_indices.numel()}")

            unique_val_counts = torch.unique(eval_indices[:,0].to(torch.float32), return_counts=True)[1]

            print(f"Max support points: {torch.max(unique_val_counts)}")
            print(f"Min support points: {torch.min(unique_val_counts)}")
            print(f"Avg support points: {torch.sum(unique_val_counts)/unique_val_counts.numel()}")

        return eval_indices
    

    def _build_weight_map(self, element_size = 3, dimension = 2):

        weight_map = nn.Sequential(
            nn.Linear(element_size * dimension, 8),
            nn.Sigmoid(),
            nn.Linear(8, 8),
            nn.Sigmoid(),
            nn.Linear(8, 8),
            nn.Sigmoid(),
            nn.Linear(8, element_size),
            nn.Sigmoid()
        )

        self.register_module('weight_map', weight_map)

        return
    

    def eval_weight_map(self, domain, element_size = 3, dimension = 2):

        element_list = domain.adjacency

        el_points = domain.points[element_list].reshape(-1, element_size * dimension)

        el_weights =  self.weight_map(el_points)

        el_weights = el_weights

        weights = torch.zeros(domain.points.shape[0]).to(el_points)

        weights.scatter_add_(0, element_list.reshape(-1), el_weights.reshape(-1))

        return weights


    '''
    Compute QuadConv integral
    
    Input:
        features: a tensor of shape (batch size  X num of points X input channels)

    Output: tensor of shape (batch size X num of output points X output channels)
    '''
    def forward(self, features):

        #get evaluation indices
        if self.cached:
            eval_indices = self.eval_indices
        else:
            eval_indices = self._compute_eval_indices()

        #get weights
        weights = self.eval_weight_map(self.domain)[eval_indices[:,1]]

        #compute eval locs
        if self.output_same:
            eval_locs = self.domain.points[eval_indices[:,0]] - self.domain.points[eval_indices[:,1]]
        else:
            eval_locs = self.range.points[eval_indices[:,0]] - self.domain.points[eval_indices[:,1]]


        #compute filter
        filters = self.G(eval_locs) / self.in_channels

        integral = self._integrate(weights, filters, features, eval_indices)

        #add bias
        if self.bias is not None:
            integral += self.bias

        return integral
    
    def _integrate(self, weights, filters, features, eval_indices):

        '''
        #compute interior of the integral (including weights) for all left hand sides of the integral
        weights = weights.reshape(1, -1 , 1, 1)

        filters = filters.view(1, *filters.shape)

        features = features[:,:,eval_indices[:,1]].unsqueeze(2).permute(0, 3, 1, 2)

        values = (weights * filters * features).sum(dim=2).permute(0, 2, 1)
        '''
        
        #compute quadrature as weights*filters*features
        values = torch.einsum('n, nij, bin -> bjn',
                                weights,
                                filters,
                                features[:,:,eval_indices[:,1]])

        #setup integral array
        integral = values.new_zeros(features.shape[0], self.out_channels, self.range.shape[0])

        #scatter values via addition into integral array
        integral.scatter_add_(2, eval_indices[:,0].expand(features.shape[0], self.out_channels, -1), values)

        return integral