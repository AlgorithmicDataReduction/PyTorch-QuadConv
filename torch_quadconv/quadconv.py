'''
'''

import torch
import torch.nn as nn
import numpy as np

from .utils.misc import Sin

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
            decay_param = None,
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
        self.weight_map = None
        self.einsum_op = None

        #decay parameter
        if decay_param == None:
            self.decay_param = 4/np.sqrt(self.domain.points)
        else:
            self.decay_param = decay_param

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

            self.filter = self._create_mlp(mlp_spec)
            self.H = lambda z: self.filter(z).reshape(-1, self.in_channels, self.out_channels)

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

        #linear layer settings
        activation = Sin()
        bias = False

        #build mlp
        mlp = nn.Sequential()

        for i in range(len(mlp_channels)-2):
            mlp.append(nn.Linear(mlp_channels[i], mlp_channels[i+1], bias=bias))
            mlp.append(activation)

        mlp.append(nn.Linear(mlp_channels[-2], mlp_channels[-1], bias=bias))

        return mlp

    '''
    Calculate bump vector norm.

    Input:
        z: evaluation locations, [out_points, in_points, spatial_dim]
    '''
    def _bump_arg(self, z):
        return torch.linalg.vector_norm(z, dim=(2), keepdims = True)
        # a, b = 2, 0.5
        # a, b = 1, 1
        # return torch.sqrt(z[:,:,0]**2/a**2 + z[:,:,1]**2/b**2)

    '''
    Compute indices associated with non-zero filters.

    Input:
        self.domain / self.range: 
    '''
    def _compute_eval_indices(self):

        #get output points
        input_points = self.domain.points
        output_points = input_points if self.output_same else self.range.points

        #determine indices
        #NOTE: The following block is what we would want to loop on for computing these evaluation indices in batches
        ####
        locs = output_points.unsqueeze(1) - input_points.unsqueeze(0)

        bump_arg = self._bump_arg(locs)

        tf_vec = (bump_arg <= self.decay_param).squeeze()
        idx = torch.nonzero(tf_vec, as_tuple=False)
        ####

        if self.cache:
            self.register_buffer('eval_indices', idx, persistent=False)
            self.cached = True

        if self.verbose:
            print(f"\nQuadConv eval_indices: {idx.numel()}")

            hist = torch.histc(idx[:,1].to(torch.float32), bins=self.out_points, min=0, max=self.out_points-1)

            print(f"Max support points: {torch.max(hist)}")
            print(f"Min support points: {torch.min(hist)}")
            print(f"Avg support points: {torch.sum(hist)/hist.numel()}")

        return idx
    

    def _build_weight_map(self, element_size = 3, dimension = 2):

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
    

    def eval_weight_map(self, domain, element_size = 3, dimension = 2):

        element_list = domain.adjacency

        el_points = domain.points[element_list].reshape(-1, element_size * dimension)

        el_weights =  self.weight_map(el_points)

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
        filters = self.G(eval_locs)

        integral = self._integrate(weights, filters, features, eval_indices)

        #add bias
        if self.bias is not None:
            integral += self.bias

        return integral
    
    #@torch.compile
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

    '''
    Apply operator via quadrature approximation of convolution with features and learned filter.

    Input:
        mesh: MeshHandler object
        features: a tensor of shape (batch size X input channels X num input points)

    Output: tensor of shape (batch size X output channels X num output points)
    '''
    def forward(self, mesh, features):

        #get evaluation indices
        if self.cached:
            eval_indices = self.eval_indices
        else:
            eval_indices = self._compute_eval_indices(mesh)

        #get weights
        weights = mesh.weights()[eval_indices[:,1]]

        #compute eval locs
        if self.output_same:
            eval_locs = mesh.input_points()[eval_indices[:,0]] - mesh.input_points()[eval_indices[:,1]]
        else:
            eval_locs = mesh.output_points()[eval_indices[:,0]] - mesh.input_points()[eval_indices[:,1]]
            mesh.step()

        #compute filter
        filters = self.G(eval_locs)

        integral = self._integrate(weights, filters, features, eval_indices)

        #add bias
        if self.bias is not None:
            integral += self.bias

        return integral
