'''
Learned quadrature convolutions.
'''

import torch
import torch.nn as nn
import torch_scatter

from core.FastGL.glpair import glpair
from scipy.integrate import newton_cotes

from core.utilities import Sin

'''
Quadrature convolution operator.

Input:
    spatial_dim: space dimension
    num_points_in: number of input points
    num_points_out: number of output points
    in_channels: input feature channels
    out_channels: output feature channels
    filter_seq: complexity of point to filter operation

NOTE: The points in and points out actually refer to the number of points along
each spatial dimension.
'''
class QuadConvLayer(nn.Module):

    def __init__(self,*,
            spatial_dim,
            num_points_in,
            num_points_out,
            in_channels,
            out_channels,
            filter_seq,
            use_bias = False,
            mlp_mode = 'single',
            quad_type = 'newton',
            composite_quad_order = 2
        ):
        super().__init__()

        #valid spatial dim
        assert spatial_dim > 0

        #set hyperparameters
        self.spatial_dim = spatial_dim
        self.num_points_in = num_points_in
        self.num_points_out = num_points_out
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.use_bias = use_bias
        self.composite_quad_order = composite_quad_order

        #decay parameter
        self.decay_param = (self.num_points_in/16)**2

        #quadrature
        if quad_type == 'newton':
            self.quad = self.newton_cotes_quad
        elif quad_type == 'gauss':
            self.quad = self.gauss_quad
        else:
            raise ValueError(f'Quadrature type {self.quad_type} is not supported.')

        #bias
        if self.use_bias:
            bias = torch.empty(1, self.out_channels, self.num_points_out)
            self.bias = nn.Parameter(nn.init.xavier_uniform_(bias), requires_grad=True)

        #cache indices
        self.cache()

        #initialize filter
        self.init_filter(filter_seq)

    '''
    Initialize the layer filter.

    Input:
        filter_seq: mlp feature sequence
    '''
    def init_filter(self, filter_seq):
        #NOTE: This is just a placeholder
        # self.G = nn.Parameter(torch.randn(self.out_channels, self.in_channels), requires_grad=True).expand(self.idx.shape[0], -1, -1)

        #point to matrix operation is stack of point to vector mlp operations
        self.G = nn.Sequential()

        activation = Sin
        bias = False

        feature_seq = (self.spatial_dim, *filter_seq, self.in_channels*self.out_channels)

        for i in range(len(feature_seq)-2):
            self.G.append(nn.Linear(feature_seq[i], feature_seq[i+1], bias=bias))
            self.G.append(activation())

        self.G.append(nn.Linear(feature_seq[-2], feature_seq[-1], bias=bias))
        self.G.append(nn.Unflatten(1, (self.in_channels, self.out_channels)))

        return

    '''
    Get Gaussian quadrature weights and nodes.

    Input:
        num_points: number of points
    '''
    def gauss_quad(self, num_points):
        num_points = int(num_points**(1/self.spatial_dim))

        quad_weights = torch.zeros(num_points)
        quad_nodes = torch.zeros(num_points)

        for i in range(num_points):
            _, quad_weights[i], quad_nodes[i] = glpair(num_points, i+1)

        return quad_weights, quad_nodes

    '''
    Get Newton-Cotes quadrature weights and nodes.
    NOTE: This function returns the composite rule, so its required that the order
    of the quadrature rule divides evenly into N.

    Input:
        num_points: number of points
        x0: left end point
        x1: right end point
    '''
    def newton_cotes_quad(self, num_points, x0=0, x1=1):
        num_points = int(num_points**(1/self.spatial_dim))
        rep = [int(num_points/self.composite_quad_order)]

        dx = (x1-x0)/(self.composite_quad_order-1)

        weights, _ = newton_cotes(self.composite_quad_order-1, 1)
        weights = torch.tile(torch.Tensor(dx*weights), rep)

        return weights, torch.linspace(x0, x1, num_points)

    '''
    Compute convolution output locations.
    '''
    def output_locs(self):
        _, mesh_nodes = self.quad(self.num_points_out)

        node_list = [mesh_nodes]*self.spatial_dim

        output_locs = torch.dstack(torch.meshgrid(*node_list, indexing='xy')).view(-1, self.spatial_dim)

        return output_locs

    '''
    Compute indices associated with non-zero filters
    '''
    def cache(self):
        _, quad_nodes = self.quad(self.num_points_in)
        output_locs = self.output_locs()

        #create mesh
        node_list = [quad_nodes]*self.spatial_dim
        nodes = torch.meshgrid(*node_list, indexing='xy')
        nodes = torch.dstack(nodes).view(-1, self.spatial_dim)

        #determine indices
        #NOTE: This seems to be the source of the memory issues
        locs = (output_locs.repeat_interleave(nodes.shape[0], dim=0) - nodes.repeat(output_locs.shape[0], 1)).view(output_locs.shape[0], nodes.shape[0], self.spatial_dim)

        bump_arg = torch.linalg.vector_norm(locs, dim=(2), keepdims = True)**4

        tf_vec = (bump_arg <= 1/self.decay_param).squeeze()
        idx = torch.nonzero(tf_vec, as_tuple=False)

        self.eval_locs = nn.Parameter(locs[tf_vec, :], requires_grad=False)
        self.eval_indices = nn.Parameter(idx, requires_grad=False)

        return

    '''
    Apply operator via quadrature approximation of convolution with features and learned filter.

    Input:
        features:  a tensor of shape (batch size  X num of points X input channels)

    Output: tensor of shape (batch size X num of output points X output channels)

    NOTE: THIS WOULD ALL BE A LOT EASIER IF WE WERE DOING CHANNELS_LAST BUT PYTORCH DOESN'T LIKE THAT
    '''
    def forward(self, features):
        integral = torch.zeros(features.shape[0], self.out_channels, self.num_points_out, device=features.device)

        #compute filter feature mat-vec products
        # values = torch.matmul(features[:,self.eval_indices[:,1],:], self.G(self.eval_locs))
        values = torch.einsum('bni, nij -> bnj', features[:,:,self.eval_indices[:,1]].reshape(features.shape[0], -1, self.in_channels), self.G(self.eval_locs)).reshape(features.shape[0], self.out_channels, -1)

        #both of the following are valid
        #NOTE: segment_coo should be faster because the indices are ordered
        torch_scatter.segment_coo(values, self.eval_indices[:,0].expand(features.shape[0], self.out_channels, -1), integral, reduce="sum")
        # torch_scatter.scatter(values, self.eval_indices[:,0], 2, integral, reduce="sum")

        #add bias
        if self.use_bias:
            integral += self.bias

        return integral
