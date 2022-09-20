'''
Learned quadrature convolutions.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

from core.FastGL.glpair import glpair

from core.utilities import Sin

'''
Quadrature convolution operator.

Input:
    point_dim: space dimension
    num_points_in: number of input points
    num_points_out: number of output points
    channels_in: input feature channels
    channels_out: output feature channels
    filter_size: complexity of point to filter operation
    use_bias: add bias term to output of layer

NOTE: The points in and points out actually refer to the number of points along
each spatial dimension.
'''
class QuadConvLayer(nn.Module):
    def __init__(self,
                    point_dim,
                    num_points_in,
                    num_points_out,
                    channels_in,
                    channels_out,
                    filter_size,
                    use_bias = False
                    ):
        super().__init__()

        #set hyperparameters
        self.point_dim = point_dim
        self.num_points_in = num_points_in
        self.num_points_out = num_points_out
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.use_bias = use_bias

        #decay parameter
        self.decay_param = (self.num_points_in/16)**2

        #bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(1, self.channels_out, self.num_points_out))

        #initialize filter
        # NOTE: This is just a placeholder
        self.G = nn.Parameter(torch.randn(self.channels_out, self.channels_in), requires_grad=True)

        #cache indices
        self.cache()

    '''
    Get Gaussian quadrature weights and nodes.

    Input:
        num_points: number of points
    '''
    def gauss_quad(self, num_points):
        num_points = int(num_points)

        quad_weights = torch.zeros(num_points)
        quad_nodes = torch.zeros(num_points)

        for i in range(num_points):
            _, quad_weights[i], quad_nodes[i] = glpair(num_points, i+1)

        return quad_weights, quad_nodes

    '''
    Compute convolution output locations.

    Input:
        n_points: number of output points
    '''
    def output_locs(self):
        _, mesh_nodes = self.gauss_quad(self.num_points_out**(1/self.point_dim))

        node_list = [mesh_nodes]*self.point_dim

        output_locs = torch.dstack(torch.meshgrid(*node_list, indexing='xy')).view(-1, self.point_dim)

        return output_locs

    '''
    Compute indices associated with non-zero filters
    '''
    def cache(self):
        print('Caching nonzero evaluation indices.')

        _, quad_nodes = self.gauss_quad(self.num_points_in**(1/self.point_dim))
        output_locs = self.output_locs()

        #create mesh
        node_list = [quad_nodes]*self.point_dim
        nodes = torch.meshgrid(*node_list, indexing='xy')
        nodes = torch.dstack(nodes).view(-1, self.point_dim)

        #determine indices
        #NOTE: This seems to be the source of the memory issues
        eval_locs = (output_locs.repeat_interleave(nodes.shape[0], dim=0) - nodes.repeat(output_locs.shape[0], 1)).view(output_locs.shape[0], nodes.shape[0], self.point_dim)

        bump_arg = torch.linalg.vector_norm(eval_locs, dim=(2), keepdims = True)**4

        tf_vec = (bump_arg <= 1/self.decay_param).squeeze()
        idx = torch.nonzero(tf_vec, as_tuple=False)

        self.eval_indices = idx

        print('Evaluation indices cached.')

        return

    '''
    Apply operator via quadrature approximation of convolution with features and learned filter.

    Input:
        features:  a tensor of shape (batch size X channels X number of points)

    Output: tensor of shape (batch size X output channels X number of output points)
    '''
    def forward(self, features):
        integral = torch.zeros(features.shape[0], self.channels_out, self.num_points_out, device=features.device)

        #compute convolutions

        '''
        For-loop approach.

        Much too slow.
        '''
        # for index in self.eval_indices:
        #     integral[:,:,index[0]] += torch.matmul(self.G, features[:,:,index[1]].T).T

        '''
        This approach uses the pytorch_scatter library.

        segment_coo: 24,889 μs
        scatter: 21,784 μs

        segment_coo should be faster because the indices are ordered
        '''
        values = torch.matmul(self.G, features[:,:,self.eval_indices[:,1]])

        #both of the following are valid
        # torch_scatter.segment_coo(values, self.eval_indices[:,0].to(features.device).expand(features.shape[0], self.channels_out, -1), integral, reduce="sum")
        torch_scatter.scatter(values, self.eval_indices[:,0].to(features.device), 2, integral, reduce="sum")

        #add bias
        if self.use_bias:
            integral += self.bias

        return integral

################################################################################

'''
QuadConvLayer block

Input:
    point_dim: space dimension
    channels_in: input feature channels
    channels_out: output feature channels
    n_points_in: number of input points
    n_points_out: number of output points
    filter_size: complexity of point to filter operation
    adjoint: downsample or upsample
    use_bias: add bias term to output of layer
    activation1:
    activation2:
'''
class QuadConvBlock(nn.Module):
    def __init__(self,
                    point_dim,
                    channels_in,
                    channels_out,
                    n_points_in,
                    n_points_out,
                    filter_size,
                    adjoint = False,
                    use_bias = False,
                    activation1 = nn.CELU(alpha=1),
                    activation2 = nn.CELU(alpha=1)
                    ):
        super().__init__()

        self.adjoint = adjoint
        self.activation1 = activation1
        self.activation2 = activation2

        if self.adjoint:
            conv1_point_num = n_points_out
            conv1_channel_num = channels_out
        else:
            conv1_point_num = n_points_in
            conv1_channel_num = channels_in

        self.conv1 = QuadConvLayer(point_dim,
                                    channels_in = conv1_channel_num,
                                    channels_out = conv1_channel_num,
                                    filter_size = filter_size,
                                    use_bias = use_bias
                                    )
        self.batchnorm1 = nn.InstanceNorm1d(conv1_channel_num)

        self.conv2 = QuadConvLayer(point_dim,
                                    channels_in = channels_in,
                                    channels_out = channels_out,
                                    filter_size = filter_size,
                                    use_bias = use_bias
                                    )
        self.batchnorm2 = nn.InstanceNorm1d(channels_out)


        if n_points_in and n_points_out:
            self.conv1.set_quad(conv1_point_num)
            self.conv2.set_quad(n_points_in)

            self.conv1.set_output_locs(conv1_point_num)
            self.conv2.set_output_locs(n_points_out)

    '''
    Forward mode
    '''
    def forward_op(self, data):
        x = data

        x1 = self.conv1(x)
        x1 = self.activation1(self.batchnorm1(x1))
        x1 = x1 + x

        x2 = self.conv2(x1)
        x2 = self.activation2(self.batchnorm2(x2))

        return x2

    '''
    Adjoint mode
    '''
    def adjoint_op(self, data):
        x = data

        x2 = self.conv2(x)
        x2 = self.activation2(self.batchnorm2(x2))

        x1 = self.conv1(x2)
        x1 = self.activation1(self.batchnorm1(x1))
        x1 = x1 + x2

        return x1

    '''
    Apply operator
    '''
    def forward(self, data):
        if self.adjoint:
            output = self.adjoint_op(data)
        else:
            output = self.forward_op(data)

        return output
