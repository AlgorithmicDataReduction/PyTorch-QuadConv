'''
Various quadrature convolution blocks.
'''

import numpy as np

import torch
import torch.nn as nn

from .quadconv import QuadConvLayer

'''
Quadrature convolution block with skip connections.

Input:
    num_points_in: number of input points
    num_points_out: number of output points
    in_channels: input feature channels
    out_channels: output feature channels
    use_bias: whether or not to use bias
    adjoint: downsample or upsample
    activation1:
    activation2:
    kwargs: quadrature convolution layer arguments
'''
class QuadConvBlock(nn.Module):

    def __init__(self,*,
            num_points_in,
            num_points_out,
            in_channels,
            out_channels,
            adjoint = False,
            activation1 = nn.CELU,
            activation2 = nn.CELU,
            **kwargs
        ):
        super().__init__()

        #set attributes
        self.adjoint = adjoint

        #block details
        if self.adjoint:
            conv1_point_num = num_points_out
            conv1_channel_num = out_channels
        else:
            conv1_point_num = num_points_in
            conv1_channel_num = in_channels

        #buid layers, normalizations, and activations
        self.conv1 = QuadConvLayer(num_points_in = conv1_point_num,
                                    num_points_out = conv1_point_num,
                                    in_channels = conv1_channel_num,
                                    out_channels = conv1_channel_num,
                                    **kwargs
                                    )
        self.norm1 = nn.BatchNorm1d(conv1_channel_num)
        self.activation1 = activation1()

        self.conv2 = QuadConvLayer(num_points_in = num_points_in,
                                    num_points_out = num_points_out,
                                    in_channels = in_channels,
                                    out_channels = out_channels,
                                    **kwargs
                                    )
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.activation2 = activation2()

        return

    def cache(self, nodes, weight_map):
        if self.adjoint:
            output_points = self.conv2.cache(nodes, weight_map)
            self.conv1.cache(output_points, weight_map)
        else:
            self.conv1.cache(nodes, weight_map)
            output_points = self.conv2.cache(nodes, weight_map)

        return output_points

    '''
    Forward mode
    '''
    def forward_op(self, data):
        x = data

        x1 = self.conv1(x)
        x1 = self.activation1(self.norm1(x1))
        x1 = x1 + x

        x2 = self.conv2(x1)
        x2 = self.activation2(self.norm2(x2))

        return x2

    '''
    Adjoint mode
    '''
    def adjoint_op(self, data):
        x = data

        x2 = self.conv2(x)
        x2 = self.activation2(self.norm2(x2))

        x1 = self.conv1(x2)
        x1 = self.activation1(self.norm1(x1))
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

################################################################################

'''
Quadrature convolution block with skip connections and pooling.

Input:
    spatial_dim: spatial dimension of input data
    num_points_in: number of input points
    num_points_out: number of output points
    in_channels: input feature channels
    out_channels: output feature channels
    adjoint: downsample or upsample
    activation1:
    activation2:
    kwargs: quadrature convolution layer arguments
'''
class PoolQuadConvBlock(nn.Module):

    def __init__(self,*,
            spatial_dim,
            num_points_in,
            num_points_out,
            in_channels,
            out_channels,
            adjoint = False,
            activation1 = nn.CELU,
            activation2 = nn.CELU,
            **kwargs,
        ):
        super().__init__()

        #NOTE: channel flexibility can be added later
        assert in_channels == out_channels

        #set attributes
        self.adjoint = adjoint

        #set pool type
        self.spatial_dim = spatial_dim

        layer_lookup = { 1 : (nn.MaxPool1d),
                         2 : (nn.MaxPool2d),
                         3 : (nn.MaxPool3d),
        }

        Pool = layer_lookup[spatial_dim]

        #pooling or upsamling
        if self.adjoint:
            self.resample = nn.Upsample(scale_factor=2)
        else:
            self.resample = Pool(2)

        #buid layers, normalizations, and activations
        self.conv1 = QuadConvLayer(spatial_dim = spatial_dim,
                                    num_points_in = num_points_in,
                                    num_points_out = num_points_in,
                                    in_channels = in_channels,
                                    out_channels = in_channels,
                                    **kwargs)
        self.batchnorm1 = nn.InstanceNorm1d(in_channels)
        self.activation1 = activation1()

        self.conv2 = QuadConvLayer(spatial_dim = spatial_dim,
                                    num_points_in = num_points_in,
                                    num_points_out = num_points_in,
                                    in_channels = in_channels,
                                    out_channels = out_channels,
                                    **kwargs)
        self.batchnorm2 = nn.InstanceNorm1d(out_channels)
        self.activation2 = activation2()

        return

    '''
    Forward mode
    '''
    def forward_op(self, data):
        x = data

        x1 = self.conv1(x)
        x1 = self.activation1(self.batchnorm1(x1))

        x2 = self.conv2(x1)
        x2 = self.activation2(self.batchnorm2(x2) + x)

        sq_shape = int(np.sqrt(x2.shape[-1]))

        dim_pack = [sq_shape] * self.spatial_dim

        return self.resample(x2.reshape(x2.shape[0], x2.shape[1], *dim_pack)).reshape(x2.shape[0], x2.shape[1], -1)

    '''
    Adjoint mode
    '''
    def adjoint_op(self, data):

        sq_shape = int(np.sqrt(data.shape[-1]))

        dim_pack = [sq_shape] * self.spatial_dim

        x = self.resample(data.reshape(data.shape[0], data.shape[1], *dim_pack)).reshape(data.shape[0], data.shape[1], -1)

        x1 = self.conv1(x)
        x1 = self.activation1(self.batchnorm1(x1))

        x2 = self.conv2(x1)
        x2 = self.activation2(self.batchnorm2(x2) + x)

        return x2

    '''
    Apply operator
    '''
    def forward(self, data):
        if self.adjoint:
            output = self.adjoint_op(data)
        else:
            output = self.forward_op(data)

        return output
