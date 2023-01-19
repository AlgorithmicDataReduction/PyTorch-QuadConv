'''
Various QuadConv-based blocks blocks.

Input:
    in_points: number of input points
    out_points: number of output points
    in_channels: input feature channels
    out_channels: output feature channels
    use_bias: whether or not to use bias
    adjoint: downsample or upsample
    activation1:
    activation2:
    kwargs: quadrature convolution layer arguments
'''

import numpy as np

import torch
import torch.nn as nn

from core.torch_quadconv import QuadConv

################################################################################

class SkipBlock(nn.Module):

    def __init__(self,*,
            in_points,
            out_points,
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
            conv1_point_num = out_points
            conv1_channel_num = out_channels
        else:
            conv1_point_num = in_points
            conv1_channel_num = in_channels

        #buid layers, normalizations, and activations
        self.conv1 = QuadConv(in_points = conv1_point_num,
                                out_points = conv1_point_num,
                                in_channels = conv1_channel_num,
                                out_channels = conv1_channel_num,
                                output_same = True,
                                **kwargs
                                )
        self.norm1 = nn.BatchNorm1d(conv1_channel_num)
        self.activation1 = activation1()

        self.conv2 = QuadConv(in_points = in_points,
                                out_points = out_points,
                                in_channels = in_channels,
                                out_channels = out_channels,
                                **kwargs
                                )
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.activation2 = activation2()

        return

    '''
    Forward mode
    '''
    def forward_op(self, handler, data):
        x = data

        x1 = self.conv1(handler, x)
        x1 = self.activation1(self.norm1(x1))
        x1 = x1 + x

        x2 = self.conv2(handler, x1)
        x2 = self.activation2(self.norm2(x2))

        return x2

    '''
    Adjoint mode
    '''
    def adjoint_op(self, handler, data):
        x = data

        x2 = self.conv2(handler, x)
        x2 = self.activation2(self.norm2(x2))

        x1 = self.conv1(handler, x2)
        x1 = self.activation1(self.norm1(x1))
        x1 = x1 + x2

        return x1

    '''
    Apply operator
    '''
    def forward(self, input):

        handler, data = input[0], input[1]

        if self.adjoint:
            data = self.adjoint_op(handler, data)
        else:
            data = self.forward_op(handler, data)

        return (handler, data)

################################################################################

class PoolBlock(nn.Module):

    def __init__(self,*,
            spatial_dim,
            in_points,
            out_points,
            in_channels,
            out_channels,
            adjoint = False,
            activation1 = nn.CELU,
            activation2 = nn.CELU,
            **kwargs
        ):
        super().__init__()

        #NOTE: channel flexibility can be added later
        assert in_channels == out_channels, f"In channels must match out channels to maintain compatibility with the skip connection"

        #set attributes
        self.adjoint = adjoint

        if self.adjoint:
            self.out_points = in_points * 2**(spatial_dim)
            block_points = in_points * 2**(spatial_dim)
        elif not self.adjoint:
            self.out_points = in_points / 2**(spatial_dim)
            block_points = in_points


        assert self.out_points == out_points, f"User assigned number of output points ({out_points}) does not match the actual number ({self.out_points})"


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

        #build layers, normalizations, and activations
        self.conv1 = QuadConv(spatial_dim = spatial_dim,
                                in_points = block_points,
                                out_points = block_points,
                                in_channels = in_channels,
                                out_channels = in_channels,
                                output_same = True,
                                **kwargs)
        self.batchnorm1 = nn.InstanceNorm1d(in_channels)
        self.activation1 = activation1()

        self.conv2 = QuadConv(spatial_dim = spatial_dim,
                                in_points = block_points,
                                out_points = block_points,
                                in_channels = in_channels,
                                out_channels = out_channels,
                                output_same = True,
                                **kwargs)
        self.batchnorm2 = nn.InstanceNorm1d(out_channels)
        self.activation2 = activation2()

	
        return

    '''
    Forward mode
    '''
    def forward_op(self, handler, data):
        x = data

        x1 = self.conv1(handler, x)
        x1 = self.activation1(self.batchnorm1(x1))

        x2 = self.conv2(handler, x1)
        x2 = self.activation2(self.batchnorm2(x2) + x)

        sq_shape = int(np.sqrt(x2.shape[-1]))

        dim_pack = [sq_shape] * self.spatial_dim

        output = self.resample(x2.reshape(x2.shape[0], x2.shape[1], *dim_pack)).reshape(x2.shape[0], x2.shape[1], -1)

        handler.step()

        return output

    '''
    Adjoint mode
    '''
    def adjoint_op(self, handler, data):

        handler.step()

        sq_shape = int(np.sqrt(data.shape[-1]))

        dim_pack = [sq_shape] * self.spatial_dim

        x = self.resample(data.reshape(data.shape[0], data.shape[1], *dim_pack)).reshape(data.shape[0], data.shape[1], -1)

        x1 = self.conv1(handler, x)
        x1 = self.activation1(self.batchnorm1(x1))

        x2 = self.conv2(handler, x1)
        x2 = self.activation2(self.batchnorm2(x2) + x)

        return x2

    '''
    Apply operator
    '''
    def forward(self, input):

        handler, data = input[0], input[1]

        if self.adjoint:
            data = self.adjoint_op(handler, data)
        else:
            data = self.forward_op(handler, data)

        return (handler, data)
