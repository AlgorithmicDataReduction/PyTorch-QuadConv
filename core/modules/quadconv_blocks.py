'''

'''

import torch
import torch.nn as nn

from .quadconv import QuadConvLayer

'''
QuadConvLayer block

Input:
    num_points_in: number of input points
    num_points_out: number of output points
    in_channels: input feature channels
    out_channels: output feature channels
    use_bias: add bias term to output of layer
    adjoint: downsample or upsample
    activation1:
    activation2:
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

        #set hyperparameters
        self.adjoint = adjoint

        #buid block pieces
        if self.adjoint:
            conv1_point_num = num_points_out
            conv1_channel_num = out_channels
        else:
            conv1_point_num = num_points_in
            conv1_channel_num = in_channels

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
QuadConvLayer + Pooling block

Input:
    in_channels: input feature channels
    out_channels: output feature channels
    adjoint: downsample or upsample
    activation1:
    activation2:
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

        # channel flexibility can be added later
        assert in_channels == out_channels

        #set hyperparameters
        self.adjoint = adjoint

        layer_lookup = { 1 : (nn.MaxPool1d),
                         2 : (nn.MaxPool2d),
                         3 : (nn.MaxPool3d),
        }

        Pool = layer_lookup[spatial_dim]

        if self.adjoint:
            self.resample = nn.Upsample(scale_factor=2)
        else:
            self.resample = Pool(2)

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

        return self.resample(x2)

    '''
    Adjoint mode
    '''
    def adjoint_op(self, data):
        x = self.resample(data)

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
