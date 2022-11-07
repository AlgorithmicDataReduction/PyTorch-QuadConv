'''
Various standard convolution blocks.
'''

import numpy as np

import torch
import torch.nn as nn

'''
Convolution block with skip connections

Input:
    spatial_dim: spatial dimension of input data
    num_points_in: number of input points
    num_points_out: number of output points
    in_channels: input feature channels
    out_channels: output feature channels
    kernel_size: convolution kernel size
    use_bias: whether or not to use bias
    adjoint: downsample or upsample
    activation1:
    activation2:
'''
class ConvBlock(nn.Module):

    def __init__(self,*,
            spatial_dim,
            num_points_in,
            num_points_out,
            in_channels,
            out_channels,
            kernel_size = 3,
            adjoint = False,
            activation1 = nn.CELU,
            activation2 = nn.CELU,
            **kwargs
        ):
        super().__init__()

        #set attributes
        self.adjoint = adjoint

        #set layer types
        if spatial_dim == 1:
            Conv1 = nn.Conv1d

            if self.adjoint:
                Conv2 = nn.ConvTranspose1d
            else:
                Conv2 = nn.Conv1d

            Norm = nn.InstanceNorm1d

        elif spatial_dim == 2:
            Conv1 = nn.Conv2d

            if self.adjoint:
                Conv2 = nn.ConvTranspose2d
            else:
                Conv2 = nn.Conv2d

            Norm = nn.InstanceNorm2d

        elif spatial_dim == 3:
            Conv1 = nn.Conv3d

            if self.adjoint:
                Conv2 = nn.ConvTranspose3d
            else:
                Conv2 = nn.Conv3d

            Norm = nn.InstanceNorm3d

        #build convolution layers, normalization layers, and activations
        if self.adjoint:
            conv1_channel_num = out_channels
            stride = int(np.floor((num_points_out-1-(kernel_size-1))/(num_points_in-1)))
            self.conv2 = Conv2(in_channels,
                                out_channels,
                                kernel_size,
                                stride=stride,
                                output_padding=stride-1
                                )
        else:
            conv1_channel_num = in_channels
            stride = int(np.floor((num_points_in-1-(kernel_size-1))/(num_points_out-1)))
            self.conv2 = Conv2(in_channels,
                                out_channels,
                                kernel_size,
                                stride=stride
                                )

        self.conv1 = Conv1(conv1_channel_num,
                            conv1_channel_num,
                            kernel_size,
                            padding='same'
                            )
        self.norm1 = Norm(conv1_channel_num)
        self.activation1 = activation1()

        self.norm2 = Norm(out_channels)
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
Convolution block with skip connections and pooling.

Input:
    spatial_dim: spatial dimension of input data
    in_channels: input feature channels
    out_channels: output feature channels
    kernel_size: convolution kernel size
    adjoint: downsample or upsample
    use_bias: whether or not to use bias
    activation1:
    activation2:
'''

class PoolConvBlock(nn.Module):

    def __init__(self,*,
            spatial_dim,
            in_channels,
            out_channels,
            kernel_size = 3,
            adjoint = False,
            activation1 = nn.CELU,
            activation2 = nn.CELU,
            **kwargs
        ):
        super().__init__()

        #set attributes
        self.adjoint = adjoint

        #NOTE: channel flexibility can be added later with a 1x1 conv layer in the resampling operator
        assert in_channels == out_channels

        #make sure the user sets this as no default is provided
        assert spatial_dim > 0

        #set layer types
        layer_lookup = {
            1 : (nn.Conv1d, nn.InstanceNorm1d, nn.MaxPool1d),
            2 : (nn.Conv2d, nn.InstanceNorm2d, nn.MaxPool2d),
            3 : (nn.Conv3d, nn.InstanceNorm3d, nn.MaxPool3d),
        }

        Conv, Norm, Pool = layer_lookup[spatial_dim]

        #pooling or upsampling
        if self.adjoint:
            self.resample = nn.Upsample(scale_factor=2)
        else:
            self.resample = Pool(2)

        #build convolution layers, normalization layers, and activations
        self.conv1 = Conv(in_channels,
                            in_channels,
                            kernel_size,
                            padding='same'
                            )
        self.norm1 = Norm(in_channels)
        self.activation1 = activation1()

        self.conv2 = Conv(in_channels,
                            out_channels,
                            kernel_size,
                            padding='same'
                            )
        self.norm2 = Norm(out_channels)
        self.activation2 = activation2()

    '''
    Forward mode
    '''
    def forward_op(self, data):
        x = data

        x1 = self.conv1(x)
        x1 = self.activation1(self.norm1(x1))

        x2 = self.conv2(x1)
        x2 = self.activation2(self.norm2(x2) + x)

        return self.resample(x2)

    '''
    Adjoint mode
    '''
    def adjoint_op(self, data):
        x = self.resample(data)

        x1 = self.conv1(x)
        x1 = self.activation1(self.norm1(x1))

        x2 = self.conv2(x1)
        x2 = self.activation2(self.norm2(x2) + x)

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
