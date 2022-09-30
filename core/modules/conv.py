'''

'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
ConvLayer block

Input:
    dimension: space dimension
    channels_in: input feature channels
    channels_out: output feature channels
    kernel_size: convolution kernel size
    use_bias: add bias term to output of layer
    adjoint: downsample or upsample
    activation1:
    activation2:
'''
class ConvBlock(nn.Module):
    def __init__(self,
                    dimension,
                    num_points_in,
                    num_points_out,
                    channels_in,
                    channels_out,
                    kernel_size = 3,
                    use_bias = False,
                    adjoint = False,
                    activation1 = nn.CELU(alpha=1),
                    activation2 = nn.CELU(alpha=1)
                    ):
        super().__init__()

        self.adjoint = adjoint
        self.activation1 = activation1
        self.activation2 = activation2

        if dimension == 1:

            Conv1 = nn.Conv1d

            if self.adjoint:
                Conv2 = nn.ConvTranspose1d
            else:
                Conv2 = nn.Conv1d

            Norm = nn.InstanceNorm1d

        elif dimension == 2:
            Conv1 = nn.Conv2d

            if self.adjoint:
                Conv2 = nn.ConvTranspose2d
            else:
                Conv2 = nn.Conv2d

            Norm = nn.InstanceNorm2d

        elif dimension == 3:
            Conv1 = nn.Conv3d

            if self.adjoint:
                Conv2 = nn.ConvTranspose3d
            else:
                Conv2 = nn.Conv3d

            Norm = nn.InstanceNorm3d

        if self.adjoint:
            conv1_channel_num = channels_out
            stride = int(np.floor((um_points_out-1-(kernel_size-1))/(num_points_in-1)))
            self.conv2 = Conv2(channels_in,
                                channels_out,
                                kernel_size,
                                stride=stride,
                                output_padding=stride-1
                                )
        else:
            conv1_channel_num = channels_in
            stride = int(np.floor((num_points_in-1-(kernel_size-1))/(num_points_out-1)))
            self.conv2 = Conv2(channels_in,
                                channels_out,
                                kernel_size,
                                stride=stride
                                )

        self.conv1 = Conv1(conv1_channel_num,
                            conv1_channel_num,
                            kernel_size,
                            padding='same'
                            )

        self.batchnorm1 = Norm(conv1_channel_num)

        self.batchnorm2 = Norm(channels_out)

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

'''
ConvLayer + Pooling block

Input:
    dimension: space dimension
    channels_in: input feature channels
    channels_out: output feature channels
    N_in: number of input points
    N_out: number of output points
    adjoint: downsample or upsample
    use_bias: add bias term to output of layer
    activation1:
    activation2:
'''

class PoolConvBlock(nn.Module):

    __allowed = (
                  'kernel_size',
                  'adjoint',
                  'use_bias',
                  'activation1',
                  'activation2',
                  )

    kernel_size = 3
    adjoint = False
    use_bias = True
    activation1 = nn.CELU(alpha=1)
    activation2 = nn.CELU(alpha=1)

    def __init__(self, dimension,
                       in_channels,
                       out_channels,
                       **kwargs
                       ):
        super().__init__()

        for key, value in kwargs.items():
            if key in self.__allowed:
                setattr(self, key,value)

        # channel flexibility can be added later with a 1x1 conv layer in the resampling operator
        assert in_channels == out_channels

        # make sure the user sets this as no default is provided
        assert dimension > 0

        layer_lookup = { 1 : (nn.Conv1d, nn.InstanceNorm1d, nn.MaxPool1d),
                         2 : (nn.Conv2d, nn.InstanceNorm2d, nn.MaxPool2d),
                         3 : (nn.Conv3d, nn.InstanceNorm3d, nn.MaxPool3d),
        }

        Conv, Norm, Pool = layer_lookup[dimension]

        if self.adjoint:
            self.resample = nn.Upsample(scale_factor=2)
        else:
            self.resample = Pool(2)


        self.conv1 = Conv(in_channels,
                            in_channels,
                            self.kernel_size,
                            padding='same'
                            )

        self.conv2 = Conv(in_channels,
                            out_channels,
                            self.kernel_size,
                            padding='same'
                            )

        self.batchnorm1 = Norm(in_channels)

        self.batchnorm2 = Norm(out_channels)

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
