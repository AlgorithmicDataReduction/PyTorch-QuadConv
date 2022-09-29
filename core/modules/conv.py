'''

'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
QuadConvLayer block

Input:
    point_dim: space dimension
    num_points_in: number of input points
    num_points_out: number of output points
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
                    point_dim,
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

        if point_dim == 1:
            Conv1 = nn.Conv1d

            if self.adjoint:
                Conv2 = nn.ConvTranspose1d
            else:
                Conv2 = nn.Conv1d

            Norm = nn.InstanceNorm1d

        elif point_dim == 2:
            Conv1 = nn.Conv2d

            if self.adjoint:
                Conv2 = nn.ConvTranspose2d
            else:
                Conv2 = nn.Conv2d

            Norm = nn.InstanceNorm2d

        elif point_dim == 3:
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
