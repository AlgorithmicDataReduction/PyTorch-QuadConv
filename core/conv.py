'''

'''

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
QuadConvLayer block

Input:
    point_dim: space dimension
    channels_in: input feature channels
    channels_out: output feature channels
    N_in: number of input points
    N_out: number of output points
    adjoint: downsample or upsample
    use_bias: add bias term to output of layer
    activation1:
    activation2:
'''
class ConvBlock(nn.Module):
    def __init__(self,
                    point_dim,
                    channels_in,
                    channels_out,
                    N_in,
                    N_out,
                    kernel_size = 3,
                    stride = 1,
                    padding = 0,
                    dilation = 1,
                    adjoint = False,
                    use_bias = False,
                    activation1 = nn.CELU(alpha=1),
                    activation2 = nn.CELU(alpha=1)
                    ):
        super().__init__()

        self.adjoint = adjoint
        self.activation1 = activation1
        self.activation2 = activation2

        if point_dim == 2:
            if self.adjoint:
                Conv = nn.ConvTranspose2d
            else:
                Conv = nn.Conv2d
        elif point_dim == 3:
            if self.adjoint:
                Conv = nn.ConvTranspose3d
            else:
                Conv = nn.Conv3d

        if self.adjoint:
            conv1_channel_num = channels_out
        else:
            conv1_channel_num = channels_in

        self.conv1 = Conv(conv1_channel_num,
                            conv1_channel_num,
                            kernel_size,
                            stride=stride,
                            padding=padding
                            )
        self.batchnorm1 = torch.nn.InstanceNorm1d(conv1_channel_num)

        self.conv2 = Conv(channels_in,
                            channels_out,
                            kernel_size,
                            stride=stride,
                            padding=padding
                            )
        self.batchnorm2 = torch.nn.InstanceNorm1d(channels_out)

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
