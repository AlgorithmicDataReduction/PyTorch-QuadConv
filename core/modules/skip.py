'''
Encoder and decoder modules based on the convolution block with skips.
'''

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm as spn

from .quadconv_blocks import QuadConvBlock
from .conv_blocks import ConvBlock

'''
Encoder module

Input:
    conv_type: convolution type
    latent_dim: dimension of latent representation
    point_seq: number of points at each block stage
    channel_seq: number of channels at each block stage
    input_shape: input data shape
    latent_activation: mlp activation
    kwargs: keyword arguments for quadconv block
'''
class Encoder(nn.Module):
    def __init__(self,*,
            conv_type,
            latent_dim,
            point_seq,
            channel_seq,
            input_shape,
            latent_activation = nn.CELU,
            **kwargs
        ):
        super().__init__()

        #establish block type and output shape
        if conv_type == 'standard':
            Block = ConvBlock
        elif conv_type == 'quadrature':
            Block = QuadConvBlock
        else:
            raise ValueError(f'Convolution type {conv_type} not supported.')

        #build network
        self.cnn = nn.Sequential()

        for i in range(len(point_seq)-1):
            self.cnn.append(Block(num_points_in = point_seq[i],
                                    num_points_out = point_seq[i+1],
                                    in_channels = channel_seq[i],
                                    out_channels = channel_seq[i+1],
                                    **kwargs
                                    ))

        self.conv_out_shape = self.cnn(torch.zeros(input_shape)).shape

        self.flat = nn.Flatten(start_dim=1, end_dim=-1)

        self.linear = nn.Sequential()

        self.linear.append(spn(nn.Linear(self.conv_out_shape.numel(), latent_dim)))
        self.linear.append(latent_activation())
        self.linear.append(spn(nn.Linear(latent_dim, latent_dim)))
        self.linear.append(latent_activation())

        self.linear(self.flat(torch.zeros(self.conv_out_shape)))

    def forward(self, x):
        x = self.cnn(x)
        x = self.flat(x)
        output = self.linear(x)

        return output

'''
Decoder module

Input:
    conv_type: convolution type
    latent_dim: dimension of latent representation
    point_seq: number of points at each block stage
    channel_seq: number of channels at each block stage
    input_shape: input data shape
    latent_activation: mlp activation
    activation1: block activation 1
    activation2: block activation 2
    kwargs: keyword arguments for quadconv block
'''
class Decoder(nn.Module):
    def __init__(self,*,
            conv_type,
            latent_dim,
            point_seq,
            channel_seq,
            input_shape,
            latent_activation = nn.CELU,
            activation1 = nn.CELU,
            activation2 = nn.CELU,
            **kwargs
        ):
        super().__init__()

        #establish block type
        if conv_type == 'standard':
            Block = ConvBlock
        elif conv_type == 'quadrature':
            Block = QuadConvBlock
        else:
            raise ValueError(f'Convolution type {conv_type} not supported.')

        #build network
        self.unflat = nn.Unflatten(1, input_shape[1:])

        self.linear = nn.Sequential()

        self.linear.append(spn(nn.Linear(latent_dim, latent_dim)))
        self.linear.append(latent_activation())
        self.linear.append(spn(nn.Linear(latent_dim, input_shape.numel())))
        self.linear.append(latent_activation())

        self.linear(torch.zeros(latent_dim))

        self.cnn = nn.Sequential()

        for i in range(len(point_seq)-1, 0, -1):
            self.cnn.append(Block(num_points_in = point_seq[i],
                                    num_points_out = point_seq[i-1],
                                    in_channels = channel_seq[i],
                                    out_channels = channel_seq[i-1],
                                    adjoint = True,
                                    activation1 = activation1 if i!=1 else nn.Identity,
                                    activation2 = activation2 if i!=1 else nn.Identity,
                                    **kwargs
                                    ))

    def forward(self, x):
        x = self.linear(x)
        x = self.unflat(x)
        output = self.cnn(x)

        return output
