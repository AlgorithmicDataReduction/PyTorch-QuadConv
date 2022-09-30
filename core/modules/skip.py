'''

'''

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm as spn

from .quadconv import QuadConvBlock
from .conv import ConvBlock

'''
Encoder module
'''
class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        #establish block type
        conv_type = kwargs.pop('conv_type')
        if conv_type == 'standard':
            Block = ConvBlock
        elif conv_type == 'quadrature':
            Block = QuadConvBlock

        #specific args
        point_dim = kwargs.pop('point_dim')
        latent_dim = kwargs.pop('latent_dim')
        point_seq = kwargs.pop('point_seq')
        channel_seq = kwargs.pop('channel_seq')
        input_shape = kwargs.pop('input_shape')

        #activations
        forward_activation = kwargs.pop('forward_activation')
        latent_activation = kwargs.pop('latent_activation')
        self.activation1 = latent_activation()
        self.activation2 = latent_activation()

        #build network
        self.cnn = nn.Sequential()

        for i in range(len(point_seq)-1):
            self.cnn.append(Block(point_dim,
                                    point_seq[i],
                                    point_seq[i+1],
                                    channel_seq[i],
                                    channel_seq[i+1],
                                    activation1 = forward_activation(),
                                    activation2 = forward_activation(),
                                    **kwargs
                                    ))

        if conv_type == 'standard':
            self.conv_out_shape = self.cnn(torch.zeros(input_shape)).shape
        elif conv_type == 'quadrature':
            self.conv_out_shape = torch.Size((1, channel_seq[-1], point_seq[-1]))
        else:
            raise ValueError('Convolution type "{conv_type}" is not valid.')

        self.flat = nn.Flatten(start_dim=1, end_dim=-1)

        self.linear = nn.Sequential()

        self.linear.append(spn(nn.Linear(self.conv_out_shape.numel(), latent_dim)))
        self.linear.append(self.activation1)
        self.linear.append(spn(nn.Linear(latent_dim, latent_dim)))
        self.linear.append(self.activation2)

        self.linear(self.flat(torch.zeros(self.conv_out_shape)))

    def forward(self, x):
        x = self.cnn(x)
        x = self.flat(x)
        output = self.linear(x)

        return output

'''
Decoder module
'''
class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        #establish block type
        conv_type = kwargs.pop('conv_type')
        if conv_type == 'standard':
            Block = ConvBlock
        elif conv_type == 'quadrature':
            Block = QuadConvBlock

        #specific args
        point_dim = kwargs.pop('point_dim')
        latent_dim = kwargs.pop('latent_dim')
        point_seq = kwargs.pop('point_seq')
        channel_seq = kwargs.pop('channel_seq')
        input_shape = kwargs.pop('input_shape')

        #activations
        forward_activation = kwargs.pop('forward_activation')
        latent_activation = kwargs.pop('latent_activation')
        self.activation1 = latent_activation()
        self.activation2 = latent_activation()

        #build network
        self.unflat = nn.Unflatten(1, input_shape[1:])

        self.linear = nn.Sequential()

        self.linear.append(spn(nn.Linear(latent_dim, latent_dim)))
        self.linear.append(self.activation1)
        self.linear.append(spn(nn.Linear(latent_dim, input_shape.numel())))
        self.linear.append(self.activation2)

        self.linear(torch.zeros(latent_dim))

        self.cnn = nn.Sequential()

        for i in range(len(point_seq)-1, 0, -1):
            self.cnn.append(Block(point_dim,
                                    point_seq[i],
                                    point_seq[i-1],
                                    channel_seq[i],
                                    channel_seq[i-1],
                                    adjoint = True,
                                    activation1 = forward_activation() if i!=1 else nn.Identity(),
                                    activation2 = forward_activation(),
                                    **kwargs
                                    ))

    def forward(self, x):
        x = self.linear(x)
        x = self.unflat(x)
        output = self.cnn(x)

        return output
