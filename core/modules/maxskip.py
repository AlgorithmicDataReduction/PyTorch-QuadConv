'''

'''

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm as spn

from typing import List

from .quadconv import QuadConvLayer
from .quadconv_blocks import PoolQuadConvBlock
from .conv_blocks import PoolConvBlock

'''
Encoder module
'''
class Encoder(nn.Module):

    def __init__(self,*,
            conv_type,
            conv_params,
            spatial_dim,
            latent_dim,
            stages,
            input_shape,
            forward_activation = nn.CELU,
            latent_activation = nn.CELU,
            **kwargs
        ):
        super().__init__()

        #activations
        self.activation1 = latent_activation()
        self.activation2 = latent_activation()

        arg_stack = self.package_args(conv_params, stages)

        if conv_type == 'standard':
            Block = PoolConvBlock
            init_layer = nn.Conv1d(**arg_stack[0])

        elif conv_type == 'quadrature':
            Block = PoolQuadConvBlock
            init_layer = QuadConvLayer(spatial_dim = spatial_dim, **arg_stack[0])

        #build network
        self.cnn = nn.Sequential()

        self.cnn.append(init_layer)

        for i in range(stages):
            self.cnn.append(Block(spatial_dim = spatial_dim,
                                    **arg_stack[i+1],
                                    activation1 = forward_activation if i!=1 else nn.Identity,
                                    activation2 = forward_activation
                                    ))

        self.conv_out_shape = self.cnn(torch.randn(size=input_shape)).shape

        self.flat = nn.Flatten(start_dim=1, end_dim=-1)

        self.linear = nn.Sequential()

        self.linear.append(spn(nn.Linear(self.conv_out_shape.numel(), latent_dim)))
        self.linear.append(self.activation1)
        self.linear.append(spn(nn.Linear(latent_dim, latent_dim)))
        self.linear.append(self.activation2)

        self.out_shape = self.linear(self.flat(torch.zeros(self.conv_out_shape)))

    def forward(self, x):
        x = self.cnn(x)
        x = self.flat(x)
        output = self.linear(x)

        return output

    def package_args(self, args : dict, stages : int):
        for key in args:
            if isinstance(args[key],List) and len(args[key]) == 1:
                args[key] = args[key]*(stages+1)

        arg_stack = [{ key : args[key][i] for key in args } for i in range(stages+1)]

        return arg_stack

'''
Decoder module
'''
class Decoder(nn.Module):

    def __init__(self,*,
            conv_type,
            conv_params,
            spatial_dim,
            latent_dim,
            stages,
            input_shape,
            forward_activation = nn.CELU,
            latent_activation = nn.CELU,
            **kwargs
        ):
        super().__init__()

        #activations
        self.activation1 = latent_activation()
        self.activation2 = latent_activation()

        arg_stack = self.package_args(conv_params, stages)

        if conv_type == 'standard':
            Block = PoolConvBlock
            init_layer = nn.Conv1d(**arg_stack[0] )

        elif conv_type == 'quadrature':
            Block = PoolQuadConvBlock
            init_layer = QuadConvLayer(spatial_dim = spatial_dim, **arg_stack[0])

        #build network
        self.unflat = nn.Unflatten(1, input_shape[1:])

        self.linear = nn.Sequential()

        self.linear.append(spn(nn.Linear(latent_dim, latent_dim)))
        self.linear.append(self.activation1)
        self.linear.append(spn(nn.Linear(latent_dim, input_shape.numel())))
        self.linear.append(self.activation2)

        self.linear(torch.zeros(latent_dim))

        self.cnn = nn.Sequential()

        for i in reversed(range(stages)):
            self.cnn.append(Block(spatial_dim = spatial_dim,
                                    **arg_stack[i+1],
                                    activation1 = forward_activation if i!=1 else nn.Identity,
                                    activation2 = forward_activation,
                                    adjoint = True
                                    ))

        self.cnn.append(init_layer)

    def forward(self, x):
        x = self.linear(x)
        x = self.unflat(x)
        output = self.cnn(x)

        return output

    def package_args(self, args : dict, stages : int):
        in_channels = args['in_channels']
        out_channels = args['out_channels']

        args['in_channels'] = out_channels
        args['out_channels'] = in_channels

        if 'num_points_in' in args and 'num_points_out' in args:

            N_in = args['num_points_in']
            N_out = args['num_points_out']

            args['num_points_in'] = N_out
            args['num_points_out'] = N_in

        for key in args:
            if isinstance(args[key],List) and len(args[key]) == 1:
                args[key] = args[key]*(stages+1)

        arg_stack = [{ key : args[key][i] for key in args } for i in range(stages+1)]

        return arg_stack
