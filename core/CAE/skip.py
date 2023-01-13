'''
Encoder and decoder modules based on the convolution block with skips.

Input:
    stages: number of convolution block stages
    conv_params: convolution parameters
    latent_dim: dimension of latent representation
    input_shape: input data shape
    latent_activation: mlp activation
    activation1: block activation 1
    activation2: block activation 2
    kwargs: keyword arguments for conv block
'''

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm as spn

from core.utils import package_args, swap
from core.conv_blocks import SkipBlock

################################################################################

class Encoder(nn.Module):

    def __init__(self,*,
            stages,
            conv_params,
            latent_dim,
            input_shape,
            latent_activation = nn.CELU,
            **kwargs
        ):
        super().__init__()

        #block arguments
        arg_stack = package_args(stages, conv_params)

        #build network
        self.cnn = nn.Sequential()

        for i in range(stages):
            self.cnn.append(SkipBlock(**arg_stack[i], **kwargs))

        self.conv_out_shape = self.cnn(torch.zeros(input_shape)).shape

        self.flat = nn.Flatten(start_dim=1, end_dim=-1)

        self.linear = nn.Sequential()

        self.linear.append(spn(nn.Linear(self.conv_out_shape.numel(), latent_dim)))
        self.linear.append(latent_activation())
        self.linear.append(spn(nn.Linear(latent_dim, latent_dim)))
        self.linear.append(latent_activation())

        #dry run
        self.linear(self.flat(torch.zeros(self.conv_out_shape)))

    def forward(self, x):
        x = self.cnn(x)
        x = self.flat(x)
        output = self.linear(x)

        return output

################################################################################

class Decoder(nn.Module):

    def __init__(self,*,
            stages,
            conv_params,
            latent_dim,
            input_shape,
            latent_activation = nn.CELU,
            activation1 = nn.CELU,
            activation2 = nn.CELU,
            **kwargs
        ):
        super().__init__()

        #block arguments
        arg_stack = package_args(stages, swap(conv_params), mirror=True)

        #build network
        self.unflat = nn.Unflatten(1, input_shape[1:])

        self.linear = nn.Sequential()

        self.linear.append(spn(nn.Linear(latent_dim, latent_dim)))
        self.linear.append(latent_activation())
        self.linear.append(spn(nn.Linear(latent_dim, input_shape.numel())))
        self.linear.append(latent_activation())

        #dry run
        self.linear(torch.zeros(latent_dim))

        self.cnn = nn.Sequential()

        for i in range(stages):
            self.cnn.append(SkipBlock(**arg_stack[i],
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
