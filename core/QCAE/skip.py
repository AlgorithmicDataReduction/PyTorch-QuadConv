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
    kwargs: keyword arguments for quadconv block
'''

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm as spn

from core.utilities import package_args, swap
from core.quadconv_blocks import SkipBlock

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
        self.qcnn = nn.Sequential()

        for i in range(stages):
            self.qcnn.append(SkipBlock(**arg_stack[i], **kwargs))

        self.conv_out_shape = torch.Size((1, conv_params['out_channels'][-1], conv_params['out_points'][-1]))

        self.flat = nn.Flatten(start_dim=1, end_dim=-1)

        self.linear = nn.Sequential()

        self.linear.append(spn(nn.Linear(self.conv_out_shape.numel(), latent_dim)))
        self.linear.append(latent_activation())
        self.linear.append(spn(nn.Linear(latent_dim, latent_dim)))
        self.linear.append(latent_activation())

        self.linear(self.flat(torch.zeros(self.conv_out_shape)))

    def forward(self, mesh, x):
        _, x = self.qcnn((mesh, x))
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

        self.linear(torch.zeros(latent_dim))

        self.qcnn = nn.Sequential()

        for i in range(stages):
            self.qcnn.append(SkipBlock(**arg_stack[i],
                                        adjoint = True,
                                        activation1 = activation1 if i!=1 else nn.Identity,
                                        activation2 = activation2 if i!=1 else nn.Identity,
                                        **kwargs
                                        ))

    def forward(self, mesh, x):
        x = self.linear(x)
        x = self.unflat(x)
        _, output = self.qcnn((mesh, x))

        return output
