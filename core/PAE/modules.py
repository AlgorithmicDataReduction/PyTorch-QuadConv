'''
Encoder and decoder modules based on the convolution block with skips and pooling.

Input:
    spatial_dim: spatial dimension of input data
    stages: number of convolution block stages
    params: convolution parameters
    latent_dim: dimension of latent representation
    input_shape: input data shape
    forward_activation: block activations
    latent_activation: mlp activations
    kwargs: keyword arguments for conv block
'''

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm as spn

from core.utilities import package_args, swap
from core.conv_blocks import PoolBlock
from core.point_blocks import PointBlock

################################################################################

class Encoder(nn.Module):

    def __init__(self,*,
            spatial_dim,
            stages,
            params,
            latent_dim,
            point_in_shape,
            forward_activation = nn.CELU,
            latent_activation = nn.CELU,
            **kwargs
        ):
        super().__init__()

        #block arguments
        arg_stack = package_args(stages, params)

        #point branch
        self.point_branch = nn.Sequential()

        for i in range(stages):
            self.point_branch.append(PointBlock(**arg_stack[i]))

        self.point_out_shape = self.point_branch(torch.zeros(point_in_shape)).shape

        self.point_branch.append(nn.Flatten(start_dim=1))
        self.point_branch.append(spn(nn.Linear(self.point_out_shape.numel(), latent_dim)))
        self.point_branch.append(latent_activation())

        #linear
        self.linear = nn.Sequential()

        self.linear.append(spn(nn.Linear(latent_dim, latent_dim)))
        self.linear.append(latent_activation())

        #dry run
        self.linear(torch.flatten(torch.zeros(latent_dim)))

    '''
    Forward
    '''
    def forward(self, x):

        z = self.linear(self.point_branch(x))

        return z

################################################################################

class Decoder(nn.Module):

    def __init__(self,*,
            spatial_dim,
            stages,
            params,
            latent_dim,
            point_in_shape,
            forward_activation = nn.CELU,
            latent_activation = nn.CELU,
            **kwargs
        ):
        super().__init__()

        #block arguments
        arg_stack = package_args(stages, swap(params), mirror=True)

        #linear
        self.linear = nn.Sequential()

        self.linear.append(spn(nn.Linear(latent_dim, latent_dim)))
        self.linear.append(latent_activation())

        #dry run
        self.linear(torch.zeros(latent_dim))

        #point branch
        self.point_branch = nn.Sequential()

        self.point_branch.append(spn(nn.Linear(latent_dim, point_in_shape.numel())))
        self.point_branch.append(latent_activation())
        self.point_branch.append(nn.Unflatten(1, point_in_shape[1:]))

        for i in range(stages):
            self.point_branch.append(PointBlock(**arg_stack[i],
                                        activation = forward_activation if i != stages-1 else nn.Identity))

    '''
    Forward
    '''
    def forward(self, z):

        x = self.point_branch(self.linear(z))

        return x
