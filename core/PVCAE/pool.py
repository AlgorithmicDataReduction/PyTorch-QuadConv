'''
Encoder and decoder modules based on the convolution block with skips and pooling
and shared MLP point-based block.

Input:
    spatial_dim: spatial dimension of input data
    stages: number of convolution block stages
    conv_params: convolution parameters
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
from core.conv_blocks import PoolBlock, PointBlock

################################################################################

class Encoder(nn.Module):

    def __init__(self,*,
            spatial_dim,
            conv_stages,
            point_stages,
            conv_params,
            point_params,
            conv_latent_dim,
            point_latent_dim,
            conv_in_shape,
            point_in_shape,
            forward_activation = nn.CELU,
            latent_activation = nn.CELU,
            **kwargs
        ):
        super().__init__()

        #block arguments
        conv_arg_stack = package_args(conv_stages+1, conv_params)
        point_arg_stack = package_args(point_stages, point_params)

        #layer switch
        layer_lookup = {
            1 : (nn.Conv1d),
            2 : (nn.Conv2d),
            3 : (nn.Conv3d),
            }

        Conv = layer_lookup[spatial_dim]

        #conv branch
        self.cnn = nn.Sequential()

        init_layer = Conv(**conv_arg_stack[0])

        self.cnn.append(init_layer)

        for i in range(1, conv_stages):
            self.cnn.append(PoolBlock(spatial_dim = spatial_dim,
                                        **conv_arg_stack[i],
                                        activation1 = forward_activation,
                                        activation2 = forward_activation,
                                        **kwargs
                                        ))

        self.conv_out_shape = self.cnn(torch.zeros(conv_in_shape)).shape

        self.cnn.append(nn.Flatten(start_dim=1, end_dim=-1))
        self.cnn.append(spn(nn.Linear(self.conv_out_shape.numel(), conv_latent_dim)))
        self.cnn.append(latent_activation())
        self.cnn.append(spn(nn.Linear(conv_latent_dim, conv_latent_dim)))
        self.cnn.append(latent_activation())

        #point branch
        self.point_branch = nn.Sequential()

        for i in range(point_stages):
            self.point_branch.append(PointBlock(**point_arg_stack[i]))

        self.point_out_shape = self.point_branch(torch.zeros(point_in_shape)).shape

        self.point_branch.append(nn.Flatten(start_dim=1))
        self.point_branch.append(spn(nn.Linear(self.point_out_shape.numel(), point_latent_dim)))
        self.point_branch.append(latent_activation())
        self.point_branch.append(spn(nn.Linear(point_latent_dim, point_latent_dim)))
        self.point_branch.append(latent_activation())

    '''
    Forward
    '''
    def forward(self, pf, vf):

        pf = self.point_branch(pf)
        vf = self.cnn(vf)

        output = torch.cat((pf, vf), dim=1)

        return output

################################################################################

class Decoder(nn.Module):

    def __init__(self,*,
            spatial_dim,
            conv_stages,
            point_stages,
            conv_params,
            point_params,
            conv_latent_dim,
            point_latent_dim,
            conv_in_shape,
            point_in_shape,
            forward_activation = nn.CELU,
            latent_activation = nn.CELU,
            **kwargs
        ):
        super().__init__()

        self.latent_radix = point_latent_dim

        #block arguments
        conv_arg_stack = package_args(conv_stages+1, swap(conv_params), mirror=True)
        point_arg_stack = package_args(point_stages, swap(point_params), mirror=True)

        #layer switch
        layer_lookup = {
            1 : (nn.Conv1d),
            2 : (nn.Conv2d),
            3 : (nn.Conv3d),
            }

        Conv = layer_lookup[spatial_dim]

        #point branch
        self.point_branch = nn.Sequential()

        self.point_branch.append(spn(nn.Linear(point_latent_dim, point_latent_dim)))
        self.point_branch.append(latent_activation())
        self.point_branch.append(spn(nn.Linear(point_latent_dim, point_in_shape.numel())))
        self.point_branch.append(latent_activation())
        self.point_branch.append(nn.Unflatten(1, point_in_shape[1:]))

        for i in range(point_stages):
            self.point_branch.append(PointBlock(**point_arg_stack[i],
                                        activation = forward_activation if i != point_stages-1 else nn.Identity))

        #conv branch
        self.cnn = nn.Sequential()

        self.cnn.append(spn(nn.Linear(conv_latent_dim, conv_latent_dim)))
        self.cnn.append(latent_activation())
        self.cnn.append(spn(nn.Linear(conv_latent_dim, conv_in_shape.numel())))
        self.cnn.append(latent_activation())
        self.cnn.append(nn.Unflatten(1, conv_in_shape[1:]))

        for i in range(conv_stages-1):
            self.cnn.append(PoolBlock(spatial_dim = spatial_dim,
                                        **conv_arg_stack[i],
                                        activation1 = forward_activation,
                                        activation2 = forward_activation,
                                        adjoint = True,
                                        **kwargs
                                        ))

        init_layer = Conv(**conv_arg_stack[-1])

        self.cnn.append(init_layer)

    '''
    Forward
    '''
    def forward(self, z):
        zp, zv = z[:,:self.latent_radix], z[:,self.latent_radix:]

        pf = self.point_branch(zp)
        vf = self.cnn(zv)

        return pf, vf
