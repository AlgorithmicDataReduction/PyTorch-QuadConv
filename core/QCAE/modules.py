'''
Encoder and decoder modules based on the convolution block with skips and pooling.
'''

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm as spn

from torch_quadconv import QuadConv

from core.utilities import package_args
from core.quadconv_blocks import PoolBlock

'''
Encoder module.

Input:
    spatial_dim: spatial dimension of input data
    stages: number of convolution stages
    conv_params: convolution parameters
    latent_dim: dimension of latent representation
    input_shape: input data shape
    forward_activation: block activations
    latent_activation: mlp activations
'''
class Encoder(nn.Module):

    def __init__(self,*,
            spatial_dim,
            stages,
            conv_params,
            latent_dim,
            input_shape,
            forward_activation = nn.CELU,
            latent_activation = nn.CELU
        ):
        super().__init__()

        #block arguments
        arg_stack = package_args(stages, conv_params)

        #build network
        self.init_layer = QuadConv(spatial_dim = spatial_dim, **arg_stack[0])

        self.qcnn = nn.Sequential()

        for i in range(1, stages):
            self.qcnn.append(PoolBlock(spatial_dim = spatial_dim,
                                        **arg_stack[i],
                                        activation1 = forward_activation,
                                        activation2 = forward_activation
                                        ))

        self.conv_out_shape = torch.Size((1, conv_params['out_channels'][-1], int(conv_params['num_points_out'][-1]/4)))

        self.flat = nn.Flatten(start_dim=1, end_dim=-1)

        self.linear = nn.Sequential()

        self.linear.append(spn(nn.Linear(self.conv_out_shape.numel(), latent_dim)))
        self.linear.append(latent_activation())
        self.linear.append(spn(nn.Linear(latent_dim, latent_dim)))
        self.linear.append(latent_activation())

        self.out_shape = self.linear(self.flat(torch.zeros(self.conv_out_shape)))

    '''
    Forward
    '''
    def forward(self, mesh, x):
        x = self.init_layer(mesh, x)
        x = self.qcnn(mesh, x)
        x = self.flat(x)
        output = self.linear(x)

        return output

'''
Decoder module

Input:
    spatial_dim: spatial dimension of input data
    stages: number of convolution stages
    conv_params: convolution parameters
    latent_dim: dimension of latent representation
    input_shape: input data shape
    forward_activation: block activations
    latent_activation: mlp activations
'''
class Decoder(nn.Module):

    def __init__(self,*,
            spatial_dim,
            stages,
            conv_params,
            latent_dim,
            input_shape,
            forward_activation = nn.CELU,
            latent_activation = nn.CELU
        ):
        super().__init__()

        #block arguments
        arg_stack = package_args(stages, conv_params)

        #build network
        self.unflat = nn.Unflatten(1, input_shape[1:])

        self.linear = nn.Sequential()

        self.linear.append(spn(nn.Linear(latent_dim, latent_dim)))
        self.linear.append(latent_activation())
        self.linear.append(spn(nn.Linear(latent_dim, input_shape.numel())))
        self.linear.append(latent_activation())

        self.linear(torch.zeros(latent_dim))

        self.qcnn = nn.Sequential()

        for i in range(stages, 0, -1):
            self.qcnn.append(PoolBlock(spatial_dim = spatial_dim,
                                        **arg_stack[i],
                                        activation1 = forward_activation,
                                        activation2 = forward_activation,
                                        adjoint = True
                                        ))

        self.init_layer = QuadConv(spatial_dim = spatial_dim, **arg_stack[0])

    '''
    Forward
    '''
    def forward(self, mesh, x):
        x = self.linear(x)
        x = self.unflat(x)
        x = self.qcnn(mesh, x)
        output = self.init_layer(mesh, x)

        return output
