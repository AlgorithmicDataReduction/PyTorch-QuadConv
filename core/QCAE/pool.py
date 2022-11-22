'''
Encoder and decoder modules based on the convolution block with skips and pooling.

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

from torch_quadconv import QuadConv

from core.utilities import package_args, swap
from core.quadconv_blocks import PoolBlock

################################################################################

class Encoder(nn.Module):

    def __init__(self,*,
            spatial_dim,
            stages,
            conv_params,
            latent_dim,
            input_shape,
            forward_activation = nn.CELU,
            latent_activation = nn.CELU,
            **kwargs
        ):
        super().__init__()

        #block arguments
        arg_stack = package_args(stages+1, conv_params)

        #build network
        self.init_layer = QuadConv(spatial_dim = spatial_dim, **arg_stack[0])

        self.qcnn = nn.Sequential()

        for i in range(1, stages+1):
            self.qcnn.append(PoolBlock(spatial_dim = spatial_dim,
                                        **arg_stack[i],
                                        activation1 = forward_activation,
                                        activation2 = forward_activation
                                        ))

        self.conv_out_shape = torch.Size((1, conv_params['out_channels'][-1], int(conv_params['out_points'][-1])))

        self.flat = nn.Flatten(start_dim=1, end_dim=-1)

        self.linear = nn.Sequential()

        self.linear.append(spn(nn.Linear(self.conv_out_shape.numel(), latent_dim)))
        self.linear.append(latent_activation())
        self.linear.append(spn(nn.Linear(latent_dim, latent_dim)))
        self.linear.append(latent_activation())

        #dry run
        self.out_shape = self.linear(self.flat(torch.zeros(self.conv_out_shape)))

    '''
    Forward
    '''
    def forward(self, mesh, x):
        x = self.init_layer(mesh, x)
        _, x = self.qcnn((mesh, x))
        x = self.flat(x)
        output = self.linear(x)

        return output

################################################################################

class Decoder(nn.Module):

    def __init__(self,*,
            spatial_dim,
            stages,
            conv_params,
            latent_dim,
            input_shape,
            forward_activation = nn.CELU,
            latent_activation = nn.CELU,
            **kwargs
        ):
        super().__init__()

        #block arguments
        arg_stack = package_args(stages+1, swap(conv_params), mirror=True)

        #build network
        self.unflat = nn.Unflatten(1, input_shape[1:])

        self.linear = nn.Sequential()

        self.linear.append(spn(nn.Linear(latent_dim, latent_dim)))
        self.linear.append(latent_activation())
        self.linear.append(spn(nn.Linear(latent_dim, input_shape.numel())))
        self.linear.append(latent_activation())

        #dry run
        self.linear(torch.zeros(latent_dim))

        self.qcnn = nn.Sequential()

        for i in range(stages-1):
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
        _, x = self.qcnn((mesh, x))
        output = self.init_layer(mesh, x)

        return output
