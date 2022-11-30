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

from core.utilities import package_args, swap
from core.conv_blocks import PoolBlock

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

        #layer switch
        layer_lookup = {
            1 : (nn.Conv1d),
            2 : (nn.Conv2d),
            3 : (nn.Conv3d),
            }

        Conv = layer_lookup[spatial_dim]

        #build network
        init_layer = Conv(**arg_stack[0])

        self.cnn.append(init_layer)

        for i in range(1, stages):
            self.cnn.append(PoolBlock(spatial_dim = spatial_dim,
                                        **arg_stack[i],
                                        activation1 = forward_activation,
                                        activation2 = forward_activation,
                                        **kwargs
                                        ))

        self.conv_out_shape = self.cnn(torch.zeros(input_shape)).shape

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
    def forward(self, x):
        x = self.cnn(x)
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

        #layer switch
        layer_lookup = {
            1 : (nn.Conv1d),
            2 : (nn.Conv2d),
            3 : (nn.Conv3d),
            }

        Conv = layer_lookup[spatial_dim]

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

        for i in range(stages-1):
            self.cnn.append(PoolBlock(spatial_dim = spatial_dim,
                                        **arg_stack[i],
                                        activation1 = forward_activation,
                                        activation2 = forward_activation,
                                        adjoint = True,
                                        **kwargs
                                        ))

        init_layer = Conv(**arg_stack[0])

        self.cnn.append(init_layer)

    '''
    Forward
    '''
    def forward(self, x):
        x = self.linear(x)
        x = self.unflat(x)
        output = self.cnn(x)

        return output
