'''

'''

import numpy as np
import torch
from torch import nn, optim
from torch.nn.utils.parametrizations import spectral_norm as spn
import pytorch_lightning as pl

from .quadconv import PoolQuadConvBlock, QuadConvLayer
from .conv import PoolConvBlock

'''
Encoder module
'''
class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        #establish block type
        conv_params = kwargs.pop('conv_params')

        #specific args
        dimension = kwargs.pop('dimension')
        latent_dim = kwargs.pop('latent_dim')
        stages = kwargs.pop('stages')
        #channel_seq = kwargs.pop('channel_seq')
        input_shape = kwargs.pop('input_shape')

        #activations
        forward_activation = kwargs.pop('forward_activation')
        latent_activation = kwargs.pop('latent_activation')
        self.activation1 = latent_activation()
        self.activation2 = latent_activation()

        channel_seq = conv_params['channel_seq']
        conv_type = conv_params['conv_type']

        if conv_type == 'standard':

            initial_kernel_size = conv_params['initial_kernel_size']

            Block = PoolConvBlock
            init_layer = nn.Conv1d( 
                                    1, 
                                    channel_seq[0], 
                                    kernel_size=initial_kernel_size, 
                                    stride=1, 
                                    padding='same', 
                                    bias=False
                                    )

        elif conv_type == 'quadrature':

            mlp_channels = conv_params['mlp_channels']
            point_seq = conv_params['point_seq']

            Block = PoolQuadConvBlock
            init_layer = QuadConvLayer( 
                                        1,
                                        1,
                                        channel_seq[0],
                                        N_in = point_seq[0],
                                        N_out = point_seq[0],
                                        mlp_channels = mlp_channels,
                                        use_bias=False,
                                        )



        #build network
        self.cnn = nn.Sequential()

        self.cnn.append(init_layer)

        for i in range(stages):
            self.cnn.append(Block(  dimension,
                                    channel_seq[i],
                                    channel_seq[i+1],
                                    N_in = point_seq[i],
                                    N_out = point_seq[i],
                                    activation1 = forward_activation(),
                                    activation2 = forward_activation(),
                                    **kwargs
                                    ))

        self.cnn_out_shape = self.cnn(torch.randn(size=input_shape)).shape

        self.flat = nn.Flatten(start_dim=1, end_dim=-1)

        self.linear = nn.Sequential()

        self.linear.append(spn(nn.Linear(self.cnn_out_shape.numel(), latent_dim)))
        self.linear.append(self.activation1)
        self.linear.append(spn(nn.Linear(latent_dim, latent_dim)))
        self.linear.append(self.activation2)

        self.linear(self.flat(torch.zeros(self.cnn_out_shape)))

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

        conv_params = kwargs.pop('conv_params')

        #specific args
        dimension = kwargs.pop('dimension')
        latent_dim = kwargs.pop('latent_dim')
        stages = kwargs.pop('stages')
        input_shape = kwargs.pop('input_shape')

        #activations
        forward_activation = kwargs.pop('forward_activation')
        latent_activation = kwargs.pop('latent_activation')
        self.activation1 = latent_activation()
        self.activation2 = latent_activation()

        #establish block type
        channel_seq = conv_params['channel_seq']
        conv_type = conv_params['conv_type']

        if conv_type == 'standard':

            initial_kernel_size = conv_params['initial_kernel_size']

            Block = PoolConvBlock
            init_layer = nn.Conv1d( 
                                    channel_seq[0], 
                                    1, 
                                    kernel_size=initial_kernel_size, 
                                    stride=1, 
                                    padding='same', 
                                    bias=False
                                    )

        elif conv_type == 'quadrature':

            mlp_channels = conv_params['mlp_channels']
            point_seq = conv_params['point_seq']

            Block = PoolQuadConvBlock
            init_layer = QuadConvLayer( 
                                        1,
                                        channel_seq[0],
                                        1,
                                        N_in = point_seq[0],
                                        N_out = point_seq[0],
                                        mlp_channels = mlp_channels,
                                        use_bias=False,
                                        )

        #build network
        self.unflat = nn.Unflatten(1, input_shape[1:])

        self.linear = nn.Sequential()

        self.linear.append(spn(nn.Linear(latent_dim, latent_dim)))
        self.linear.append(self.activation1)
        self.linear.append(spn(nn.Linear(latent_dim, input_shape.numel())))
        self.linear.append(self.activation2)

        self.linear(torch.zeros(latent_dim))

        self.cnn = nn.Sequential()

        for i in range(stages).__reversed__():
            self.cnn.append(Block(  dimension,
                                    channel_seq[i],
                                    channel_seq[i],
                                    N_in = point_seq[i],
                                    N_out = point_seq[i],
                                    activation1 = forward_activation() if i!=1 else nn.Identity(),
                                    activation2 = forward_activation(),
                                    adjoint = True,
                                    **kwargs
                                    ))

        self.cnn.append(init_layer)

    def forward(self, x):
        x = self.linear(x)
        x = self.unflat(x)
        output = self.cnn(x)

        return output

'''
Quadrature convolution based autoencoder

    dimension: space dimension (e.g. 3D)
    latent_dim: dimension of latent representation
    point_seq: number of points along each dimension
    channel_seq:
    mlp_channels:
'''
class AutoEncoder(pl.LightningModule):
    def __init__(self,
                    dimension,
                    latent_dim,
                    conv_params,
                    forward_activation = nn.CELU,
                    latent_activation = nn.CELU,
                    output_activation = nn.Tanh,
                    loss_fn = nn.functional.mse_loss,
                    noise_scale = 0.0,
                    input_shape = None,
                    optimizer = "adam",
                    learning_rate = 1e-2,
                    profiler = None,
                    **kwargs
                    ):
        super().__init__()

        #save model hyperparameters under self.hparams
        self.save_hyperparameters(ignore=[  'input_shape',
                                            'optimizer',
                                            'output_activation',
                                            'loss_fn',
                                            'noise_scale',
                                            'learning_rate',
                                            'profiler' ])

        #model pieces
        self.encoder = Encoder(**self.hparams,
                                input_shape=input_shape)
        self.decoder = Decoder(**self.hparams,
                                input_shape=self.encoder.cnn_out_shape)

        #training hyperparameters
        self.dimension = dimension
        self.loss_fn = self.sobolev_loss
        self.latent_bool = False
        self.noise_scale = noise_scale
        self.output_activation = output_activation()
        self.optimizer = optimizer
        self.learning_rate = learning_rate

        self.profiler = profiler

    def sobolev_loss(self, pred, x, order=1, lambda_r = torch.tensor(0.25)):

        batch_size = pred.shape[0]

        if self.dimension == 1:

            stencil = torch.tensor([-1.0, 2.0, -1.0], device=self.device)*1/2
            stencil = torch.reshape(stencil, (1,1,3)).repeat(x.shape[1], x.shape[1], 1)

            temp_pred = pred
            temp_x = x

            loss = torch.sum((temp_pred-temp_x)**2)**(0.5)

            for i in range(order):
                temp_x = torch.nn.functional.conv1d(temp_x, stencil)
                temp_pred = torch.nn.functional.conv1d(temp_pred, stencil)

                loss += torch.sqrt(lambda_r**(i+1)) * torch.sum((temp_pred-temp_x)**2)**(0.5)

        elif self.dimension == 2:

            sq_shape = np.sqrt(x.shape[2]).astype(int)

            numel = sq_shape * sq_shape

            temp_x = torch.reshape(x, (x.shape[0],x.shape[1],sq_shape,sq_shape))
            temp_pred = torch.reshape(pred, (pred.shape[0],pred.shape[1],sq_shape,sq_shape))

            #compute function squared l2 error
            loss = torch.nn.functional.mse_loss(temp_pred,temp_x)

            #compute derivatives squared l2 error
            stencil = torch.tensor([[0.0, -1.0, 0.0],[-1.0, 4.0, -1.0],[0.0, -1.0, 0.0]], device=self.device)*1/4
            stencil = torch.reshape(stencil, (1,1,3,3)).repeat(1, x.shape[1], 1, 1)

            for i in range(order):
                temp_x = torch.nn.functional.conv2d(temp_x, stencil)
                temp_pred = torch.nn.functional.conv2d(temp_pred, stencil)

                loss += torch.sqrt(lambda_r**(i+1)) * torch.nn.functional.mse_loss(temp_pred,temp_x)

        return loss/batch_size

    @staticmethod
    def add_args(parent_parser):
        parser = parent_parser.add_argument_group("AutoEncoder")

        parser.add_argument('--conv_type', type=str)
        parser.add_argument('--dimension', type=int)
        parser.add_argument('--latent_dim', type=int)
        #parser.add_argument('--point_seq', type=int, nargs='+')
        #parser.add_argument('--channel_seq', type=int, nargs='+')
        #parser.add_argument('--mlp_channels', type=int, nargs='+')
        parser.add_argument('--conv_params', type=dict)

        return parent_parser

    def forward(self, x):
        return self.output_activation(self.decoder(self.encoder(x)))

    def training_step(self, batch, idx):
        #encode and add noise to latent rep.
        latent = self.encoder(batch)
        if self.noise_scale != 0.0:
            latent = latent + self.noise_scale*torch.randn(latent.shape, device=self.device)

        #decode
        pred = self.output_activation(self.decoder(latent))

        #compute loss
        loss = self.loss_fn(pred, batch)

        if self.latent_bool:

            loss += torch.sum(torch.abs(latent))/100

        self.log('train_loss', loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, idx):
        pred = self(batch)

        dim = tuple([i for i in range(1, pred.ndim)])

        n = torch.sqrt(torch.sum((pred-batch)**2, dim=dim))
        d = torch.sqrt(torch.sum((batch)**2, dim=dim))

        error = torch.sum(n/d)/pred.shape[0]

        self.log('val_err', error, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch, idx):
        pred = self(batch)

        dim = tuple([i for i in range(1, pred.ndim)])

        n = torch.sqrt(torch.sum((pred-batch)**2, dim=dim))
        d = torch.sqrt(torch.sum((batch)**2, dim=dim))

        error = torch.sum(n/d)/pred.shape[0]

        self.log('test_err', error, on_step=True, on_epoch=True, sync_dist=True)

    def predict_step(self, batch, idx):
        return self(batch)

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            raise NotImplementedError("Optimizers besides Adam not currently supported.")

        return optimizer
