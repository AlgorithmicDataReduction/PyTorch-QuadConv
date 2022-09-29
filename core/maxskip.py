'''

'''

from typing import List
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

    __allowed = ('',

    )

    def __init__(self, **kwargs):
        super().__init__()

        for key, value in kwargs.items():
            if key in self.__allowed:
                setattr(self, key,value)

        #establish block type
        conv_type = kwargs['conv_type']
        conv_params = kwargs['conv_params']

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
        

        arg_stack = self.package_args(conv_params, stages)

        if conv_type == 'standard':

            Block = PoolConvBlock
            init_layer = nn.Conv1d(**arg_stack[0])

        elif conv_type == 'quadrature':

            Block = PoolQuadConvBlock
            init_layer = QuadConvLayer( dimension = dimension, **arg_stack[0] )


        #build network
        self.cnn = nn.Sequential()

        self.cnn.append(init_layer)

        for i in range(stages):
            self.cnn.append(Block(  dimension = dimension,
                                    **arg_stack[i+1],
                                    activation1 = forward_activation() if i!=1 else nn.Identity(),
                                    activation2 = forward_activation(),
                                    ))

        self.cnn_out_shape = self.cnn(torch.randn(size=input_shape)).shape

        self.flat = nn.Flatten(start_dim=1, end_dim=-1)

        self.linear = nn.Sequential()

        self.linear.append(spn(nn.Linear(self.cnn_out_shape.numel(), latent_dim)))
        self.linear.append(self.activation1)
        self.linear.append(spn(nn.Linear(latent_dim, latent_dim)))
        self.linear.append(self.activation2)

        self.out_shape = self.linear(self.flat(torch.zeros(self.cnn_out_shape)))

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
    def __init__(self, **kwargs):
        super().__init__()

        #establish block type
        conv_type = kwargs['conv_type']
        conv_params = kwargs['conv_params']

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

        arg_stack = self.package_args(conv_params, stages)

        if conv_type == 'standard':

            Block = PoolConvBlock
            init_layer = nn.Conv1d( **arg_stack[0] )

        elif conv_type == 'quadrature':

            Block = PoolQuadConvBlock
            init_layer = QuadConvLayer( dimension = dimension, **arg_stack[0] )

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
            self.cnn.append(Block(  dimension = dimension,
                                    **arg_stack[i+1],
                                    activation1 = forward_activation() if i!=1 else nn.Identity(),
                                    activation2 = forward_activation(),
                                    adjoint = True,
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

        if 'N_in' in args and 'N_out' in args:

            N_in = args['N_in']
            N_out = args['N_out']

            args['N_in'] = N_out
            args['N_out'] = N_in

        for key in args:
            if isinstance(args[key],List) and len(args[key]) == 1:
                args[key] = args[key]*(stages+1)

        arg_stack = [{ key : args[key][i] for key in args } for i in range(stages+1)]

        return arg_stack

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
