'''

'''

import numpy as np
import torch
from torch import nn, optim
from torch.nn.utils.parametrizations import spectral_norm as spn
import pytorch_lightning as pl

from .quadconv import QuadConvBlock
from .conv import ConvBlock

'''
Encoder module
'''
class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        #establish block type
        conv_type = kwargs.pop('conv_type')
        if conv_type == 'standard':
            Block = ConvBlock
        elif conv_type == 'quadrature':
            Block = QuadConvBlock

        #specific args
        point_dim = kwargs.pop('point_dim')
        latent_dim = kwargs.pop('latent_dim')
        point_seq = kwargs.pop('point_seq')
        channel_seq = kwargs.pop('channel_seq')
        input_shape = kwargs.pop('input_shape')

        #activations
        forward_activation = kwargs.pop('forward_activation')
        latent_activation = kwargs.pop('latent_activation')
        self.activation1 = latent_activation()
        self.activation2 = latent_activation()

        #build network
        self.cnn = nn.Sequential()

        for i in range(len(point_seq)-1):
            self.cnn.append(Block(point_dim,
                                    channel_seq[i],
                                    channel_seq[i+1],
                                    N_in = point_seq[i],
                                    N_out = point_seq[i+1],
                                    activation1 = forward_activation(),
                                    activation2 = forward_activation(),
                                    **kwargs
                                    ))

        if conv_type == 'standard':
            self.cnn_out_shape = self.cnn(torch.zeros(input_shape)).shape
        else:
            self.cnn_out_shape = torch.Size((1, channel_seq[-1], point_seq[-1]**point_dim))

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

        #establish block type
        conv_type = kwargs.pop('conv_type')
        if conv_type == 'standard':
            Block = ConvBlock
        elif conv_type == 'quadrature':
            Block = QuadConvBlock

        #specific args
        point_dim = kwargs.pop('point_dim')
        latent_dim = kwargs.pop('latent_dim')
        point_seq = kwargs.pop('point_seq')
        channel_seq = kwargs.pop('channel_seq')
        input_shape = kwargs.pop('input_shape')

        #activations
        forward_activation = kwargs.pop('forward_activation')
        latent_activation = kwargs.pop('latent_activation')
        self.activation1 = latent_activation()
        self.activation2 = latent_activation()

        #build network
        self.unflat = nn.Unflatten(1, input_shape[1:])

        self.linear = nn.Sequential()

        self.linear.append(spn(nn.Linear(latent_dim, latent_dim)))
        self.linear.append(self.activation1)
        self.linear.append(spn(nn.Linear(latent_dim, input_shape.numel())))
        self.linear.append(self.activation2)

        self.linear(torch.zeros(latent_dim))

        self.cnn = nn.Sequential()

        for i in range(len(point_seq)-1, 0, -1):
            self.cnn.append(Block(point_dim,
                                    channel_seq[i],
                                    channel_seq[i-1],
                                    N_in = point_seq[i],
                                    N_out = point_seq[i-1],
                                    activation1 = forward_activation() if i!=1 else nn.Identity(),
                                    activation2 = forward_activation(),
                                    adjoint = True,
                                    **kwargs
                                    ))

    def forward(self, x):
        x = self.linear(x)
        x = self.unflat(x)
        output = self.cnn(x)

        return output

'''
Quadrature convolution based autoencoder

    point_dim: space dimension (e.g. 3D)
    latent_dim: dimension of latent representation
    point_seq: number of points along each dimension
    channel_seq:
    mlp_channels:
'''
class AutoEncoder(pl.LightningModule):
    def __init__(self,
                    conv_type,
                    point_dim,
                    latent_dim,
                    point_seq,
                    channel_seq,
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
        self.save_hyperparameters(ignore=['loss_fn',
                                            'noise_scale',
                                            'forward_activation',
                                            'latent_activation',
                                            'output_activation',
                                            'input_shape',
                                            'optimizer',
                                            'learning_rate',
                                            'profiler'])

        #model pieces
        self.encoder = Encoder(**self.hparams,
                                forward_activation=forward_activation,
                                latent_activation=latent_activation,
                                input_shape=input_shape)
        self.decoder = Decoder(**self.hparams,
                                forward_activation=forward_activation,
                                latent_activation=latent_activation,
                                input_shape=self.encoder.cnn_out_shape)

        #training hyperparameters
        self.loss_fn = loss_fn
        self.noise_scale = noise_scale
        self.output_activation = output_activation()
        self.optimizer = optimizer
        self.learning_rate = learning_rate

        self.profiler = profiler

    @staticmethod
    def add_args(parent_parser):
        parser = parent_parser.add_argument_group("AutoEncoder")

        parser.add_argument('--conv_type', type=str)
        parser.add_argument('--point_dim', type=int)
        parser.add_argument('--latent_dim', type=int)
        parser.add_argument('--point_seq', type=int, nargs='+')
        parser.add_argument('--channel_seq', type=int, nargs='+')
        parser.add_argument('--mlp_channels', type=int, nargs='+')

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