'''

'''

from importlib import import_module

import torch
from torch import nn
import pytorch_lightning as pl

'''
Convolution based autoencoder.

Input:
    module:
    input_shape:
'''
class AutoEncoder(pl.LightningModule):

    def __init__(self,*,
            module,
            input_shape,
            loss_fn = "MSELoss",
            optimizer = "Adam",
            learning_rate = 1e-2,
            noise_scale = 0.0,
            output_activation = nn.Tanh,
            **kwargs
        ):
        super().__init__()

        #import the encoder and decoder
        module = import_module('core.modules.' + module)

        #training hyperparameters
        self.loss_fn = getattr(nn, loss_fn)()
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.noise_scale = noise_scale

        #model pieces
        self.output_activation = output_activation()

        self.encoder = module.Encoder(input_shape=input_shape, **kwargs)
        self.decoder = module.Decoder(input_shape=self.encoder.conv_out_shape, **kwargs)

        return

    @staticmethod
    def add_args(parent_parser):
        parser = parent_parser.add_argument_group("AutoEncoder")

        # parser.add_argument()

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

        error = n/d

        self.log('val_err', torch.mean(error), on_step=False, on_epoch=True, sync_dist=True)

        return

    def test_step(self, batch, idx):
        pred = self(batch)

        dim = tuple([i for i in range(1, pred.ndim)])

        n = torch.sqrt(torch.sum((pred-batch)**2, dim=dim))
        d = torch.sqrt(torch.sum((batch)**2, dim=dim))

        error = n/d

        self.log('test_avg_err', torch.mean(error), on_step=False, on_epoch=True, sync_dist=True,
                    reduce_fx=torch.mean)
        self.log('test_max_err', torch.max(error), on_step=False, on_epoch=True, sync_dist=True,
                    reduce_fx=torch.max)

        return

    def predict_step(self, batch, idx):
        return self(batch)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer)(self.parameters(), lr=self.learning_rate)

        return optimizer
