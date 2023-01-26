'''
'''

from importlib import import_module

import torch
from torch import nn

import pytorch_lightning as pl

from core.torch_quadconv import MeshHandler
from core.torch_quadconv.utils.sobolev import SobolevLoss
from core.torch_quadconv.utils.quadloss import QuadLoss

'''
Quadrature convolution autoencoder.

Input:
    module: python module to import containing encoder and decoder classes
    spatial_dim: spatial dimension of data
    point_seq:
    data_info:
    loss_fn: loss function specification
    optimizer: optimizer specification
    learning_rate: learning rate
    noise_scale: scale of noise to be added to latent representation in training
    output_activation: final activation
    kwargs: keyword arguments to be passed to encoder and decoder
'''
class Model(pl.LightningModule):

    def __init__(self,*,
            module,
            spatial_dim,
            data_info,
            point_seq,
            quad_map = "newton_cotes_quad",
            weight_activation = 'Identity',
            normalize_weights = False,
            loss_fn = "MSELoss",
            optimizer = "Adam",
            learning_rate = 1e-2,
            noise_scale = 0.0,
            output_activation = nn.Tanh,
            **kwargs
        ):
        super().__init__()

        #save hyperparameters for checkpoint loading
        self.save_hyperparameters(ignore=["data_info"])

        #import the encoder and decoder
        module = import_module('core.QCAE.' + module)

        #training hyperparameters
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.noise_scale = noise_scale

        #unpack data info
        input_shape = data_info['input_shape']
        input_nodes = data_info['input_nodes']
        input_weights = data_info['input_weights']

        #
        self.example_input_array = torch.zeros(input_shape)

        #model pieces
        self.mesh = MeshHandler(input_nodes, input_weights, quad_map=quad_map, weight_activation=weight_activation, normalize_weights=normalize_weights).cache(point_seq, mirror=True)

        #loss function
        #NOTE: There is probably a bit of a better way to do this, but this
        #should work for now.
        if loss_fn == 'SobolevLoss':
            self.loss_fn = SobolevLoss(spatial_dim=spatial_dim)
        elif loss_fn == 'QuadLoss':
            self.loss_fn = QuadLoss(mesh_pointer=self.mesh)
        else:
            self.loss_fn = getattr(nn, loss_fn)()

        self.output_activation = output_activation()

        self.encoder = module.Encoder(spatial_dim=spatial_dim,
                                        **kwargs)
        self.decoder = module.Decoder(input_shape=self.encoder.conv_out_shape,
                                        spatial_dim=spatial_dim,
                                        **kwargs)

        return

    #NOTE: Not currently used
    @staticmethod
    def add_args(parent_parser):
        parser = parent_parser.add_argument_group("AutoEncoder")

        # parser.add_argument()

        return parent_parser

    '''
    Forward pass of encoder.

    Input:
        x: input data

    Output: compressed data
    '''
    def encode(self, x):
        return self.encoder(self.mesh, x)

    '''
    Forward pass of decoder.

    Input:
        z: compressed data

    Output: compressed data reconstruction
    '''
    def decode(self, z):
        return self.output_activation(self.decoder(self.mesh, z))

    '''
    Forward pass of model.

    Input:
        x: input data

    Output: compressed data reconstruction
    '''
    def forward(self, x):
        return self.decode(self.encode(x))

    '''
    Single training step.

    Input:
        batch: batch of data
        idx: batch index

    Output: pytorch loss object
    '''
    def training_step(self, batch, idx):
        #encode and add noise to latent rep.
        latent = self.encode(batch)
        if self.noise_scale != 0.0:
            latent = latent + self.noise_scale*torch.randn(latent.shape, device=self.device)

        #decode
        pred = self.decode(latent)

        #compute loss
        loss = self.loss_fn(pred, batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    '''
    Single validation_step; logs validation error.

    Input:
        batch: batch of data
        idx: batch index
    '''
    def validation_step(self, batch, idx):
        #predictions
        pred = self(batch)

        #compute average relative reconstruction error
        dim = tuple([i for i in range(1, pred.ndim)])

        n = torch.sqrt(torch.sum((pred-batch)**2, dim=dim))
        d = torch.sqrt(torch.sum((batch)**2, dim=dim))

        error = n/d

        #log validation error
        self.log('val_err', torch.mean(error), on_step=False, on_epoch=True, sync_dist=True)

        return

    '''
    Single test step; logs average and max test error

    Input:
        batch: batch of data
        idx: batch index
    '''
    def test_step(self, batch, idx):
        #predictions
        pred = self(batch)

        #compute relative reconstruction error
        dim = tuple([i for i in range(1, pred.ndim)])

        n = torch.sqrt(torch.sum((pred-batch)**2, dim=dim))
        d = torch.sqrt(torch.sum((batch)**2, dim=dim))

        error = n/d

        #log average and max error w.r.t batch
        self.log('test_avg_err', torch.mean(error), on_step=False, on_epoch=True, sync_dist=True,
                    reduce_fx=torch.mean)
        self.log('test_max_err', torch.max(error), on_step=False, on_epoch=True, sync_dist=True,
                    reduce_fx=torch.max)

        return

    '''
    Single prediction step.

    Input:
        batch: batch of data
        idx: batch index

    Output: compressed data reconstruction
    '''
    def predict_step(self, batch, idx):
        return self(batch)

    '''
    Instantiates optimizer

    Output: pytorch optimizer
    '''
    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer)(self.parameters(), lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5)
        scheduler_config = {"scheduler": scheduler, "monitor": "val_err"}

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
