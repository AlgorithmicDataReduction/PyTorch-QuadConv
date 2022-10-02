'''
Utility functions.
'''

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only

import matplotlib.pyplot as plt
import gif

import warnings

'''
Makes a GIF of the model output on the test data provided by the data module and
saves it to the appropriate lightning log.

Input:
    trainer: lightning trainer
    datamodule: data module
    model: model to use
'''
def make_gif(trainer, datamodule, model):
    #run on test data
    if model:
        results = trainer.predict(model=model, datamodule=datamodule)
    else:
        results = trainer.predict(ckpt_path='best', datamodule=datamodule)

    data = datamodule.agglomerate(results)

    #if multichannel then just take first channel
    if data.dim() > datamodule.point_dim+1:
        data = data[...,0]

    #check dimension of data
    if datamodule.point_dim == 1:
        #gif frame closure
        @gif.frame
        def plot(i):
            fig, ax = plt.subplots(1, 2)

            ax[0].plot(datamodule.get_sample(i))
            ax[0].set_title("Uncompressed")

            ax[1].plot(data[i,:])
            ax[1].set_title("Reconstructed")

    elif datamodule.point_dim == 2:
        #gif frame closure
        @gif.frame
        def plot(i):
            fig, ax = plt.subplots(1, 2)

            ax[0].imshow(datamodule.get_sample(i), vmin=-1, vmax=1, origin='lower')
            ax[0].set_title("Uncompressed")

            im = ax[1].imshow(data[i,:,:], vmin=-1, vmax=1, origin='lower')
            ax[1].set_title("Reconstructed")

            fig.colorbar(im, ax=ax.ravel().tolist(), location='bottom')

    elif datamodule.point_dim == 3:
        warnings.warn("Warning...GIF create for 3d data not supported.", )
        return

    #build frames
    frames = [plot(i) for i in range(data.shape[0])]

    #save gif
    gif.save(frames, f'{trainer.logger.log_dir}/{"last" if model else "best"}.gif', duration=50)

    return

'''
Module wrapper around sin function; allows it to operate as a layer.
'''
class Sin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

'''
Sobolev loss function, which computes the loss as a sum of the function l2 and
derivative l2 losses.

Input:
    order: max derivative order
    lambda_r: derivative error weighting
'''
class SobolevLoss(nn.Module):

    def __init__(self,*,
            spatial_dim,
            order = 1,
            lambda_r = torch.tensor(0.25)
        ):
        super().__init__()

        if spatial_dim == 1:
            stencil = torch.tensor([-1.0, 2.0, -1.0])*1/2
            stencil = torch.reshape(stencil, (1,1,3))

        elif spatial_dim == 2:
            stencil = torch.tensor([[0.0, -1.0, 0.0],[-1.0, 4.0, -1.0],[0.0, -1.0, 0.0]])*1/4
            stencil = torch.reshape(stencil, (1,1,3,3))

        elif spatial_dim == 3:
            raise NotImplementedError('A spatial dimension of 3 is not currently supported.')

        else:
            raise ValueError(f'A spatial dimension of {spatial_dim} is invalid.')

        self.stencil = nn.Parameter(stencil, requires_grad=False)

        self.spatial_dim = spatial_dim
        self.order = order
        self.lambda_r = lambda_r

        return

    '''
    Input:
        pred: predictions
        target: actual values
    '''
    def forward(self, pred, target):
        #
        batch_size = pred.shape[0]
        channels = pred.shape[1]

        #compute function l2 error
        loss = nn.functional.mse_loss(pred, target)

        #copy predictions and target
        _pred = pred
        _target = target

        #setup
        if self.spatial_dim == 1:
            conv = nn.functional.conv1d

            stencil = self.stencil.repeat(channels, channels, 1)

        elif self.spatial_dim == 2:
            sq_shape = np.sqrt(_pred.shape[2]).astype(int)

            _pred = torch.reshape(_pred, (_pred.shape[0], _pred.shape[1], sq_shape, sq_shape))
            _target = torch.reshape(_target, (_target.shape[0], _target.shape[1], sq_shape, sq_shape))

            conv = nn.functional.conv2d

            stencil = self.stencil.repeat(1, channels, 1, 1)

        #compute derivative l2 losses
        for i in range(self.order):
            _pred = conv(_pred, stencil)
            _target = conv(_target, stencil)

            loss += torch.sqrt(self.lambda_r**(i+1)) * nn.functional.mse_loss(_pred, _target)

        return loss/pred.shape[0]

'''
Custom Tensorboard logger.
'''
class Logger(TensorBoardLogger):
    def __init__(self,
            **kwargs
        ):
        super().__init__(**kwargs)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        metrics.pop('epoch', None)
        return super().log_metrics(metrics, step)

    @rank_zero_only
    def save(self):
        pass
