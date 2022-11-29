'''
Miscellaneous utility functions.
'''

import numpy as np
import matplotlib.pyplot as plt
import gif
import warnings
from typing import List

import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only

'''
Generates a side-by-side GIF of the raw data and the model reconstruction for the
test dataset; logs the result.

Input:
    trainer: lightning trainer
    datamodule: data module
    model: model to use or none to use best checkpoint
'''
def make_gif(trainer, datamodule, model):
    #run on test data
    if model == None:
        results = trainer.predict(ckpt_path='best', datamodule=datamodule)
    else:
        results = trainer.predict(model=model, datamodule=datamodule)

    #agglomerate the data if necessary (if tiling was used)
    data = datamodule.agglomerate(results)

    #if multichannel then just take first channel
    if data.shape[-1] > 1:
        data = data[...,0]

    data = data.squeeze()

    #get plotting function
    plot_func = datamodule.get_plot_func()

    if plot_func == None:
        return

    #gif frame closure
    @gif.frame
    def plot(i):
        fig, ax = plt.subplots(1, 2)

        plot_func(datamodule.get_sample(i), ax[0])
        ax[0].set_title("Uncompressed")

        im = plot_func(data[i,...], ax[1])
        ax[1].set_title("Reconstructed")

        if datamodule.spatial_dim == 2:
            # mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
            fig.colorbar(im, ax=ax.ravel().tolist(), location='bottom')

    #build frames
    frames = [plot(i) for i in range(data.shape[0])]

    #save gif
    gif.save(frames, f'{trainer.logger.log_dir}/{"last" if model else "best"}.gif', duration=50)

    return

'''
Sobolev loss function; computes the loss as a sum of the function l2 and
derivative l2 losses.

Input:
    spatial_dim: spatial dimension of data
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

        #build centered difference stencil
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

        #set attributes
        self.stencil = nn.Parameter(stencil, requires_grad=False)

        self.spatial_dim = spatial_dim
        self.order = order
        self.lambda_r = lambda_r

        return

    '''
    Compute loss between batch of predictions and reference targets.

    Input:
        pred: predictions
        target: reference values

    Output: average sobolev loss
    '''
    def forward(self, pred, target):
        #get a few details
        batch_size = pred.shape[0]
        channels = pred.shape[1]

        #compute function l2 error
        loss = nn.functional.mse_loss(pred, target)**(0.5)

        #copy predictions and target
        _pred = pred
        _target = target

        #setup
        if self.spatial_dim == 1:
            conv = nn.functional.conv1d

            stencil = self.stencil.expand(channels, channels, -1)

        elif self.spatial_dim == 2:

            if _target.dim() == 3:
                sq_shape = np.sqrt(_pred.shape[2]).astype(int)

                _pred = torch.reshape(_pred, (_pred.shape[0], _pred.shape[1], sq_shape, sq_shape))
                _target = torch.reshape(_target, (_target.shape[0], _target.shape[1], sq_shape, sq_shape))

            conv = nn.functional.conv2d

            stencil = self.stencil.expand(1, channels, -1, -1)

        #compute derivative l2 losses
        for i in range(self.order):
            _pred = conv(_pred, stencil)
            _target = conv(_target, stencil)

            loss += torch.sqrt(self.lambda_r**(i+1)) * nn.functional.mse_loss(_pred, _target)**(0.5)

        #return average loss w.r.t batch
        return loss/pred.shape[0]

'''
Custom Tensorboard logger; does not log hparams.yaml or the epoch metric.
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

'''
Package conv parameters.

Input:
    kwargs: keyword arguments
'''
def package_args(stages:int, kwargs:dict, mirror=False):

    for key, value in kwargs.items():
        if len(value) == 1:
            kwargs[key] = value*(stages)
        elif mirror:
            value.reverse() #inplace

    arg_stack = [{ key : value[i] for key, value in kwargs.items() } for i in range(stages)]

    return arg_stack

'''
Swap input and output points and channels
'''
def swap(conv_params):
    temp = conv_params["in_points"]
    conv_params["in_points"] = conv_params["out_points"]
    conv_params["out_points"] = temp

    temp = conv_params["in_channels"]
    conv_params["in_channels"] = conv_params["out_channels"]
    conv_params["out_channels"] = temp

    return conv_params
