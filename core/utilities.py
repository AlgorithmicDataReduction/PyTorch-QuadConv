'''
Utility functions.
'''

import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only

import matplotlib.pyplot as plt
import gif

import warnings

'''
Module wrapper around sin function; allows it to operate as a layer.
'''
class Sin(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

'''
Sobolev loss function, which computes the loss as a sum of the function l2 and
derivative l2 losses.

Input:
    pred: predictions
    target: actual values
    order: max derivative order
    lambda_r: derivative error weighting
'''
def sobolev_loss(pred, target, order=1, lambda_r=torch.tensor(0.25)):
    #setup
    batch_size = pred.shape[0]

    sq_shape = np.sqrt(x.shape[2]).astype(int)
    numel = sq_shape * sq_shape

    temp_x = torch.reshape(x, (x.shape[0],x.shape[1],sq_shape,sq_shape))
    temp_pred = torch.reshape(pred, (pred.shape[0],pred.shape[1],sq_shape,sq_shape))

    #compute function l1 error
    loss = torch.sum((temp_pred-temp_x)**2)

    #compute derivatives l1 error
    stencil = torch.tensor([[0.0, -1.0, 0.0],[-1.0, 4.0, -1.0],[0.0, -1.0, 0.0]], device=x.device)*1/4
    stencil = torch.reshape(stencil, (1,1,3,3)).repeat(1, x.shape[1], 1, 1)

    for i in range(order):
        temp_x = torch.nn.functional.conv2d(temp_x, stencil)
        temp_pred = torch.nn.functional.conv2d(temp_pred, stencil)

        loss += lambda_r[i] * torch.sum((temp_pred-temp_x)**2)

    return loss/bs

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
Custom Tensorboard logger.
'''
class Logger(TensorBoardLogger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        metrics.pop('epoch', None)
        return super().log_metrics(metrics, step)
