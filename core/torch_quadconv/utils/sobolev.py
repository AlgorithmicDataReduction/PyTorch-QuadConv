'''
'''

import torch
import torch.nn as nn

import numpy as np

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
