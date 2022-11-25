'''
Miscellaneous utility functions.
'''

import warnings

import torch
import torch.nn as nn

from scipy.integrate import newton_cotes

from .FastGL.glpair import glpair

################################################################################

'''
Module wrapper for sin function; allows it to operate as a layer.
'''
class Sin(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

################################################################################

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

        #return average loss w.r.t batch
        return loss/pred.shape[0]

################################################################################

'''
Get Gaussian quadrature weights and nodes.

Input:
    input_points: input points
    num_points: number of output points
'''
def gauss_quad(input_points, num_points):

    spatial_dim = input_points.shape[1]

    num_points = int(num_points**(1/spatial_dim))

    weights = torch.zeros(num_points)
    nodes = torch.zeros(num_points)

    for i in range(num_points):
        _, weights[i], nodes[i] = glpair(num_points, i+1)

    #nodes
    nodes = [nodes]*spatial_dim
    nodes = torch.meshgrid(*nodes, indexing='xy')
    nodes = torch.dstack(nodes).view(-1, spatial_dim)

    #weights
    weights = [weights]*spatial_dim
    weights =  torch.meshgrid(*weights, indexing='xy')
    weights = torch.dstack(weights).reshape(-1, spatial_dim)
    weights = torch.prod(weights, dim=1)

    return nodes, weights

'''
Get Newton-Cotes quadrature weights and nodes.
NOTE: This function returns the composite rule, so its required that the order
of the quadrature rule divides evenly into N.

Input:
    input_points: input points
    num_points: number of output points
    composite_quad_order: composite qudrature order
    x0: left end point
    x1: right end point
'''
def newton_cotes_quad(input_points, num_points, composite_quad_order=2, x0=0, x1=1):

    spatial_dim = input_points.shape[1]

    num_points = int(num_points**(1/spatial_dim))

    #nodes
    dx = (x1-x0)/(composite_quad_order-1)

    nodes = torch.linspace(x0, x1, num_points)
    nodes = [nodes]*spatial_dim
    nodes = torch.meshgrid(*nodes, indexing='xy')
    nodes = torch.dstack(nodes).view(-1, spatial_dim)

    #weights
    rep = [int(num_points/composite_quad_order)]

    weights, _ = newton_cotes(composite_quad_order-1, 1)
    weights = torch.tile(torch.Tensor(dx*weights), rep)

    weights = [weights]*spatial_dim
    weights =  torch.meshgrid(*weights, indexing='xy')
    weights = torch.dstack(weights).reshape(-1, spatial_dim)
    weights = torch.prod(weights, dim=1)

    return nodes, weights

'''
'''
def random_downsample(input_points, num_points):

    if input_points.dim() == 3:
        dim = 1
    else:
        dim = 0

    #only using one permuation here which is a bit weird
    idxs = torch.randperm(input_points.shape[dim])[:num_points]

    return torch.index_select(input_points, dim, idxs)
