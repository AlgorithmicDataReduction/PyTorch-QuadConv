'''
Quadratrure and agglomeration functions.
'''

import torch

from scipy.integrate import newton_cotes
from .FastGL.glpair import glpair

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

def newton_cotes_no_weights(*args, **kwargs):
    return newton_cotes_quad(*args, **kwargs)[0]

'''
NOTE: only using one permuation here which is a bit weird
'''
def random_downsample(input_points, num_points):

    idxs = torch.randperm(input_points.shape[0], device=input_points.device)[:num_points]

    return input_points[idxs, :], None
