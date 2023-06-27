'''
Quadratrure, downsampling, and agglomeration functions.
'''

import torch

from math import prod
from scipy.integrate import newton_cotes

################################################################################

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
def newton_cotes_quad(input_points, num_points, composite_quad_order=2):

    spatial_dim = input_points.shape[1]

    num_points = round(num_points**(1/spatial_dim))

    assert num_points%composite_quad_order == 0, f'Composite qudarature order {composite_quad_order} does not divide evenly into number of points {num_points}.'

    coord_min,_ = torch.min(input_points,dim=0)
    coord_max,_ = torch.max(input_points,dim=0)

    #nodes
    nodes = []
    for i in range(spatial_dim):
        nodes.append(torch.linspace(coord_min[i], coord_max[i], num_points))

    dx = (coord_max-coord_min) / (composite_quad_order-1)

    nodes = torch.meshgrid(*nodes, indexing='xy')
    nodes = torch.dstack(nodes).view(-1, spatial_dim)

    #weights
    rep = [int(num_points/composite_quad_order)]
    nc_weights = torch.as_tensor(newton_cotes(composite_quad_order-1, 1)[0],dtype=torch.float)
    weights = []

    for i in range(spatial_dim):
        weights.append(torch.tile(torch.Tensor((dx[i]/rep[0])*nc_weights), rep))

    weights =  torch.meshgrid(*weights, indexing='xy')
    weights = torch.dstack(weights).reshape(-1, spatial_dim)
    weights = torch.prod(weights, dim=1)

    return nodes, weights

'''
Builds a uniform grid of points.

Input:
    input_points: input pointss
    num_points: number of output points
    ratio: ratio of points along each spatial dimension
    base: logspace base, fractional has decreasing spacing from min to max
    const: whether to use constant weights or learned weights
'''
def param_quad(input_points, num_points, ratio=[], base=[], const=False):

    spatial_dim = input_points.shape[1]

    num_points = round((num_points/prod(ratio))**(1/spatial_dim))

    #check ratio
    if len(ratio) == 0:
        ratio = [1]*spatial_dim
    else:
        assert len(ratio) == spatial_dim

    #check base
    if len(base) == 0:
        base = [None]*spatial_dim
    else:
        assert len(base) == spatial_dim

    #grid dimensions
    coord_min,_ = torch.min(input_points,dim=0)
    coord_max,_ = torch.max(input_points,dim=0)

    #nodes
    nodes = []
    for i in range(spatial_dim):
        #linearly spaced nodes
        if base[i] == None:
            nodes.append(torch.linspace(coord_min[i], coord_max[i], num_points*ratio[i]))

        #logarithmically spaced nodes
        else:
            p = torch.logspace(0, 1, steps=num_points*ratio[i], base=base[i])
            p = coord_min[i] + ((coord_max[i]-coord_min[i])/(base[i]-1))*(p-1)

            nodes.append(p)

    nodes = torch.cartesian_prod(*nodes)

    #weights
    weights = torch.ones(nodes.shape[0]) if const else None

    return nodes, weights

'''
Randomly downsample the input points.

NOTE: only using one permuation here which is a bit weird

Input:
    input_points: input points
    num_points: number of points to sample
    const: whether to use constant weights or learned weights
'''
def random_downsample(input_points, num_points, const=False, seed=None):

    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
    else:
        gen = None

    idxs = torch.randperm(input_points.shape[0], device=input_points.device, generator=gen)[:num_points]

    #weights
    weights = torch.ones(num_points) if const else None

    return input_points[idxs, :], weights

################################################################################

def const_unit_weights(input_points, num_points):
    return torch.ones(num_points)

def newton_cotes_quad_square(spatial_dim, num_points):
    return newton_cotes_quad(torch.tensor([[1.0]*spatial_dim,[0.0]*spatial_dim]), num_points)

def newton_cotes_quad_n5(input_points, num_points):
    return newton_cotes_quad(input_points, num_points, 5)

def newton_cotes_quad_n5_square(spatial_dim, num_points):
    return newton_cotes_quad_n5(torch.tensor([[1.0]*spatial_dim,[0.0]*spatial_dim]), num_points)

################################################################################

def log_linear_weights(input_points, num_points, points_per_dim=[30, 30], base=4, composite_quad_order=2):

    coord_min, _ = torch.min(input_points, dim=0)
    coord_max, _ = torch.max(input_points, dim=0)

    #weights
    dx = torch.logspace(0, 1, points_per_dim[0], base=base)
    dx = (dx-1)/(base-1)

    dy = (coord_max[1]-coord_min[1]) / (composite_quad_order-1)

    weights = []

    #log dimension (x)
    trap = [(dx[1]-dx[0])/2]
    trap.extend([(dx[i+1]-dx[i-1])/2 for i in range(1, points_per_dim[0]-1)])
    trap.append((dx[-1]-dx[-2])/2)

    weights.append(torch.tensor(trap))

    #linear dimension (y)
    rep = [int(points_per_dim[1]/composite_quad_order)]
    nc_weights = torch.as_tensor(newton_cotes(composite_quad_order-1, 1)[0], dtype=torch.float)

    weights.append(torch.tile(torch.Tensor((dy/rep[0])*nc_weights), rep))

    #combine and reshape
    weights =  torch.meshgrid(*weights, indexing='xy')
    weights = torch.dstack(weights).reshape(-1, 2)
    weights = torch.prod(weights, dim=1)

    return weights
