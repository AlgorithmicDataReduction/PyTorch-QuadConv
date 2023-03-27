'''
Quadratrure, downsampling, and agglomeration functions.
'''

import torch

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
    x0: left end point
    x1: right end point
'''
def param_quad(input_points, num_points):

    spatial_dim = input_points.shape[1]

    num_points = round(num_points**(1/spatial_dim))

    coord_min,_ = torch.min(input_points,dim=0)
    coord_max,_ = torch.max(input_points,dim=0)

    #nodes
    nodes = []
    for i in range(spatial_dim):
        nodes.append(torch.linspace(coord_min[i], coord_max[i], num_points))

    nodes = torch.meshgrid(*nodes, indexing='xy')
    nodes = torch.dstack(nodes).view(-1, spatial_dim)

    return nodes, None

def param_quad_const_weights(input_points, num_points):

    spatial_dim = input_points.shape[1]

    num_points = round(num_points**(1/spatial_dim))

    coord_min,_ = torch.min(input_points,dim=0)
    coord_max,_ = torch.max(input_points,dim=0)

    #nodes
    nodes = []
    for i in range(spatial_dim):
        nodes.append(torch.linspace(coord_min[i], coord_max[i], num_points))

    nodes = torch.meshgrid(*nodes, indexing='xy')
    nodes = torch.dstack(nodes).view(-1, spatial_dim)

    return nodes, torch.ones(nodes.shape[0])

'''
Randomly downsample the input points.

NOTE: only using one permuation here which is a bit weird

Input:
    input_points: input points
    num_points: number of points to sample
'''
def random_downsample(input_points, num_points):

    idxs = torch.randperm(input_points.shape[0], device=input_points.device)[:num_points]

    return input_points[idxs, :], None

def random_downsample_const_weights(input_points, num_points):

    idxs = torch.randperm(input_points.shape[0], device=input_points.device)[:num_points]

    return input_points[idxs, :], torch.ones(num_points)

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
