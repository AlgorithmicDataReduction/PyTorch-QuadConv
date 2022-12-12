'''
Agglomeration functions.
'''

import torch

import numpy
import ctypes

################################################################################

'''

'''
def agglomerate():
    pass

'''
NOTE: only using one permuation here which is a bit weird
'''
def random_downsample(input_points, num_points):

    if input_points.dim() == 3:
        dim = 1
    else:
        dim = 0

    idxs = torch.randperm(input_points.shape[dim], device=input_points.device)[:num_points]

    return torch.index_select(input_points, dim, idxs)
