'''
Agglomeration functions.
'''

import torch

import numpy as np
from ctypes import CDLL, POINTER, c_float

################################################################################

'''
Agglomerate a mesh.

Input:
    input_points: input points with shape [num input points, spatial dimension]
    adjacency: connectivity of input points
    num_output_points: number of output points

NOTE: Compile the .so file using the following command
    gcc -shared -o <output_name>.so -fPIC <name>.c
'''
def agglomerate(input_points, adjacency, num_output_points):

    spatial_dim = input_points.shape[1]

    #input array
    input_p = input_points.numpy().ctypes.data_as(POINTER(c_float)) #c pointer to underlying data

    #output array
    output_points = np.zeros((num_output_points, spatial_dim), dtype=np.float32) #new numpy array
    output_p = output_points.ctypes.data_as(POINTER(c_float)) #c pointer to underlying data

    #call c function
    lib_path = "/home/rs-coop/Documents/Research/ASCR-Compression/QuadConv/c_test/test.so"
    lib = CDLL(lib_path)
    lib.test(input_p, output_p, num_output_points*spatial_dim) #modifies the data of output_points

    return torch.from_numpy(output_points) #torch tensor from output_points

'''
NOTE: only using one permuation here which is a bit weird
'''
def random_downsample(input_points, num_points):

    idxs = torch.randperm(input_points.shape[0], device=input_points.device)[:num_points]

    return torch.index_select(input_points, 0, idxs)
