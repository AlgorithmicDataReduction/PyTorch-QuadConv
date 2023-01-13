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

/*
NOTE: All multi-dimensional arrays are falttened C-style, i.e. row-wise.

Input:
  activity: bool* pointer of size (num_points X stages), active points at each stage
  points: double* pointer of size (num_points X spatial_dim), mesh points
  adj_indices: int* pointer of size (num_points+1), METIS xadj
  adjacency: int* pointer of size (2*(number of edges)), METIS adjncy
  boundary: int* pointer of size (num_boundary_points), indicies of boundary points in points array
  spatial_dim: int, spatial dimension of mesh points
  num_points: int, number of mesh points
  num_boundary_points: int, number of boundary points
  stages: int, number of coarsening stages
  factor: int, agglomerate division factor
*/
void agglomerate(activity, points, adj_indices, adjacency, boundary,
                  int spatial_dim, int num_points, int num_boundary_points, int stages, int factor)
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
