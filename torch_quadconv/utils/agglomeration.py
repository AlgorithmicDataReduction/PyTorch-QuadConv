'''
Agglomeration functions.
'''

import os
import sys
import numpy as np
from ctypes import CDLL, POINTER, c_double, c_bool, c_int

import torch

################################################################################

'''
Agglomerate a mesh.

Input:
    points: input points with shape (num input points, spatial dimension)
    bd_point_ind: indices of boundary points in points
    element_pos: points[eind[eptr[i]:eptr[i+1]]] constitutes an element
    element_ind: indices of points in points to construct elements
    levels: number of agglomeration levels
    factor: agglomerate division factor

NOTE: Compile the .so file using the following command
    gcc -shared -o <output_name>.so -fPIC <name>.c

/*
NOTE: All multi-dimensional arrays are falttened C-style, i.e. row-wise.

Input:
  activity: bool* pointer of size (num_points X stages), active points at each stage (first column is true and rest are false)
  points: double* pointer of size (num_points X spatial_dim), mesh points
  element_pos: int* pointer of size (num_elements+1)
  element_ind: int* pointer of size (element_pos[-1])
  bd_point_ind: int* pointer of size (num_boundary_points), indicies of boundary points in points array
  spatial_dim: int, spatial dimension of mesh points
  num_points: int, number of mesh points
  num_elements: int, number of mesh elements
  num_bd_points: int, number of boundary points
  levels: int, number of coarsening levels
  factor: int, agglomerate division factor
*/
void agglomerate(activity, points, element_indices, elements, boundary, int spatial_dim,
                    int num_points, int num_elements, int num_boundary_points, int stages, int factor)
'''
def agglomerate(points, elements, levels, factor):

    print("Agglomerating...")

    #extract element details
    bd_point_ind = elements.bd_point_ind
    element_pos = elements.element_pos
    element_ind = elements.element_ind

    #extract some attributes
    num_points, spatial_dim = points.shape
    num_bd_points = bd_point_ind.shape[0]
    num_elements = element_pos.shape[0] - 1

    #create activity array
    activity = np.zeros((num_points, levels), dtype=np.bool)
    activity_ptr = activity.ctypes.data_as(POINTER(c_bool))

    #get input pointers
    points_ptr = points.astype(np.float64).ctypes.data_as(POINTER(c_double))
    element_pos_ptr = element_pos.astype(np.int32).ctypes.data_as(POINTER(c_int))
    element_ind_ptr = element_ind.astype(np.int32).ctypes.data_as(POINTER(c_int))
    bd_point_ind_ptr = bd_point_ind.astype(np.int32).ctypes.data_as(POINTER(c_int))

    #call C function
    lib_path = os.path.join(os.path.dirname(__file__), "libopossum_agglom.so")
    lib = CDLL(lib_path)
    lib.agglomerate(activity_ptr, points_ptr, element_pos_ptr, element_ind_ptr, bd_point_ind_ptr, spatial_dim, num_points, num_elements, num_bd_points, levels, factor)

    return torch.tensor(activity)
