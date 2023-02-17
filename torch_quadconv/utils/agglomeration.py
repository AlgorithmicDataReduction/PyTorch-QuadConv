'''
Agglomeration functions.
'''

import os
import sys
import numpy as np
from ctypes import CDLL, POINTER, c_double, c_bool, c_int

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
  activity: bool* pointer of size (num_points X stages), active points at each stage (first column is true and rest are false)
  points: double* pointer of size (num_points X spatial_dim), mesh points
  element_indices: int* pointer of size (num_elements+1)
  elements: int* pointer of size (element_indices[-1])
  boundary: int* pointer of size (num_boundary_points), indicies of boundary points in points array
  spatial_dim: int, spatial dimension of mesh points
  num_points: int, number of mesh points
  num_elements: int, number of mesh elements
  num_boundary_points: int, number of boundary points
  stages: int, number of coarsening stages
  factor: int, agglomerate division factor
*/
void agglomerate(activity, points, element_indices, elements, boundary, int spatial_dim,
                    int num_points, int num_elements, int num_boundary_points, int stages, int factor)
'''
def agglomerate(points, boundary_points, element_indices, elements, stages=1, factor=4):

    #extract some attributes
    num_points, spatial_dim = points.shape
    num_elements = element_indices.shape[0] - 1
    num_boundary_points = boundary_points.shape[0]

    #create activity array
    activity = np.zeros((num_points, stages), dtype=np.bool)
    activity_p = activity.ctypes.data_as(POINTER(c_bool))

    #get input pointers
    points_p = points.ctypes.data_as(POINTER(c_double))
    element_indices_p = element_indices.ctypes.data_as(POINTER(c_int))
    elements_p = elements.ctypes.data_as(POINTER(c_int))
    boundary_points_p = boundary_points.ctypes.data_as(POINTER(c_int))

    #call c function
    lib_path = os.path.join(os.path.dirname(__file__), "libtest.so")
    lib = CDLL(lib_path)
    lib.agglomerate(activity_p, points_p, element_indices_p, elements_p, boundary_points_p,spatial_dim, num_points, num_elements, num_boundary_points, stages, factor)

    return activity
