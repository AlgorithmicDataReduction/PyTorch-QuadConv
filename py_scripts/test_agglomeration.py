import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

from core.torch_quadconv.utils.agglomeration import agglomerate

with h5.File("data/ignition_mesh/mesh.hdf5", "r") as file:
    points = file["points"][...]
    bd_points = file["boundary_points"][...]
    element_indices = file["element_indices"][...]
    elements = file["elements"][...]

stages = 2

activity = agglomerate(points, bd_points, element_indices, elements, stages=stages)

# fig, ax = plt.subplots(1, 1, figsize=(10,10))
# ax.scatter(points[:,0], points[:,1])

# plt.show()

for i in range(stages):
    sub_points = points[activity[:,i]]

    # print(sub_points.shape[0])
    #
    # fig, ax = plt.subplots(1, 1, figsize=(10,10))
    # ax.scatter(sub_points[:,0], sub_points[:,1])
    #
    # plt.show()
