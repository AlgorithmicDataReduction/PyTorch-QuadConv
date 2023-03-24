import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

from torch_quadconv.utils.agglomeration import agglomerate
from torch_quadconv.mesh_handler import Elements

with h5.File("data/mesh.hdf5", "r") as file:
    points = file["points"][...]
    bd_point_ind = file["boundary_point_indices"][...]
    element_pos = file["element_positions"][...]
    element_ind = file["element_indices"][...]

print(f"Points: {points.shape}")
print(f"Boundary Point Indices: {bd_point_ind.shape[0]}, {np.max(bd_point_ind)}")
print(f"Element Positions: {element_pos.shape[0]-1}, {element_pos[-1]}, {np.max(element_pos)}")
print(f"Elements Indices: {element_ind.shape[0]}, {3*(element_pos.shape[0]-1)}, {np.max(element_ind)}")

levels = 1
factor = 100

#check aligned, writeable, and C contiguous
print(points.flags["CARRAY"])
print(bd_point_ind.flags["CARRAY"])
print(element_pos.flags["CARRAY"])
print(element_ind.flags["CARRAY"])

print(element_pos[-2])
print(element_pos[-1])

print(element_ind[-3])
print(element_ind[-2])
print(element_ind[-1])

print(points.dtype)
print(element_ind.dtype)

activity = agglomerate(points, Elements(element_pos, element_ind, bd_point_ind), levels=levels, factor=factor)

fig, axis = plt.subplots(2, 2, figsize=(10,10), constrained_layout=True)

axis[0,0].scatter(points[:,0], points[:,1], s=10)
axis[0,0].axes.get_xaxis().set_visible(False)
axis[0,0].axes.get_yaxis().set_visible(False)
axis[0,0].set_title(f"Original Mesh, {points.shape[0]} points")

for i, ax in enumerate(np.ravel(axis)[1:]):
    try:
        sub_points = points[activity[:,i]]
        ax.scatter(sub_points[:,0], sub_points[:,1], s=10)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_title(f"Stage {i+1}, {sub_points.shape[0]} points")
    except: pass

fig.suptitle(f"Agglomeration, factor {factor}")

plt.show()
