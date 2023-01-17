import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import h5py as h5

with h5.File("data/ignition_mesh/mesh.hdf5", "r") as file:
    points = file["points"][...]
    bd_points = file["boundary_points"][...]

tri = Triangulation(points[:,0], points[:,1])
#
# plt.tripcolor(triangulation, bd_points, vmin=0, vmax=1, facecolors=None)

# plt.scatter(points[:,0], points[:,1], bd_points.astype(np.float32)+1.0)
#
# plt.show()

element_indices = np.array([3*i for i in range(tri.triangles.shape[0]+1)])
elements = tri.triangles.reshape((-1))

# from scipy.spatial import Delaunay
#
# tri = Delaunay(points)

# connectivity = [[] for i in range(points.shape[0])]
#
# for edge in tri.edges:
#     connectivity[edge[0]].append(edge[1])
#     connectivity[edge[1]].append(edge[0])
#
# count = 0
#
# for neighbors in connectivity:
#     adjacency_indices.append(count)
#     count += len(neighbors)
#
#     adjacency.extend(neighbors)
#
# adjacency_indices.append(count)

print(tri.triangles.shape)
print(len(element_indices))
print(len(elements))

print(element_indices[-1])

with h5.File("data/ignition_mesh/mesh.hdf5", "a") as file:

    # file.create_dataset("element_indices", data=element_indices)
    # file.create_dataset("elements", data=elements)

    file["element_indices"][...] = element_indices
    file["elements"][...] = elements

    print(file.keys())
