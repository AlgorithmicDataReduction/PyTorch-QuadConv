import numpy as np
import h5py as h5

from core.torch_quadconv.utils.agglomeration import agglomerate

with h5.File("data/ignition_mesh/mesh.hdf5", "r") as file:
    points = file["points"][...]
    bd_points = file["boundary_points"][...]
    element_indices = file["element_indices"][...]
    elements = file["elements"][...]

output = agglomerate(points, bd_points, element_indices, elements)

assert len(output) == 1
assert output[0].shape[0] == points.shape[0]
assert np.all(output[0] == points)
