
import numpy as np
from torch_quadconv.utils.agglomeration import agglomerate

if __name__=="__main__":
    points = np.random.randn(10, 2)
    bd_points = np.random.randn(3, 2)
    e_indices = np.array([0,1,2])
    elements = np.array([0,1,2])

    activity = agglomerate(points, bd_points, e_indices, elements, 5)

    print(activity)
