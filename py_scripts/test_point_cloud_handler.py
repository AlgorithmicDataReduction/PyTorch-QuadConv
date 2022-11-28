import torch
from core.torch_quadconv.point_cloud_handler import PointCloudHandler

if __name__=="__main__":

    point_seq = [100, 50, 25]

    pc = PointCloudHandler(point_seq)

    pc.cache(torch.randn(8, 100, 2))

    print(pc.input_points.shape)
    print(pc.output_points.shape)
    print(pc.weights.shape)

    pc.step()

    print(pc._current_index)

    print(pc.input_points.shape)
    print(pc.output_points.shape)
    print(pc.weights.shape)

    pc.step()

    print(pc._current_index)

    print(pc.input_points)
    print(pc.output_points)
    print(pc.weights.shape)
