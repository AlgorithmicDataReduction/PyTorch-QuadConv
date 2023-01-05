import torch
from core.voxelizer import Voxelizer

N = 100
C = 3
D = 2

# points = torch.rand(N, D)
# features = torch.randn(N, C)

points = torch.tensor([[0.0,0.0], [0.25,0.7], [0.5,0.5]])
print(f"Points: {points.shape}, {points.dtype}")

features = torch.tensor([[1.0,2.0,3.0]])
features = torch.unsqueeze(features, 0)
print(f"Features: {features.shape}, {features.dtype}")

print("\n")

vox = Voxelizer(points, 0.5)

grid = vox.voxelize(features)

print(grid.shape)
print(grid)

new_features = vox.devoxelize(grid)

assert torch.allclose(new_features, features)
