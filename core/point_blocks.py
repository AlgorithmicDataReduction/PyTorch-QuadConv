'''
'''

import torch
import torch.nn as nn

'''
'''
class PointBlock(nn.Module):

    def __init__(self,*,
        in_channels,
        out_channels,
        activation = nn.CELU,
        **kwargs):
        super().__init__()

        self.shared_mlp = nn.Conv1d(in_channels, out_channels, 1)
        self.norm = nn.InstanceNorm1d(out_channels, affine=True)
        self.activation = activation()

    def forward(self, data):
        out = self.shared_mlp(data)
        out = self.norm(out)
        out = self.activation(out)

        return out
