'''
Miscellaneous utility functions.
'''

import torch
import torch.nn as nn

################################################################################

'''
Module wrapper for sin function; allows it to operate as a layer.
'''
class Sin(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)
