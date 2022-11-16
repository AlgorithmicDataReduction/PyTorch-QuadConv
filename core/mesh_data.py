from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl

from torch_quadconv import MeshDataModule

'''
Extension of MeshDataModule with a few extra bits.
'''

class DataModule(MeshDataModule):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    '''
    Agglomerate data by concatenating and bining.

    Input:
        data: all batched data

    NOTE: I dont think this will work properly when points is None
    NOTE: There is probably a better way to do this histogram stuff, but unfortunately
    histogramdd doesn't broadcast.
    '''
    def agglomerate(self, data):
        #first, concatenate all the batches
        data = torch.cat(data)

        #reshape to channels last
        data = torch.movedim(data, 1, -1)

        #bin to transform into grid
        glom = [self.to_grid(sample) for sample in data]

        return torch.stack(glom)

    '''
    NOTE: Not accounting for multichannel
    '''
    def to_grid(self, data):
        hist = torch.histogramdd(self.points, bins=50, weight=data.squeeze())[0]
        count = torch.histogramdd(self.points, bins=50)[0]

        return hist/count

    '''
    Get a single data sample.

    Input:
        idx: sample index
    '''
    def get_sample(self, idx):
        return self.to_grid(self.predict[idx,...])
