'''
'''

import os
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl

'''

'''
class PointCloudDataModule(pl.LightningDataModule):
    def __init__(self,
                    data_dir,
                    batch_size,
                    dimension,
                    channels=(),
                    time_chunk=1,
                    normalize=True
                    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.channels = channels
        self.time_chunk = time_chunk
        # self.transforms =
        # if normalize:
        #     self.transforms =

    @staticmethod
    def add_args(parent_parser):
        parser = parent_parser.add_argument_group("PointCloudDataModule")

        # parser.add_argument()

        return parent_parser

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            full = None

            train_size = int(0.8*len(full))
            val_size = len(full) - train_size

            self.train, self.val = random_split(full, [train_size, val_size])

        elif stage == "test" or stage is None:
            self.test = None

        elif stage == "predict" or stage is None:
            self.predict = None

        else:
            raise ValueError("Stage must be one of 'fit', 'test', or 'predict'.")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size)

    def teardown(self, stage=None):
        pass

'''

'''
class GridDataModule(pl.LightningDataModule):
    def __init__(self,
                    data_dir,
                    dimension,
                    batch_size,
                    size,
                    stride,
                    channels=(),
                    time_chunk=1,
                    num_tiles=1,
                    normalize=True,
                    split=0.8,
                    shuffle=False,
                    num_workers=4
                    ):
        super().__init__()

        self.data_dir = data_dir
        self.dimension = dimension
        self.batch_size = batch_size
        self.size = size
        self.stride = stride
        self.channels = channels
        self.time_chunk = time_chunk
        self.num_tiles = num_tiles
        self.normalize = normalize
        self.split = split
        self.shuffle = shuffle
        self.num_workers = num_workers

    @staticmethod
    def add_args(parent_parser):
        parser = parent_parser.add_argument_group("GridDataModule")

        parser.add_argument('--data_dir', type=str)
        parser.add_argument('--dimension', type=int)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument('--size', type=int)
        parser.add_argument('--stride', type=int)
        parser.add_argument('--channels', type=int, nargs='+')

        return parent_parser

    def transform(self, data):
        #extract channels
        if len(self.channels)==0:
            self.channels = tuple(range(data.shape[-1]))

        data = data[...,self.channels]

        #reshape
        if len(self.channels) > 1:
            data = torch.movedim(data, -1, 0)
            data = data.reshape(-1, **data.shape[-self.dimension:])

        for i in range(1, self.dimension+1):
            data = data.unfold(i, self.size, self.stride)

        data = data.reshape(-1, self.size, self.size, self.size).reshape(-1, 1, self.size**self.dimension)

        #normalize
        #NOTE: This might need to change for 3D data
        if self.normalize:
            mean =  torch.mean(data, dim=(0,1,2), keepdim=True)
            std_dev = torch.sqrt(torch.var(data, dim=(0,1,2), keepdim=True))
            std_dev = torch.max(std_dev, torch.tensor(1e-3))

            data = (data-mean)/std_dev

            max_val = torch.max(torch.abs(data))

            data = data/(torch.max(torch.abs(data))+1e-4)

        return data

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            data = torch.from_numpy(np.float32(np.load(os.path.join(self.data_dir, 'train.npy'))))

            data = self.transform(data)
            self.data_shape = data.shape[1:]

            if self.shuffle:
                torch.shuffle(data)

            cutoff = int(int(0.8*data.shape[0])/self.time_chunk)

            self.train, self.val = data[0:cutoff,...], data[cutoff+1:,...]

        elif stage == "test" or stage is None:
            data = torch.from_numpy(np.load(os.path.join(self.data_dir, 'test.npy')))
            self.test = self.transform(data)

        elif stage == "predict" or stage is None:
            self.predict = None

        else:
            raise ValueError("Invalid stage.")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size, num_workers=self.num_workers)

    def teardown(self, stage=None):
        pass

    def get_data(self):
        return self.train, self.val, self.data_shape
