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
                    flatten=True,
                    channels=(),
                    normalize=True,
                    split=0.8,
                    shuffle=False,
                    num_workers=4,
                    persistent_workers=True,
                    pin_memory=True
                    ):
        super().__init__()

        self.data_dir = data_dir
        self.dimension = dimension
        self.batch_size = batch_size
        self.size = size
        self.stride = stride
        self.flatten = flatten
        self.channels = channels
        self.normalize = normalize
        self.split = split
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory

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
        if len(self.channels) != 0:
            data = data[...,self.channels]

        #NOTE: This makes a lot of assumptions
        self.num_tiles = int((data.shape[1]/self.size))

        #reshape
        if len(self.channels) != 1:
            data = torch.movedim(data, -1, 0)
            data = data.reshape(-1, **data.shape[-self.dimension:])

        for i in range(1, self.dimension+1):
            data = data.unfold(i, self.size, self.stride)

        data = data.reshape(tuple([-1]+[self.size for i in range(self.dimension)])).reshape(-1, 1, self.size**self.dimension)

        #normalize
        #NOTE: This might need to change for 3D data
        if self.normalize:
            mean = torch.mean(data, dim=(0,1,2), keepdim=True)
            std_dev = torch.sqrt(torch.var(data, dim=(0,1,2), keepdim=True))
            std_dev = torch.max(std_dev, torch.tensor(1e-3))

            data = (data-mean)/std_dev

            max_val = torch.max(torch.abs(data))

            data = data/(torch.max(torch.abs(data))+1e-4)

        #NOTE: It would be better if we normalized before flattening
        if not self.flatten:
            data = data.reshape(tuple([-1, 1]+[self.size for i in range(self.dimension)]))

        return data

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            data = torch.from_numpy(np.float32(np.load(os.path.join(self.data_dir, 'train.npy'))))
            self.train_shape = data.shape

            data = self.transform(data)

            train_size = int(self.split*data.shape[0])
            val_size = data.shape[0]-train_size

            self.train, self.val = random_split(data, [train_size, val_size])

        elif stage == "test" or stage is None:
            data = torch.from_numpy(np.float32(np.load(os.path.join(self.data_dir, 'train.npy'))))
            self.train_shape = data.shape

            self.test = self.transform(data)

        elif stage == "predict" or stage is None:
            data = torch.from_numpy(np.float32(np.load(os.path.join(self.data_dir, 'train.npy'))))

            self.predict = self.transform(data)

        else:
            raise ValueError("Invalid stage.")

    def train_dataloader(self):
        return DataLoader(self.train,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers*self.trainer.num_devices,
                            shuffle=self.shuffle,
                            pin_memory=self.pin_memory,
                            persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory,
                            persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.test,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory,
                            persistent_workers=self.persistent_workers)

    def predict_dataloader(self):
        return DataLoader(self.predict,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory,
                            persistent_workers=self.persistent_workers)

    def teardown(self, stage=None):
        pass

    def get_shape(self):
        if self.flatten:
            return (1, 1, self.size**self.dimension)
        else:
            return tuple([1, 1]+[self.size for i in range(self.dimension)])

    def agglomerate(self, data):
        if self.flatten:
            data = [t.reshape(tuple([-1, 1]+[self.size for i in range(self.dimension)])) for t in data]

        #first, concatenate all the batches
        data = torch.cat(data)

        #do some other stuff
        data = data.reshape(-1, self.num_tiles, self.num_tiles, self.size, self.size)

        #reshape to time X space X channels
        processed = torch.zeros(450, self.size*self.num_tiles, self.size*self.num_tiles)
        for i in range(450):
            for j in range(self.num_tiles):
                for k in range(self.num_tiles):
                    processed[i,self.size*j:self.size*(j+1),self.size*k:self.size*(k+1)] = data[i,j,k,:,:]

        return processed
