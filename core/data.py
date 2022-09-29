'''
'''

import os
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from pathlib import Path

'''

'''
class PointCloudDataModule(pl.LightningDataModule):
    def __init__(self,
                    data_dir,
                    batch_size,
                    point_dim,
                    channels = (),
                    time_chunk = 1,
                    normalize = True
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
                    point_dim,
                    batch_size,
                    size,
                    stride,
                    flatten = True,
                    channels = (),
                    normalize = True,
                    split = 0.8,
                    shuffle = False,
                    num_workers = 4,
                    persistent_workers = True,
                    pin_memory = True
                    ):
        super().__init__()

        self.data_dir = data_dir
        self.point_dim = point_dim
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

        return parent_parser

    def transform(self, data):
        #extract channels
        if len(self.channels) != 0:
            data = data[...,self.channels]

        #per point_dim
        self.num_tiles = int(np.floor(((data.shape[1]-self.size)/self.stride)+1))

        #reshape
        if len(self.channels) != 1:
            data = torch.movedim(data, -1, 0)
            data = data.reshape(-1, **data.shape[-self.point_dim:])

        for i in range(1, self.point_dim+1):
            data = data.unfold(i, self.size, self.stride)

        data = data.reshape(tuple([-1]+[self.size for i in range(self.point_dim)])).reshape(-1, 1, self.size**self.point_dim)

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
        if self.flatten == False:
            data = data.reshape(tuple([-1, 1]+[self.size for i in range(self.point_dim)]))

        return data

    def setup(self, stage=None):
        #get all training data
        data_path = Path(self.data_dir)
        data_files = data_path.glob('*.npy')
        data_list = []

        for df in data_files:
            data_list.append( torch.from_numpy(np.float32(np.load(df))) )

        if len(data_list) == 0:
            raise Exception('No data has been found in the path given!')

        data = torch.cat(data_list, 0)

        #setup
        if stage == "fit" or stage is None:

            self.data_shape = data.shape[1:-1]

            data = self.transform(data)

            train_size = int(self.split*data.shape[0])
            val_size = data.shape[0]-train_size

            self.train, self.val = random_split(data, [train_size, val_size])

        elif stage == "test" or stage is None:

            self.test = self.transform(data)

        elif stage == "predict" or stage is None:

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

    ############################################################################

    #NOTE: this may not be the best way to do this and it doesn't account for multichannel
    def get_sample(self, idx):
        return self.predict[idx,...].reshape(self.data_shape)

    def get_shape(self):
        if self.flatten:
            return (1, 1, self.size**self.point_dim)
        else:
            return tuple([1, 1]+[self.size for i in range(self.point_dim)])

    def _stitch(self, data):
        time_steps = data.shape[0]
        processed = torch.zeros([time_steps]+[d for d in self.data_shape])

        for t in range(time_steps):
            for i in range(self.num_tiles):
                for j in range(self.num_tiles):
                    processed[t,self.stride*i:self.stride*i+self.size,self.stride*j:self.stride*j+self.size] += data[t,i,j,:,:]

        return processed

    def agglomerate(self, data):
        if self.flatten:
            data = [t.reshape(tuple([-1, 1]+[self.size for i in range(self.point_dim)])) for t in data]

        #first, concatenate all the batches
        data = torch.cat(data)

        #if data wasn't tiled then dont bother stitching
        if self.num_tiles == 1:
            return torch.swapaxes(data, 1, -1)

        #do some other stuff
        data = data.reshape(tuple([-1]+[self.num_tiles for i in range(self.point_dim)]+[self.size for i in range(self.point_dim)]))

        #create mask
        mask = self._stitch(torch.ones(tuple([1]+[self.num_tiles for i in range(self.point_dim)]+[self.size for i in range(self.point_dim)])))

        #stich everything back together
        quilt = self._stitch(data)

        return quilt/mask
