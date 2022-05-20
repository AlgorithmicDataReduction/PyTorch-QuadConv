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
                    data_dir:str,
                    batch_size:int,
                    dimension:int,
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
                    data_dir:str,
                    batch_size:int,
                    dimension:int,
                    channels=(),
                    time_chunk=1,
                    normalize=True
                    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.channels = channels
        self.time_chunk = time_chunk
        self.normalize = normalize

    @staticmethod
    def add_args(parent_parser):
        parser = parent_parser.add_argument_group("GridDataModule")

        parser.add_argument()

        return parent_parser

    def transform(data):
        #extract channels
        if len(self.channels)==0:
            self.channels = tuple(range(data.shape[-1]))

        data = data[...,self.channels]

        #normalize
        if normalize:
            pass


        return data

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            data = torch.from_numpy(np.load(os.path.join(self.data_dir, 'train.npy')))

            train_size = int(0.8*len(data))
            val_size = len(data) - train_size

            self.train, self.val = random_split(self.transform(data), [train_size, val_size])

        elif stage == "test" or stage is None:
            data = torch.from_numpy(np.load(os.path.join(self.data_dir, 'train.npy')))
            self.test = self.transform(data)

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

################################################################################

class IgnitionDataModule(pl.LightningDataModule):
    def __init__(self,
                    batch_size = 16,
                    time_chunk = 1,
                    order = None,
                    split = 0.8,
                    size = 25,
                    stride = 25,
                    noise = False,
                    normalize = True,
                    center_cut = False,
                    tile = 1,
                    use_all_channels = False,
                    num_workers = 4
                    ):
        super().__init__()

        self.batch_size = batch_size
        self.time_chunk = time_chunk
        self.order = order
        self.split = split
        self.size = size
        self.stride = stride
        self.noise = noise
        self.normalize = normalize
        self.center_cut = center_cut
        self.tile = tile
        self.use_all_channels = use_all_channels
        self.num_workers = num_workers

    def setup(self, stage=None):
        data_path = '/home/rs-coop/Documents/Research/ASCR-Compression/QuadConv/data/ignition.npy'

        if self.use_all_channels:
            idx =  (0,1,2)
        else:
            idx = (0)

        mydata = np.load(data_path)
        ignition_data = np.float32(mydata[:,74:174,0:100, idx])

        if self.center_cut:
            ignition_data = np.float32(mydata[:,99:149,0:50, idx])

        if self.noise:
            ignition_data += 0.0001*np.random.randn(*ignition_data.shape)

        ignition_data = torch.from_numpy(ignition_data)

        if self.use_all_channels:
            ignition_data = torch.movedim(ignition_data, -1, 0)
            ignition_data = ignition_data.reshape(-1, ignition_data.shape[-2], ignition_data.shape[-1] )

        ignition_data = ignition_data.unfold(1, self.size, self.stride).unfold(2, self.size, self.stride)

        ignition_data = ignition_data.reshape(-1, self.size, self.size).reshape(-1, 1, self.size*self.size)

        if self.normalize:
            mean_ignition =  torch.mean(ignition_data, dim=(0,1,2), keepdim=True)
            stddev_ignition = torch.sqrt(torch.var(ignition_data, dim=(0,1,2), keepdim=True))
            stddev_ignition = torch.max(stddev_ignition, torch.tensor(1e-3))

            ignition_data = (ignition_data - mean_ignition) / stddev_ignition

            max_val = torch.max(torch.abs(ignition_data))

            ignition_data = ignition_data / (max_val + 1e-4)

        if self.order == 'random':
            np.random.shuffle(ignition_data)

        s = ignition_data.shape
        split_num = int(np.floor(self.split * s[0]))
        cutoff = int(np.floor((split_num/self.time_chunk)))

        self.data_shape = s[1::]

        if stage == "fit" or stage is None:
            self.train = ignition_data[0:cutoff,:,:]

        elif stage == "test" or stage is None:
            self.test = ignition_data[cutoff+1::,:,:]

        else:
            raise ValueError("Stage is invalid.")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)

    def get_data(self):
        return self.train, self.test, self.data_shape
