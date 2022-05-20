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
        parser = parent_parser.add_argument_group("PointCloudLoader")

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
        parser = parent_parser.add_argument_group("PointCloudLoader")

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

def load_ignition_data(batch_size=16,
                        time_chunk = 1,
                        order = None,
                        split = 0.8,
                        size=25,
                        stride=25,
                        noise=False,
                        normalize = True,
                        dataloader = True,
                        center_cut = False,
                        tile = 1,
                        use_all_channels = False
                        ):

    data_path = '/home/rs-coop/Documents/Research/ASCR-Compression/QuadConv/data/ignition.npy'

    if use_all_channels:
        idx =  (0,1,2)
    else:
        idx = (0)

    mydata = np.load(data_path)
    ignition_data = np.float32(mydata[:,74:174,0:100,idx])

    if center_cut:
        ignition_data = np.float32(mydata[:,99:149,0:50,idx])

    if noise:
        ignition_data += 0.0001*np.random.randn(*ignition_data.shape)

    ignition_data = torch.from_numpy(ignition_data)

    if use_all_channels:
        ignition_data = torch.movedim(ignition_data, -1, 0)
        ignition_data = ignition_data.reshape(-1, ignition_data.shape[-2], ignition_data.shape[-1] )

    ignition_data = ignition_data.unfold(1,size,stride).unfold(2, size, stride)

    ignition_data = ignition_data.reshape(-1,size,size).reshape(-1,1,size*size)

    if normalize:

        mean_ignition =  torch.mean(ignition_data, dim=(0,1,2), keepdim=True)
        stddev_ignition = torch.sqrt(torch.var(ignition_data, dim=(0,1,2), keepdim=True))
        stddev_ignition = torch.max(stddev_ignition, torch.tensor(1e-3))

        ignition_data = (ignition_data - mean_ignition) / stddev_ignition

        max_val = torch.max(torch.abs(ignition_data))

        ignition_data = ignition_data / (max_val + 1e-4)

    if order == 'random':
        np.random.shuffle(ignition_data)

    s = ignition_data.shape

    split_num = int(np.floor(split * s[0]))

    cutoff = int(np.floor((split_num/time_chunk)))

    if dataloader:
        train_dl = DataLoader(ignition_data[0:cutoff,:,:],batch_size=batch_size,shuffle=True)

        if cutoff+1 >= s[0]:
            test_dl = None
        else:
            test_dl = DataLoader(ignition_data[cutoff+1::,:,:],batch_size=batch_size,shuffle=True)

    else:
        train_dl = ignition_data[0:cutoff,:,:]
        test_dl = ignition_data[cutoff+1::,:,:]

    return train_dl, test_dl, s[1::]
