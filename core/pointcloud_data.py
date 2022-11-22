'''
'''

import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl

'''
PT Lightning data module for point cloud data.

Input:
    data_dir: data directory
    spatial_dim: spatial dimension of data
    size: number of points along each spatial dimension for a single sample
    batch_size: batch size
    stride: stride along each spatial dimension between samples
    channels: which data channels to use
    normalize: whether or not to normalize the data
    split: percentage of data to use in training
    shuffle: whether or not to shuffle samples
    num_workers: number of data loading processes
    persistent_workers: whether or not to maintain data loading processes
    pin_memory: whether or not to pin data loading memory
'''
class DataModule(pl.LightningDataModule):

    def __init__(self,*,
            data_dir,
            spatial_dim,
            size,
            batch_size,
            stride = None,
            channels = (),
            normalize = True,
            split = 0.8,
            shuffle = False,
            num_workers = 4,
            persistent_workers = True,
            pin_memory = True,
        ):
        super().__init__()

        args = locals()
        args.pop('self')

        for key, value in args.items():
            setattr(self, key, value)

        return

    '''
    Obtain datamodule CL arguments
    '''
    @staticmethod
    def add_args(parent_parser):
        parser = parent_parser.add_argument_group("PointCloudModule")

        parser.add_argument('--data_dir', type=str)

        return parent_parser

    '''
    Transform data by selecting channels, tiling, reshaping, normalizing.
    '''
    def transform(self, data):
        #extract channels
        if len(self.channels) != 0:
            data = data[...,self.channels]

            if len(self.channels) == 1:
                data = torch.unsqueeze(data, -1)

        #normalize
        if self.normalize:
            dim = tuple([i for i in range(self.spatial_dim+1)])

            mean = torch.mean(data, dim=dim, keepdim=True)
            std_dev = torch.sqrt(torch.var(data, dim=dim, keepdim=True))

            data = (data-mean)/std_dev

            data = data/(torch.max(torch.abs(data)))

        #channels first
        data = torch.movedim(data, -1, 1)

        #tile
        for i in range(2, self.spatial_dim+1):
            data = data.unfold(i, self.size, self.stride)

        #reshape
        data = data.reshape(tuple([-1, 1]+[self.size for i in range(self.spatial_dim)]))

        return data

    '''
    Setup dataset.

    Input:
        stage: lightning stage
    '''
    def setup(self, stage=None):
        #get all training data
        data_path = Path(self.data_dir)
        data_files = data_path.glob('*.npy')
        data_list = []

        #load data
        for df in data_files:
            data_list.append(torch.from_numpy(np.float32(np.load(df))))

        if len(data_list) == 0:
            raise Exception(f'No data has been found in: {data_path} !')

        #concatenate data
        data = torch.cat(data_list, 0)

        #setup
        if stage == "fit" or stage is None:
            self.data_shape = data.shape[1:-1]

            #tiling
            if self.stride == None:
                self.stride = data.shape[1]

            #NOTE: This is the number of tiles per spatial dim
            self.num_tiles = int(np.floor(((data.shape[1]-self.size)/self.stride)+1))

            #pre-process data
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

        return

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
        return

    '''
    Get all data details necessary for building network.

    Ouput: (input_shape, input_nodes, input_weights)
    '''
    def get_data_info(self):
        input_shape = (1, len(self.channels), self.num_points)

        if self.points == None:
            input_points, input_weights = newton_cotes_quad(self.spatial_dim, self.num_points)
        else:
            input_points, input_weights = self.points, self.weights

        data_info = {'input_shape': input_shape,
                        'input_nodes': input_points,
                        'input_weights': input_weights}
