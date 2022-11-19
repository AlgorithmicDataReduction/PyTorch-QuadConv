'''
'''

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl

'''
PT Lightning data module for grid based time series data.

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
        parser = parent_parser.add_argument_group("StructuredDataModule")

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

    ############################################################################

    '''
    Get a single data sample.

    NOTE: this may not be the best way to do this and it doesn't account for multichannel

    Input:
        idx: sample index
    '''
    def get_sample(self, idx):
        return self.predict[idx,...].reshape(self.data_shape)

    '''
    Get the data shape

    NOTE: Not sure if this works with multichannel
    '''
    def get_data_info(self):
        return {'input_shape': tuple([1, 1]+[self.size for i in range(self.spatial_dim)])}

    '''
    Stitch tiled data back together.

    Input:
        data: tiled data
    '''
    def _stitch(self, data):
        time_steps = data.shape[0]
        processed = torch.zeros([time_steps]+[d for d in self.data_shape])

        for t in range(time_steps):
            for i in range(self.num_tiles):
                for j in range(self.num_tiles):
                    processed[t,self.stride*i:self.stride*i+self.size,self.stride*j:self.stride*j+self.size] += data[t,i,j,:,:]

        return processed

    '''
    Agglomerate data if it was tiled by stitching and normalizing.

    Input:
        data: all batched data
    '''
    def agglomerate(self, data):
        #first, concatenate all the batches
        data = torch.cat(data)

        #if data wasn't tiled then dont bother stitching
        if self.num_tiles == 1:
            data = data.reshape(tuple([-1]+list(self.data_shape)))

        else:
            #do some other stuff
            data = data.reshape(tuple([-1]+[self.num_tiles for i in range(self.spatial_dim)]+[self.size for i in range(self.spatial_dim)]))

            #create mask
            mask = self._stitch(torch.ones(tuple([1]+[self.num_tiles for i in range(self.spatial_dim)]+[self.size for i in range(self.spatial_dim)])))

            #stich everything back together
            quilt = self._stitch(data)

            #normalize
            data =  quilt/mask

        return data

    '''
    Returns a method for plotting a set of features.
    '''
    def get_plot_func(self):
        if self.spatial_dim == 1:
            plot_func = lambda f, ax: ax.plot(f)
        elif self.spatial_dim == 2:
            plot_func = lambda f, ax: ax.imshow(f, vmin=-1, vmax=1, origin='lower')
        else:
            warnings.warn("Plotting for 3D data not supported.", )

        return plot_func
