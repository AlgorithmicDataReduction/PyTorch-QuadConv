from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl

'''
PT Lightning data module for unstructured time series data.

Input:
    data_dir: data directory
    spatial_dim: spatial dimension of data
    batch_size: batch size
    channels: which data channels to use
    normalize: whether or not to normalize the data
    split: percentage of data to use in training
    shuffle: whether or not to shuffle samples
    num_workers: number of data loading processes
    persistent_workers: whether or not to maintain data loading processes
    pin_memory: whether or not to pin data loading memory
'''
class PointCloudDataModule(pl.LightningDataModule):

    def __init__(self,*,
            data_dir,
            spatial_dim,
            batch_size,
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

    @staticmethod
    def add_args(parent_parser):
        parser = parent_parser.add_argument_group("UnstructuredDataModule")

        parser.add_argument('--data_dir', type=str)

        return parent_parser

    '''
    Transform features by selecting channels, reshaping, and normalizing.
    '''
    def transform(self, features):
        #extract channels
        if len(self.channels) != 0:
            features = features[...,self.channels]

        #normalize
        if self.normalize:
            mean = torch.mean(features, dim=(0,1), keepdim=True)
            stdv = torch.sqrt(torch.var(features, dim=(0,1), keepdim=True))

            features = (features-mean)/stdv

            features = features/(torch.max(torch.abs(features)))

        #channels first
        features = torch.movedim(features, -1, 1)

        return features

    def setup(self, stage=None):
        data_path = Path(self.data_dir)

        points_file = data_path.joinpath('points.npy')
        features_file = data_path.joinpath('features.npy')

        try:
            self.points = torch.from_numpy(np.float32(np.load(points_file)))
            features = torch.from_numpy(np.float32(np.load(features_file)))
        except Exception as e:
            raise e

        if stage == "fit" or stage is None:
            full = self.transform(features)

            train_size = int(0.8*len(full))
            val_size = len(full) - train_size

            self.train, self.val = random_split(full, [train_size, val_size])

        elif stage == "test" or stage is None:
            self.test = self.transform(features)

        elif stage == "predict" or stage is None:
            self.predict = self.transform(features)

        else:
            raise ValueError("Stage must be one of fit, test, or predict.")

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
    Agglomerate data by concatenating and bining.

    Input:
        data: all batched data
    '''
    def agglomerate(self, data):
        #first, concatenate all the batches
        data = torch.cat(data)

        #bin to transform into grid
        #NOTE: This bins value is specific to the ignition center cut data
        hist = torch.histogramdd(self.points, bins=50, weight=data)['hist']
        count = torch.histogramdd(self.points, bins=50)['hist']

        return hist/count
