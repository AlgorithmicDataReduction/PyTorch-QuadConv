from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl

from core.utilities import newton_cotes_quad

'''
PT Lightning data module for unstructured time series data.

Input:
    data_dir: data directory
    spatial_dim: spatial dimension of data
    num_points: number of points in data
    batch_size: batch size
    channels: which data channels to use
    quad_map:
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
            num_points,
            batch_size,
            channels = (),
            quad_map = None,
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

    def load(self):
        data_path = Path(self.data_dir)

        #look for points file
        points_file = data_path.joinpath('points.npy')
        try:
            self.points = torch.from_numpy(np.float32(np.load(points_file)))

            assert self.points.shape[0] == self.num_points
            assert self.points.shape[1] == self.spatial_dim

        except FileNotFoundError:
            self.points = None

        except Exception as e:
            raise e

        #glob features file
        features_files = data_path.glob('features*.npy')
        features = []

        #open and extract all features
        for file in features_files:
            try:
                features.append(torch.from_numpy(np.float32(np.load(file))))
            except Exception as e:
                raise e

        #no features found
        if len(features) == 0:
            raise Exception(f'No features have been found in: {data_path} !')

        #concatenate features
        features = torch.cat(features, 0)

        return features

    def setup(self, stage=None):
        #load features
        features = self.load()

        assert features.dim() == 3

        #setup
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
    Get all data details necessary for building network.

    Ouput: (input_shape, input_points, quad_map)
    '''
    def get_data_info(self):
        input_shape = (1, len(self.channels), self.num_points)

        if self.points == None:
            input_points = newton_cotes_quad(self.spatial_dim, self.num_points)[0]
        else:
            input_points = self.points

        return {'input_shape': input_shape, 'input_points': input_points, 'quad_map': self.quad_map}

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
