'''
'''

from warnings import warn
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from .utilities import newton_cotes_quad

'''
PT Lightning data module for unstructured point cloud data, possibly with an
associated mesh and quadrature.

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
class MeshDataModule(pl.LightningDataModule):

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

        self.train, self.val, self.test, self.predict = None, None, None, None

        self.setup("fit")

        return

    @staticmethod
    def add_args(parent_parser):
        parser = parent_parser.add_argument_group("MeshDataModule")

        parser.add_argument('--data_dir', type=str)

        return parent_parser

    '''
    Transform features by selecting channels, reshaping, and normalizing.
    '''
    def _transform(self, features):
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

    def _load_data(self):
        data_path = Path(self.data_dir)

        #look for points file
        points_file = data_path.joinpath('points.npy')
        try:
            self.points = torch.from_numpy(np.float32(np.load(points_file)))

            assert self.points.shape[0] == self.num_points
            assert self.points.shape[1] == self.spatial_dim

            #look for weights file
            points_file = data_path.joinpath('weights.npy')
            try:
                self.points = torch.from_numpy(np.float32(np.load(points_file)))

                assert self.weights.shape[0] == self.num_points

            except FileNotFoundError:
                self.weights = None

            except Exception as e:
                raise e

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
        features = self._load_data()

        assert features.dim() == 3

        #setup
        if (stage == "fit" or stage is None) and not self.train and not self.val:
            full = self._transform(features)

            train_size = int(0.8*len(full))
            val_size = len(full) - train_size

            self.train, self.val = random_split(full, [train_size, val_size])

        elif (stage == "test" or stage is None) and not self.test:
            self.test = self._transform(features)

        elif (stage == "predict" or stage is None) and not self.predict:
            self.predict = self._transform(features)

        elif stage not in ["fit", "test", "predict"]:
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

        return data_info

    def _to_grid(self, features):
        grid_shape = [int(features.shape[0]**(1/self.spatial_dim))]*self.spatial_dim
        return features.reshape(*grid_shape, features.shape[-1])

    '''
    Get a single data sample.

    Input:
        idx: sample index
    '''
    def get_sample(self, idx):
        return torch.movedim(self.predict[idx,...], 0, -1)

    '''
    Agglomerate feature batches.

    Input:
        data: all batched data
    '''
    def agglomerate(self, features):

        #first, concatenate all the batches
        features = torch.cat(features)

        #reshape to channels last
        features = torch.movedim(features, 1, -1)

        return features

    '''
    Returns a method for plotting a set of features.

    NOTE: In the future, this should use the adjacency
    '''
    def get_plot_func(self):

        if self.points == None:
            if self.spatial_dim == 1:
                plot_func = lambda f, ax: ax.plot(f)
            elif self.spatial_dim == 2:
                plot_func = lambda f, ax: ax.imshow(self._to_grid(f), vmin=-1, vmax=1, origin='lower')
            else:
                warn("Plotting for 3D data not supported.", )
        else:
            if self.spatial_dim == 2:
                #triangulate and save
                self._triangulation = plt.tri.Triangulation(self.points[:,0], self.points[:,])
                plot_func = lambda f, ax: ax.tripcolor(self._triangulation, f)
            else:
                warn("Plotting for non 2D data not supported.", )

        return plot_func
