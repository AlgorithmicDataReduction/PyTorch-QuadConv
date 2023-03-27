'''
'''

import os
from warnings import warn

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.colors import Normalize

import torch
from torch.utils.data import random_split, DataLoader

import pytorch_lightning as pl

from .mesh_handler import Elements, MeshHandler
from .utils import quadrature

'''
PT Lightning data module for unstructured point cloud data, possibly with an
associated mesh and quadrature.

Input:
    data_dir: data directory
    spatial_dim: spatial dimension of data
    num_points: number of points in data
    batch_size: batch size
    channels: which data channels to use
    quad_map: map to calculate quadrature
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
            channels,
            quad_map = None,
            quad_args = {},
            normalize = True,
            split = 0.8,
            shuffle = False,
            num_workers = 4,
            persistent_workers = True,
            pin_memory = True,
        ):
        super().__init__()

        #channels
        if isinstance(channels, list):
            assert len(channels) != 0
        elif isinstance(channels, int):
            channels = [i for i in range(channels)]
        else:
            raise ValueError("Channels must be a list or an integer")

        args = locals()
        args.pop('self')

        for key, value in args.items():
            setattr(self, key, value)

        self.train, self.val, self.test, self.predict = None, None, None, None

        return

    @staticmethod
    def add_args(parent_parser):

        parser = parent_parser.add_argument_group("MeshDataModule")

        parser.add_argument('--data_dir', type=str)

        return parent_parser

    @property
    def input_shape(self):
        return (1, len(self.channels), self.num_points)

    '''
    Load mesh features from features.hdf5 files in data directory.
    '''
    def _load_features(self):

        features_file = os.path.join(self.data_dir, "features.hdf5")

        try:
            with h5.File(features_file, 'r') as file:
                features = torch.from_numpy(file["features"][...].astype(np.float32))

                assert features.dim() == 3, f"Features has {features.dim()} dimensions, but should only have 3"

        except FileNotFoundError:
            raise Exception(f'No features have been found in: {data_path}')

        except Exception as e:
            raise e

        return features

    '''
    Transform features by selecting channels, reshaping, and normalizing.
    '''
    def _transform(self, features):

        #extract channels
        features = features[...,self.channels]

        #normalize
        if self.normalize:
            mean = torch.mean(features, dim=(0,1), keepdim=True)
            stdv = torch.sqrt(torch.var(features, dim=(0,1), keepdim=True))

            features = (features-mean)/stdv

            features = features/(torch.amax(torch.abs(features), dim=(0,1), keepdim=True))

        #channels first
        features = torch.movedim(features, -1, 1)

        return features

    def setup(self, stage=None):

        if (stage == "fit" or stage is None) and (self.train is None or self.val is None):
            #load features
            features = self._load_features()
            features = self._transform(features)

            train_size = round(self.split*len(full))
            val_size = len(full) - train_size

            self.train, self.val = random_split(features, [train_size, val_size])

        if (stage == "test" or stage is None) and self.test is None:
            #load features
            features = self._load_features()
            self.test = self._transform(features)

        if (stage == "predict" or stage is None) and self.predict is None:
            #load features
            features = self._load_features()
            self.predict = self._transform(features)

        if stage not in ["fit", "test", "predict", None]:
            raise ValueError("Stage must be one of fit, test, predict")

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
    Load mesh from mesh.hdf5 file in data directory.
    '''
    def load_mesh(self):

        mesh_file = os.path.join(self.data_dir, "mesh.hdf5")

        elements = None

        try:
            with h5.File(mesh_file, 'r') as file:

                self.points = torch.from_numpy(file["points"][...].astype(np.float32))

                assert self.num_points == self.points.shape[0], f"Expected number of points ({self.num_points}) does not match actual number ({self.points.shape[0]})"
                assert self.spatial_dim == self.points.shape[1], f"Expected spatial dimension ({self.spatial_dim}) does not match actual number ({self.points.shape[1]})"

                if self.quad_map != None:
                    weights = getattr(quadrature, self.quad_map)(self.points, self.num_points, **self.quad_args)
                else:
                    weights = None

                if 'elements' in file.keys():
                    element_pos = file['elements']["element_positions"][...]
                    element_ind = file['elements']["element_indices"][...]

                    if "boundary_point_indices" in file['elements'].keys():
                        bd_point_ind = file['elements']["boundary_point_indices"][...]
                    else:
                        bd_point_ind = None

                    elements = Elements(element_pos, element_ind, bd_point_ind)

        except FileNotFoundError:
            if self.quad_map != None:
                self.points, weights = getattr(quadrature, self.quad_map)(self.spatial_dim, self.num_points, **self.quad_args)
            else:
                raise ValueError("Quadrature map must be specified when no points file is provided")

        except Exception as e:
            raise e

        return MeshHandler(self.points, weights, elements)

    '''
    Get a single data sample.

    Input:
        idx: sample index
    '''
    def get_sample(self, idx):
        return self.predict[idx,...].squeeze()

    '''
    Agglomerate feature batches.

    Input:
        data: all batched data
    '''
    def coalesce(self, features):

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

        plot_func = None

        if self.spatial_dim == 2:
            #triangulate and save
            self._triangulation = Triangulation(self.points[:,0], self.points[:,1])
            def plot_func(f, ax, norm=Normalize, vmin=None, vmax=None, **kwargs):
                return ax.tripcolor(self._triangulation, f, norm=norm(vmin=vmin, vmax=vmax), **kwargs)

        else:
            warn("Mesh plotting only supported for 2D data.")

        return plot_func
