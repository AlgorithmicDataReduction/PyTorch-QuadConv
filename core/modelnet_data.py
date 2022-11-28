'''
'''

import numpy as np
from pathlib import Path
from bidict import bidict
from warnings import warn

import torch
import torch.nn as nn
import torch.utils.data as td
import pytorch_lightning as pl

'''
Data loader for ModelNet dataset.

Input:

'''
class Dataset(td.Dataset):

    def __init__(self,*,
            data_dir,
            files,
            labels,
            classes
        ):

        self.data_dir = data_dir
        self.files = files
        self.labels = labels
        self.classes = classes

        return

    def _get_rotation(self, scale):
        pass

    def _transform(self, points, features):

        #sub-sample points
        idxs = torch.randperm(points.shape[0])[:1024]
        points = points[idxs,:]
        features = features[:,idxs]

        #put point cloud on unit sphere
        nn.functional.normalize(points, out=points)

        #apply random rotation

        return points, features

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        points = torch.empty(10000, 3)
        features = torch.empty(3, 10000)

        data_path = Path(self.data_dir)

        label = self.classes.inverse[self.labels[idx]]

        with open(data_path/label/(self.files[idx]+'.txt')) as file:
            for i, line in enumerate(file):
                data = line.split(',')

                points[i,:] = torch.Tensor([float(element) for element in data[:3]])
                features[:,i] = torch.Tensor([float(element) for element in data[3:]])

        points, features = self._transform(points, features)

        return points, features, label

'''
PT Lightning data module for ModelNet dataset.

Input:
    data_dir: data directory
    batch_size: batch size
    objects:
    split: percentage of data to use in training
    shuffle: whether or not to shuffle samples
    num_workers: number of data loading processes
    persistent_workers: whether or not to maintain data loading processes
    pin_memory: whether or not to pin data loading memory
'''
class DataModule(pl.LightningDataModule):

    def __init__(self,*,
            data_dir,
            batch_size,
            objects = 10,
            split = 0.8,
            shuffle = True,
            num_workers = 4,
            persistent_workers = True,
            pin_memory = True,
        ):
        super().__init__()

        args = locals()
        args.pop('self')

        for key, value in args.items():
            setattr(self, key, value)

        data_path = Path(self.data_dir)

        #get class names
        self.classes = bidict()
        with open(data_path/f'modelnet{self.objects}_shape_names.txt', 'r') as file:
            for i, line in enumerate(file):
                self.classes[line.strip()] = i

        return

    '''
    Obtain datamodule CL arguments
    '''
    @staticmethod
    def add_args(parent_parser):
        parser = parent_parser.add_argument_group("PointCloudModule")

        parser.add_argument('--data_dir', type=str)

        return parent_parser

    def _load(self, file):

        data_path = Path(self.data_dir)

        files = [[] for i in range(len(self.classes))]

        with open(data_path/f'modelnet{self.objects}_train.txt') as file:
            for line in file:
                label = line.rsplit('_', maxsplit=1)[0]
                files[self.classes[label]].append(line.strip())

        return files

    def _stratified_split(self, files):

        train_files, val_files = [], []
        train_labels, val_labels = [], []

        for label, class_files in enumerate(files):
            num_files = len(class_files)
            perm = torch.randperm(num_files)

            train_size = int(self.split*num_files)

            for idx in perm[0:train_size]:
                train_files.append(class_files[idx])
                train_labels.append(label)

            for idx in perm[train_size:]:
                val_files.append(class_files[idx])
                val_labels.append(label)

            return (train_files, train_labels), (val_files, val_labels)

    def _combine(self, files):

        test_files, test_labels = [], []

        for i, class_files in enumerate(files):
            test_files.append(class_files)
            test_labels.append([i]*len(class_files))

        return test_files, test_labels

    '''
    Setup dataset.

    Input:
        stage: lightning stage
    '''
    def setup(self, stage=None):

        #setup
        if stage == "fit" or stage is None:
            #get object files
            files = self._load(f'modelnet{self.objects}_train.txt')

            (train_files, train_labels), (val_files, val_labels) = self._stratified_split(files)

            self.train = Dataset(data_dir=self.data_dir, files=train_files, labels=train_labels, classes=self.classes)
            self.val = Dataset(data_dir=self.data_dir, files=val_files, labels=val_labels, classes=self.classes)

        elif stage == "test" or stage is None:
            #get object files
            files = self._load(f'modelnet{self.objects}_test.txt')

            (test_files, test_lables) = self._combine(self, files)

            self.test = Dataset(data_dir=self.data_dir, files=test_files, labels=test_labels, classes=self.classes)

        elif stage == "predict" or stage is None:
            warn("Prediction stage not supported.")

        else:
            raise ValueError("Invalid stage.")

        return

    def train_dataloader(self):
        return td.DataLoader(self.train,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers*self.trainer.num_devices,
                                shuffle=self.shuffle,
                                pin_memory=self.pin_memory,
                                persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return td.DataLoader(self.val,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                pin_memory=self.pin_memory,
                                persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return td.DataLoader(self.test,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                pin_memory=self.pin_memory,
                                persistent_workers=self.persistent_workers)

    def predict_dataloader(self):
        raise ValueError("Prediction stage not supported.")

    def teardown(self, stage=None):
        return
