'''
Miscellaneous utility functions.
'''

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import gif
import warnings
from typing import List

import torch
import torch.nn as nn

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only

from pathlib import Path
from importlib import import_module


'''
Generates a side-by-side GIF of the raw data and the model reconstruction for the
test dataset; logs the result.

Input:
    trainer: lightning trainer
    datamodule: data module
    model: model to use or none to use best checkpoint
'''
def make_gif(trainer, datamodule, model):
    #run on test data
    if model == None:
        results = trainer.predict(ckpt_path='best', datamodule=datamodule)
    else:
        results = trainer.predict(model=model, datamodule=datamodule)

    #agglomerate the data if necessary (if tiling was used)
    data = datamodule.agglomerate(results)

    print(data.shape)

    #if multichannel then just take first channel
    if data.shape[-1] > 1:
        data = data[...,0]

    data = data.squeeze()

    #get plotting function
    plot_func = datamodule.get_plot_func()

    if plot_func == None:
        return

    #gif frame closure
    @gif.frame
    def plot(i):
        fig, ax = plt.subplots(1, 2)

        plot_func(datamodule.get_sample(i), ax[0])
        ax[0].set_title("Uncompressed")

        im = plot_func(data[i,...], ax[1])
        ax[1].set_title("Reconstructed")

        if datamodule.spatial_dim == 2:
            # mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
            fig.colorbar(im, ax=ax.ravel().tolist(), location='bottom')

    #build frames
    frames = [plot(i) for i in range(data.shape[0])]

    #save gif
    gif.save(frames, f'{trainer.logger.log_dir}/{"last" if model else "best"}.gif', duration=50)

    return

'''
Custom Tensorboard logger; does not log hparams.yaml or the epoch metric.
'''
class Logger(TensorBoardLogger):

    def __init__(self,
            **kwargs
        ):
        super().__init__(**kwargs)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        metrics.pop('epoch', None)
        return super().log_metrics(metrics, step)

    @rank_zero_only
    def save(self):
        pass

    @rank_zero_only
    def log_config(self, config):

        filename = os.path.join(self.log_dir, 'config.yaml')
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, "w") as file:
            yaml.dump(config, file)

        return

'''
Package conv parameters.

Input:
    kwargs: keyword arguments
'''
def package_args(stages, kwargs, mirror=False):

    for key, value in kwargs.items():
        if len(value) == 1:
            kwargs[key] = value*(stages)
        elif mirror:
            value.reverse() #inplace

    arg_stack = [{ key : value[i] for key, value in kwargs.items() } for i in range(stages)]

    return arg_stack

'''
Swap input and output points and channels
'''
def swap(conv_params):
    swapped_params = conv_params.copy()

    try:
        temp = swapped_params["in_points"]
        swapped_params["in_points"] = swapped_params["out_points"]
        swapped_params["out_points"] = temp
    except:
        pass

    try:
        temp = swapped_params["in_channels"]
        swapped_params["in_channels"] = swapped_params["out_channels"]
        swapped_params["out_channels"] = temp
    except:
        pass

    return swapped_params



'''
Load in the data and model associated with a given checkpoint
'''

def load_checkpoint(path_to_checkpoint, data_path):

    #### Change the entries here to analyze a new model / dataset
    model_checkpoint_path = Path(path_to_checkpoint)
    data_path = Path(data_path)
    ###################

    model_yml = list(model_checkpoint_path.glob('config.yaml'))

    with model_yml[0].open() as file:
        config = yaml.safe_load(file)

    #extract args
    #trainer_args = config['train']
    model_args = config['model']
    data_args = config['data']
    data_args['data_dir'] = data_path
    #misc_args = config['misc']

    checkpoint = list(model_checkpoint_path.rglob('epoch=*.ckpt'))

    checkpoint_dict = torch.load(checkpoint[0])

    state_dict = checkpoint_dict['state_dict']

    #setup datamodule
    module = import_module('core.' + data_args.pop('module'))
    datamodule = module.DataModule(**data_args)
    datamodule.setup(stage='analyze')
    dataset, points = datamodule.analyze_data()

    #build model
    module = import_module('core.' + model_args.pop('type') + '.model')
    model = module.Model(**model_args, data_info = datamodule.get_data_info())

    del_list = []
    for key in state_dict:
        if 'eval_indices' in key:
            del_list.append(key)

    for key in del_list:
        del state_dict[key]

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to('cpu')

    return model, dataset, points