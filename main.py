'''
Builds and trains a model based on input parameters, which can be specified via
command line arguments or an experiment YAML file.

Example usage:

- Run the test found in experiments/ignition.yml
    python main.py --experiment ignition
'''

from core.data import PointCloudDataModule, GridDataModule
from core.utilities import Logger, make_gif

import argparse
import yaml
import os
import platform
from importlib import import_module
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DeepSpeedStrategy

'''
Build and train a model.

Input:
    train_args: PT Lightning Trainer arguments
    model_args: QCNN or CNN model arguments
    data_args: dataset arguments
    extra_args: other arguments that don't fit in groups above
'''
def main(args, train_args, model_args, data_args):
    #Setup data
    data_module = GridDataModule(**data_args)
    model_args['input_shape'] = data_module.get_shape()

    #Build model
    model = model_module.AutoEncoder(**model_args)

    #Callbacks
    callbacks=[]
    if train_args['enable_checkpointing']:
        callbacks.append(ModelCheckpoint(monitor="val_err",
                                            save_last=True,
                                            save_top_k=1,
                                            mode='min',
                                            filename='{epoch}'))
    if args['early_stopping']:
        callbacks.append(EarlyStopping(monitor="val_err",
                                        patience=5,
                                        strict=False))

    #Logger
    if train_args['logger']:
        if train_args['default_root_dir'] is None:
            train_args['default_root_dir'] = os.getcwd()

        train_args['logger'] = Logger(save_dir=train_args['default_root_dir'],
                                        version=args['experiment'],
                                        default_hp_metric=False)

    #Train model
    trainer = Trainer(**train_args, callbacks=callbacks)

    # if train_args['auto_scale_batch_size']:
    #     trainer.tune(model, datamodule=data_module)

    trainer.fit(model=model, datamodule=data_module, ckpt_path=None)

    #Make GIF
    if args['make_gif']:
        make_gif(trainer, data_module, None if train_args['enable_checkpointing'] else model)

'''
Parse arguments -- note that .yaml args override command line args
'''
if __name__ == "__main__":

    if platform.system() == 'Windows':
        os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'gloo'

    #Look for CL arguments
    parser = argparse.ArgumentParser()
    train_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    model_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    data_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)


    #general args
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--make_gif", type=bool, default=False)
    parser.add_argument("--early_stopping", type=bool, default=False)

    #parse general args
    args, remaining_args = parser.parse_known_args()

    args = vars(args)

    #Load YAML config
    if args['experiment'] != None:
        try:

            exp_path = list(Path('experiments/').rglob(args['experiment'] + '*'))

            if len(exp_path) != 1:
                raise ValueError('Experiment was not uniquely specified')

            #open YAML file
            with exp_path[0].open() as file:
                config = yaml.safe_load(file)

            args.update(config['extra'])

        except Exception as e:
            raise ValueError(f"Experiment {args['experiment']} is invalid.")
    else:
        raise ValueError('A YAML must be provided to the --experiment flag')

    
    model_module = import_module('core.' + config['model']['model_type'])

    #trainer args
    train_parser = Trainer.add_argparse_args(train_parser)

    #model specific args
    model_parser = model_module.AutoEncoder.add_args(model_parser)

    #data specific args
    data_parser = GridDataModule.add_args(data_parser)

    #parse remaining args
    train_args = vars(train_parser.parse_known_args(remaining_args)[0])
    model_args = vars(model_parser.parse_known_args(remaining_args)[0])
    data_args = vars(data_parser.parse_known_args(remaining_args)[0])

    #extract args --possible avenue for unchecked args to do something strange
    train_args.update(config['train'])
    model_args.update(config['model'])
    data_args.update(config['data'])

    main(args, train_args, model_args, data_args)
