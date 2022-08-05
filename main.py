'''
Builds and trains a model based on input parameters, which can be specified via
command line arguments or an experiment YAML file.

Example usage:

- Run the test found in experiments/ignition.yml
    python main.py --experiment ignition
'''

from core.model import AutoEncoder
from core.data import PointCloudDataModule, GridDataModule
from core.utilities import Logger

import argparse
import yaml
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

'''
Build and train a model.

Input:
    trainer_args: PT Lightning Trainer arguments
    model_args: QCNN or CNN model arguments
    data_args: dataset arguments
    extra_args: other arguments that don't fit in groups above
'''
def main(args, trainer_args, model_args, data_args):
    #Setup data
    data_module = GridDataModule(**data_args)
    model_args['input_shape'] = data_module.get_shape()

    #Build model
    model = AutoEncoder(**model_args)

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
Parse arguments
'''
if __name__ == "__main__":
    #Look for CL arguments
    parser = argparse.ArgumentParser()
    train_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    model_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    data_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    #trainer args
    train_parser = Trainer.add_argparse_args(train_parser)

    #model specific args
    model_parser = AutoEncoder.add_args(model_parser)

    #data specific args
    data_parser = GridDataModule.add_args(data_parser)

    #extra args
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--make_gif", type=bool, default=False)
    parser.add_argument("--early_stopping", type=bool, default=False)

    #parse remaining args
    args = vars(parser.parse_known_args()[0])
    train_args = vars(train_parser.parse_known_args()[0])
    model_args = vars(model_parser.parse_known_args()[0])
    data_args = vars(data_parser.parse_known_args()[0])

    #Load YAML config
    if args['experiment'] != None:
        try:
            #open YAML file
            with open(f"experiments/{args['experiment']}.yml", "r") as file:
                config = yaml.safe_load(file)

            #extract args
            train_args.update(config['train'])
            model_args.update(config['model'])
            data_args.update(config['data'])
            args.update(config['extra'])

        except Exception as e:
            raise ValueError(f"Experiment {args['experiment']} is invalid.")

    main(args, train_args, model_args, data_args)
