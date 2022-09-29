'''
Builds and trains a model based on input parameters, which are specified via a
YAML configuration file and optional command line arguments.

Example usage:

- Run the test found in experiments/test.yaml
    python main.py --experiment test.yaml
'''

from core.model import AutoEncoder
from core.data import PointCloudDataModule, GridDataModule
from core.utilities import Logger, make_gif

import argparse
import yaml
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

'''
Build and train a model.

Input:
    experiment: experiment name
    trainer_args: PT Lightning Trainer arguments
    model_args: QCNN or CNN model arguments
    data_args: dataset arguments
    extra_args: other arguments
'''
def main(experiment, trainer_args, model_args, data_args, extra_args):
    #Setup data
    datamodule = GridDataModule(**data_args)
    model_args['input_shape'] = datamodule.get_shape()

    #Build model
    model = AutoEncoder(**model_args)

    #Callbacks
    callbacks=[]
    if trainer_args['enable_checkpointing']:
        callbacks.append(ModelCheckpoint(monitor="val_err",
                                            save_last=True,
                                            save_top_k=1,
                                            mode='min',
                                            filename='{epoch}'))
    if extra_args['early_stopping']:
        callbacks.append(EarlyStopping(monitor="val_err",
                                        patience=5,
                                        strict=False))

    #Logger
    if trainer_args['logger']:
        dir, name = os.path.split(experiment)
        name = os.path.splitext(name)[0]

        logger = Logger(save_dir=os.path.join(trainer_args['default_root_dir'], dir),
                        name=name, default_hp_metric=False)

        filename = os.path.join(logger.log_dir, 'config.yaml')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as file:
            yaml.dump(config, file)

        #add logger to trainer args
        trainer_args['logger'] = logger

    #Train model
    trainer = Trainer(**trainer_args, callbacks=callbacks)

    # if trainer_args['auto_scale_batch_size']:
    #     trainer.tune(model, datamodule=datamodule)

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=None)

    #Make GIF
    if extra_args['make_gif']:
        make_gif(trainer, datamodule, None if trainer_args['enable_checkpointing'] else model)

    #compute stats
    if extra_args['compute_stats']:
        trainer.test(model=None if trainer_args['enable_checkpointing'] else model,
                        ckpt_path='best' if trainer_args['enable_checkpointing'] else None,
                        datamodule=datamodule)

'''
Parse arguments
'''
if __name__ == "__main__":
    #Look for config
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None)
    args = vars(parser.parse_known_args()[0])

    #Load YAML config
    if args['experiment'] != None:
        try:
            #open YAML file
            with open(f"experiments/{args['experiment']}", "r") as file:
                config = yaml.safe_load(file)

            #extract args
            trainer_args = config['train']
            model_args = config['model']
            data_args = config['data']
            extra_args = config['extra']

        except Exception as e:
            raise ValueError(f"Experiment {args['experiment']} is invalid.")
    else:
        raise ValueError("An experiment configuration file must be provided.")

    #trainer args
    trainer_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    trainer_parser.add_argument("--default_root_dir", type=str)
    trainer_parser.add_argument("--max_time", type=str)

    #model specific args
    model_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    model_parser = AutoEncoder.add_args(model_parser)

    #data specific args
    data_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    data_parser = GridDataModule.add_args(data_parser)

    #look for other CL arguments
    trainer_args.update(vars(trainer_parser.parse_known_args()[0]))
    model_args.update(vars(model_parser.parse_known_args()[0]))
    data_args.update(vars(data_parser.parse_known_args()[0]))

    #run main script
    main(args['experiment'], trainer_args, model_args, data_args, extra_args)
