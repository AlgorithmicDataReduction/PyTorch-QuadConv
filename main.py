'''
Builds and trains a model based on input parameters, which can be specified via
command line arguments or an experiment YAML file.

Example usage:

- Run the test found in experiments/ignition.yml
    python main.py --experiment ignition
'''

from core.model import AutoEncoder
from core.data import PointCloudDataModule, GridDataModule
from core.utilities import ProgressBar, make_gif

import argparse
import yaml
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
def main(trainer_args, model_args, data_args, extra_args):
    torch.set_default_dtype(torch.float32)

    #Setup data
    data_module = GridDataModule(**data_args)
    model_args['input_shape'] = data_module.get_shape()

    #Build model
    model = AutoEncoder(**model_args)

    #Callbacks
    callbacks=[ProgressBar()]
    if extra_args['early_stopping']:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=3, strict=False))
    if train_args['enable_checkpointing']:
        callbacks.append(ModelCheckpoint(monitor="val_loss", save_last=True, save_top_k=1, mode='min'))

    #Train model
    trainer = Trainer(**train_args, callbacks=callbacks)
    trainer.fit(model=model, datamodule=data_module, ckpt_path=None)

    #Make GIF
    if extra_args['make_gif']:
        make_gif(trainer, data_module, None if train_args['enable_checkpointing'] else model)

'''
Parse arguments
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None)
    args, _ = parser.parse_known_args()

    #Look for CL arguments
    train_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    model_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    data_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    extra_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    #trainer args
    train_parser = Trainer.add_argparse_args(train_parser)

    #model specific args
    model_parser = AutoEncoder.add_args(model_parser)

    #data specific args
    data_parser = GridDataModule.add_args(data_parser)

    #extra args
    extra_parser.add_argument("--make_gif", type=bool, default=False)
    extra_parser.add_argument("--early_stopping", type=bool, default=False)

    #parse remaining args
    train_args = vars(train_parser.parse_known_args()[0])
    model_args = vars(model_parser.parse_known_args()[0])
    data_args = vars(data_parser.parse_known_args()[0])
    extra_args = vars(extra_parser.parse_known_args()[0])

    #Load YAML config
    if args.experiment != None:
        try:
            #open YAML file
            with open(f"experiments/{args.experiment}.yml", "r") as file:
                config = yaml.safe_load(file)

            #extract args
            train_args.update(config['train'])
            model_args.update(config['model'])
            data_args.update(config['data'])
            extra_args.update(config['extra'])

        except Exception as e:
            raise ValueError(f"Experiment {args.experiment} is invalid.")

    main(train_args, model_args, data_args, extra_args)
