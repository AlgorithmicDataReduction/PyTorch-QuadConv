'''
Builds, trains, and tests an autoencoder model based on input parameters, which
are specified via a YAML configuration file and optional command line arguments.

Example usage:
    - Run the test found in experiments/tests/test.yaml
        python main.py --experiment tests/test.yaml
'''

import argparse
import yaml
import os
import platform
from pathlib import Path
from importlib import import_module

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from core.utils import Logger, make_gif

'''
Build, train, and test a model.

Input:
    experiment: experiment name
    trainer_args: PT Lightning Trainer arguments
    model_args: QCNN or CNN model arguments
    data_args: dataset arguments
    misc_args: miscellaneous arguments
'''
def main(experiment, trainer_args, model_args, data_args, misc_args):

    #callbacks
    callbacks=[]
    if trainer_args['enable_checkpointing']:
        callbacks.append(ModelCheckpoint(monitor="val_err",
                                            save_last=True,
                                            save_top_k=1,
                                            mode='min',
                                            filename='{epoch}'))
    if misc_args['early_stopping']:
        callbacks.append(EarlyStopping(monitor="val_err",
                                        patience=5,
                                        strict=False))

    #logger
    if trainer_args['logger']:
        #save the configuration details
        exp_dir, exp_name = os.path.split(experiment)
        exp_name = os.path.splitext(exp_name)[0]

        logger = Logger(save_dir=os.path.join(trainer_args['default_root_dir'], exp_dir),
                        name=exp_name, default_hp_metric=False)

        config = {'train':trainer_args, 'model':model_args, 'data':data_args, 'misc':misc_args}
        logger.log_config(config)

        #add logger to trainer args
        trainer_args['logger'] = logger

    #setup datamodule
    module = import_module('core.' + data_args.pop('module'))
    datamodule = module.DataModule(**data_args)

    #build model
    module = import_module('core.' + model_args.pop('type') + '.model')
    model = module.Model(**model_args, data_info = datamodule.get_data_info())

    #build trainer
    trainer = Trainer(**trainer_args, callbacks=callbacks)

    # if trainer_args['auto_scale_batch_size']:
    #     trainer.tune(model, datamodule=datamodule)

    #train model
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=None)

    #compute testing statistics
    if misc_args['compute_stats']:
        trainer.test(model=None if trainer_args['enable_checkpointing'] else model,
                        ckpt_path='best' if trainer_args['enable_checkpointing'] else None,
                        datamodule=datamodule)

    #make GIF
    if misc_args['make_gif']:
        make_gif(trainer, datamodule, None if trainer_args['enable_checkpointing'] else model)

    return

'''
Parse arguments from configuration file and command line. Only Lightning Trainer
command line arguments will override their config file counterparts.
'''
if __name__ == "__main__":

    #OS specific setup
    if platform.system() == 'Windows':
        os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'gloo'

    if platform.system() == 'Darwin':
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    #look for config
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None, help='YAML configuration file path relative to ./experiments with or without extension.')
    experiment = vars(parser.parse_known_args()[0])['experiment']

    #load YAML config
    if experiment != None:
        try:
            exp_path = list(Path('experiments/').rglob(experiment + '*'))

            if len(exp_path) != 1:
                raise ValueError('Experiment was not uniquely specified')

            #open YAML file
            with exp_path[0].open() as file:
                config = yaml.safe_load(file)

            #extract args
            trainer_args = config['train']
            model_args = config['model']
            data_args = config['data']
            misc_args = config['misc']

        except Exception as e:
            raise ValueError(f"Experiment {experiment} is invalid.")

    else:
        raise ValueError("An experiment configuration file must be provided.")

    #trainer args
    trainer_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    trainer_parser.add_argument("--default_root_dir", type=str)
    trainer_parser.add_argument("--max_time", type=str)

    #look for trainer CL arguments
    trainer_args.update(vars(trainer_parser.parse_known_args()[0]))

    #data args
    data_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    data_parser.add_argument('--data_dir', type=str)

    #look for data CL arguments
    data_args.update(vars(data_parser.parse_known_args()[0]))


    #run main script
    main(experiment, trainer_args, model_args, data_args, misc_args)
