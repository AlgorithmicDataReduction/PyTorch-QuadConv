'''
Builds and trains a model based on input parameters, which can be specified via
command line arguments or an experiment YAML file.

Example usage:

- Run the test found in experiments/ignition.yml
    python main.py --experiment ignition
'''

from core.models import AutoEncoder
from core.data import PointCloudDataModule, GridDataModule
from core.utilities import ProgressBar

from argparse import ArgumentParser
import yaml
import torch
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

'''
Build and train a model.

Input:
    trainer_args: PT Lightning Trainer arguments
    model_args: QCNN or CNN model arguments
    data_args: dataset arguments
'''
def main(trainer_args, model_args, data_args):
    torch.set_default_dtype(torch.float32)

    #Build model
    model = AutoEncoder(**model_args)

    #Setup data
    data_module = GridDataModule(**data_args)

    #Train model
    callbacks=[ProgressBar(),
                ModelCheckpoint(monitor="val_loss", save_last=True, save_top_k=1, mode='min'),
                EarlyStopping(monitor="val_loss", patience=3, strict=False)]

    trainer = Trainer(**train_args, callbacks=callbacks)
    trainer.fit(model=model, datamodule=data_module, ckpt_path=None)

'''
Parse arguments
'''
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None, help="Named experiment")
    args, _ = parser.parse_known_args()

    #use CL config
    if args.experiment == None:
        train_parser = ArgumentParser()
        model_parser = ArgumentParser()
        data_parser = ArgumentParser()

        #trainer args
        train_parser = Trainer.add_argparse_args(train_parser)

        #model specific args
        model_parser = AutoEncoder.add_args(model_parser)

        #data specific args
        data_parser = PointCloudDataModule.add_args(data_parser)

        #parse remaining args
        train_args, _ = train_parser.parse_known_args()
        model_args, _ = model_parser.parse_known_args()
        data_args, _ = data_parser.parse_known_args()

        #convert to dictionaries
        train_args, model_args, data_args = vars(train_args), vars(model_args), vars(data_args)

    #use YAML config
    else:
        try:
            #open YAML file
            with open(f"experiments/{args.experiment}.yml", "r") as file:
                config = yaml.safe_load(file)

            #extract args
            train_args, model_args, data_args = config['train'], config['model'], config['data']

        except Exception as e:
            raise ValueError(f"Experiment {args.experiment} is invalid.")

    main(train_args, model_args, data_args)
