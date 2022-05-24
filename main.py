from core.models import QCNN, CNN
from core.data import PointCloudDataModule, IgnitionDataModule
from core.utilities import ProgressBar

from argparse import ArgumentParser
import torch
from pytorch_lightning import Trainer, LightningDataModule

'''
Example usage of this file:
    python main.py --model_type QCNN --accelerator 'gpu' --devices 1 --experiment 'ignition test' --max_epochs 1 --logger false
'''

'''

'''
def main(trainer_args, model_args, data_args):
    torch.set_default_dtype(torch.float32)

    #Build model
    if train_args.model_type == "QCNN":
        model = QCNN(**vars(model_args))
    elif train_args.model_type == "CNN":
        model = CNN(**vars(model_args))

    #Setup data
    # data_module = PointCloudDataModule(data_args)
    data_module = IgnitionDataModule()

    #Train model
    trainer = Trainer.from_argparse_args(train_args, callbacks=[ProgressBar()])
    trainer.fit(model=model, datamodule=data_module)

'''

'''
if __name__ == "__main__":
    train_parser = ArgumentParser()
    model_parser = ArgumentParser()
    data_parser = ArgumentParser()

    #trainer args
    train_parser.add_argument("--model_type", type=str, default='QCNN', help="QCNN or CNN")
    train_parser.add_argument("--experiment", type=str, default=None, help="Named experiment")
    train_parser = Trainer.add_argparse_args(train_parser)

    #data specific args
    data_parser = PointCloudDataModule.add_args(data_parser)

    #parse trainin args
    train_args, _ = train_parser.parse_known_args()

    if train_args.model_type == "QCNN":
        model_parser = QCNN.add_args(model_parser)
    elif train_args.model_type == "CNN":
        model_parser = CNN.add_args(model_parser)
    else:
        train_parser.error("Argument '--model_type' must be one of 'QCNN' or 'CNN'.")

    #parse remaining args
    model_args, _ = model_parser.parse_known_args()
    data_args, _ = data_parser.parse_known_args()

    if train_args.experiment == None:
        pass

    elif train_args.experiment == 'ignition test':
        model_args.point_dim = 2
        model_args.latent_dim = 100
        model_args.feature_dim = [16, 1, 25*25]
        model_args.point_seq = [25, 10, 5]
        model_args.channel_seq = [1, 8, 16]
        model_args.mlp_channels = [4, 8, 4]

    else:
        raise ValueError("Argument '--experiment' is not known.")

    main(train_args, model_args, data_args)
