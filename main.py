from core.models import QCNN, CNN
from core.data import PointCloudDataModule, GridDataModule
from core.utilities import ProgressBar

from argparse import ArgumentParser
import yaml
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
    model_type = trainer_args.pop("model_type", None)

    if model_type == "QCNN":
        model = QCNN(**model_args)
    elif model_type == "CNN":
        model = CNN(**model_args)
    else:
        raise ValueError("Invalid model type.")

    #Setup data
    data_module = GridDataModule(**data_args)

    #Train model
    trainer = Trainer(**train_args, callbacks=[ProgressBar()])
    trainer.fit(model=model, datamodule=data_module)

'''

'''
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None, help="Named experiment")
    args, _ = parser.parse_known_args()

    #use CL arguments
    if args.experiment == None:
        train_parser = ArgumentParser()
        model_parser = ArgumentParser()
        data_parser = ArgumentParser()

        #trainer args
        train_parser.add_argument("--model_type", type=str, default='QCNN', help="QCNN or CNN")
        train_parser = Trainer.add_argparse_args(train_parser)

        #parse training args
        train_args, _ = train_parser.parse_known_args()

        #model specific args
        if train_args.model_type == "QCNN":
            model_parser = QCNN.add_args(model_parser)
        elif train_args.model_type == "CNN":
            model_parser = CNN.add_args(model_parser)
        else:
            train_parser.error("Argument '--model_type' must be one of 'QCNN' or 'CNN'.")

        #data specific args
        data_parser = PointCloudDataModule.add_args(data_parser)

        #parse remaining args
        model_args, _ = model_parser.parse_known_args()
        data_args, _ = data_parser.parse_known_args()

        #convert to dictionaries
        train_args, model_args, data_args = vars(train_args), vars(model_args), vars(data_args)

    else:
        try:
            with open(f"experiments/{args.experiment}.yaml", "r") as file:
                config = yaml.safe_load(file)

            train_args, model_args, data_args = config['train'], config['model'], config['data']

        except Exception as e:
            raise ValueError(f"Experiment {args.experiment} is invalid.")

    main(train_args, model_args, data_args)
