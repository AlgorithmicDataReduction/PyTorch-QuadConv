import numpy as np
import torch
import gif
import matplotlib.pyplot as plt
from pathlib import Path
from core.model import AutoEncoder
from core.data import GridDataModule
import argparse

import yaml


args = {}
args['experiment'] = 'simple_transport_cnn_single'

data_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

#data specific args
data_parser = GridDataModule.add_args(data_parser)

data_args = vars(data_parser.parse_known_args()[0])

ex_path = Path(f'./experiments/simple_transport_cnn.yml')

with ex_path.open() as file:
    config = yaml.safe_load(file)

data_args.update(config['data'])

dm = GridDataModule(**data_args)

dm.setup(stage='test')

# dm.persistent_workers = False
# dm.pin_memory = False
# dm.num_workers = False
# dm.batch_size = 100
# dm.shuffle = False

train_dl = dm.test_dataloader()

@gif.frame
def plot(d):
    plt.plot(d)
    plt.ylim(-0.5,1.5)

frames = []

for d in train_dl:
    frames.extend([plot(d[i,0,:].reshape(-1,1) ) for i in range(d.shape[0])])

gif.save(frames, f'simple_transport_cnn.gif', duration=100)


# model = AutoEncoder.load_from_checkpoint(".\lightning_logs\simple_transport_cnn\checkpoints\epoch=65.ckpt", input_shape=(1,1,100))
#
# # disable randomness, dropout, etc...
# model.eval()
# model.to('cpu')
#
#
# for d in train_dl:
#
#     processed_data = model(d).detach().numpy()
#
#     frames = [plot(processed_data[i,0,:].reshape(-1,1)) for i in range(processed_data.shape[0])]
#
#     gif.save(frames, f'simple_transport_cnn_processed.gif', duration=50)
