'''
'''

import torch
from torch import nn
import pytorch_lightning as pl

from torch_quadconv import MeshHandler

from .modules import Encoder, Decoder
from core.utilities import SobolevLoss

'''


Input:
'''
class Model(pl.LightningModule):

    def __init__(self):
        super().__init__()
