'''
'''

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm as spn
import pytorch_lightning as pl

from core.torch_quadconv import PointCloudHandler
from core.quadconv_blocks import SkipBlock
from core.utilities import package_args

'''
Quadrature convolution classification model for.

Input:
    point_seq:
    data_info:
    stages: number of convolution block stages
    conv_params: convolution parameters
    optimizer: optimizer specification
    learning_rate: learning rate
'''
class Model(pl.LightningModule):

    def __init__(self,*,
            point_seq,
            data_info,
            stages,
            conv_params,
            mlp_dim,
            num_classes,
            optimizer = "Adam",
            learning_rate = 1e-2,
            **kwargs
        ):
        super().__init__()

        #save hyperparameters for checkpoint loading
        self.save_hyperparameters()

        #training hyperparameters
        self.optimizer = optimizer
        self.learning_rate = learning_rate

        #unpack data info
        input_shape = data_info['input_shape']

        #NOTE: Not sure how this works with multiple inputs
        # self.example_input_array = torch.zeros(input_shape)

        #point cloud
        self.point_cloud = PointCloudHandler(point_seq)

        #block arguments
        arg_stack = package_args(stages, conv_params)

        #build network
        self.qcnn = nn.Sequential()

        for i in range(stages):
            self.qcnn.append(SkipBlock(**arg_stack[i], **kwargs, cache=False))

        self.conv_out_shape = torch.Size((1, conv_params['out_channels'][-1], conv_params['out_points'][-1]))

        self.flat = nn.Flatten(start_dim=1, end_dim=-1)

        self.linear = nn.Sequential()

        self.linear.append(spn(nn.Linear(self.conv_out_shape.numel(), mlp_dim)))
        self.linear.append(nn.CELU())
        self.linear.append(spn(nn.Linear(mlp_dim, num_classes)))
        self.linear.append(nn.CELU())

        #dry run
        self.linear(self.flat(torch.zeros(self.conv_out_shape)))

        return

    #NOTE: Not currently used
    @staticmethod
    def add_args(parent_parser):
        parser = parent_parser.add_argument_group("AutoEncoder")

        # parser.add_argument()

        return parent_parser

    '''
    Forward pass of model.

    Input:
        x: points
        f: features

    Output:
    '''
    def forward(self, x, f):

        self.point_cloud.cache(x)

        _, x = self.qcnn((self.point_cloud, f))
        x = self.flat(x)
        output = slef.linear(x)

        return output

    '''
    '''
    def _compute_metrics(self, batch, idx):
        points, features, labels = batch

        #predict
        pred = self(points, features)

        #compute loss
        loss = F.cross_entropy(pred, labels)

        #compute accuracy
        acc = torch.mean(torch.argmax(pred, dim=1)==labels)

        return loss, acc

    '''
    Single training step.

    Input:
        batch: batch of data
        idx: batch index

    Output: pytorch loss object
    '''
    def training_step(self, batch, idx):

        loss, acc = self._compute_metrics(batch, idx)

        self.log('train_loss', loss, on_step=False, on_epoch=True)

        return loss

    '''
    Single validation_step; logs validation error.

    Input:
        batch: batch of data
        idx: batch index
    '''
    def validation_step(self, batch, idx):

        loss, acc = self._compute_metrics(batch, idx)

        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, sync_dist=True)

        return

    '''
    Single test step; logs average and max test error

    Input:
        batch: batch of data
        idx: batch index
    '''
    def test_step(self, batch, idx):

        loss, acc = self._compute_metrics(batch, idx)

        self.log('test_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, sync_dist=True)

        return

    '''
    Single prediction step.

    Input:
        batch: batch of data
        idx: batch index

    Output:
    '''
    def predict_step(self, batch, idx):
        points, features, _ = batch

        return torch.argmax(self(points, features), dim=1)

    '''
    Instantiates optimizer

    Output: pytorch optimizer
    '''
    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer)(self.parameters(), lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=250, factor=0.5)
        scheduler_config = {"scheduler": scheduler, "monitor": "val_err"}

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
