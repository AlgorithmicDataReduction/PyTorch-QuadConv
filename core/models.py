'''

'''
import torch
from torch import nn, optim
from torch.nn.utils.parametrizations import spectral_norm
import pytorch_lightning as pl

from .quadconv import QuadConvBlock

'''
Encoder module
'''
class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        point_dim = kwargs['point_dim']
        latent_dim = kwargs['latent_dim']
        point_seq = kwargs['point_seq']
        channel_seq = kwargs['channel_seq']
        mlp_channels = kwargs['mlp_channels']
        quad_type = kwargs['quad_type']
        mlp_mode = kwargs['mlp_mode']
        use_bias = kwargs['use_bias']

        #activations
        self.forward_activation = kwargs['forward_activation']
        self.latent_activation = kwargs['latent_activation']

        #build network
        self.cnn = nn.Sequential()

        for i in range(len(point_seq)-1):
            self.cnn.append(QuadConvBlock(point_dim,
                                            channel_seq[i],
                                            channel_seq[i+1],
                                            N_in = point_seq[i],
                                            N_out = point_seq[i+1],
                                            mlp_channels = mlp_channels,
                                            quad_type = quad_type,
                                            mlp_mode = mlp_mode,
                                            use_bias = use_bias,
                                            activation1 = self.forward_activation,
                                            activation2 = self.forward_activation
                                            ))

        self.flat = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear_down = spectral_norm(nn.Linear(channel_seq[-1]*(point_seq[-1]**point_dim), latent_dim))
        self.linear_down2 = spectral_norm(nn.Linear(latent_dim, latent_dim))

    def forward(self, x):
        x = self.cnn(x)
        x = self.flat(x)
        x = self.latent_activation(self.linear_down(x))
        output = self.latent_activation(self.linear_down2(x))

        return output

'''
Decoder module
'''
class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        point_dim = kwargs['point_dim']
        latent_dim = kwargs['latent_dim']
        point_seq = kwargs['point_seq']
        channel_seq = kwargs['channel_seq']
        mlp_channels = kwargs['mlp_channels']
        quad_type = kwargs['quad_type']
        mlp_mode = kwargs['mlp_mode']
        use_bias = kwargs['use_bias']

        #activations
        self.forward_activation = kwargs['forward_activation']
        self.latent_activation = kwargs['latent_activation']
        self.output_activation = kwargs['output_activation']

        #build network
        self.unflat = nn.Unflatten(1, [channel_seq[-1], point_seq[-1]**point_dim])
        self.linear_up = spectral_norm(nn.Linear(latent_dim, latent_dim))
        self.linear_up2 = spectral_norm(nn.Linear(latent_dim, (point_seq[-1]**point_dim)*channel_seq[-1]))

        self.cnn = nn.Sequential()

        for i in range(len(point_seq)-1, 0, -1):
            self.cnn.append(QuadConvBlock(point_dim,
                                            channel_seq[i],
                                            channel_seq[i-1],
                                            N_in = point_seq[i],
                                            N_out = point_seq[i-1],
                                            adjoint = True,
                                            mlp_channels = mlp_channels,
                                            quad_type = quad_type,
                                            mlp_mode = mlp_mode,
                                            use_bias = use_bias,
                                            activation1 = self.forward_activation if i!=1 else nn.Identity(),
                                            activation2 = self.forward_activation
                                            ))

    def forward(self, x):
        x = self.latent_activation(self.linear_up(x))
        x = self.latent_activation(self.linear_up2(x))
        x = self.unflat(x)
        output = self.output_activation(self.cnn(x))

        return output

'''
Quadrature convolution based autoencoder

    point_dim : space dimension (e.g. 3D)
    latent_dim : dimension of latent representation
    point_seq : number of points along each dimension
    channel_seq :
    mlp_channels :
'''
class QCNN(pl.LightningModule):
    def __init__(self,
                    point_dim,
                    latent_dim,
                    point_seq,
                    channel_seq,
                    mlp_channels,
                    quad_type = 'gauss',
                    mlp_mode = 'single',
                    use_bias = False,
                    forward_activation = nn.CELU(alpha=1),
                    latent_activation = nn.CELU(alpha=1),
                    output_activation = nn.Tanh(),
                    loss_fn = nn.MSELoss(),
                    noise_scale = 0.0
                    ):
        super().__init__()

        #save model hyperparameters under self.hparams
        self.save_hyperparameters(ignore=['noise_scale',
                                            'loss_fn',
                                            'forward_activation',
                                            'latent_activation',
                                            'output_activation'])

        #model pieces
        self.encoder = Encoder(**self.hparams,
                                forward_activation=forward_activation,
                                latent_activation=latent_activation)
        self.decoder = Decoder(**self.hparams,
                                forward_activation=forward_activation,
                                latent_activation=latent_activation,
                                output_activation=output_activation)

        #training hyperparameters
        self.loss_fn = loss_fn
        self.noise_scale = noise_scale

    @staticmethod
    def add_args(parent_parser):
        parser = parent_parser.add_argument_group("QCNN")

        parser.add_argument('--point_dim', type=int)
        parser.add_argument('--latent_dim', type=int)
        parser.add_argument('--point_seq', type=int, nargs='+')
        parser.add_argument('--channel_seq', type=int, nargs='+')
        parser.add_argument('--mlp_channels', type=int, nargs='+')

        return parent_parser

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def training_step(self, batch, idx):
        #encode and add noise to latent rep.
        latent = self.encoder(batch)
        latent += self.noise_scale*torch.randn_like(latent[0,:])

        #decode
        pred = self.decoder(latent)

        #compute loss
        loss = self.loss_fn(pred, batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, idx):
        pred = self(batch)
        loss = self.loss_fn(pred, batch)

        self.log('val_loss', loss)

    def test_step(self, batch, idx):
        pred = self(batch)
        loss = self.loss_fn(pred, batch)

        self.log('test_loss', loss)

    def predict_step(self, batch, idx):
        return self(batch)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-2)
        # lr_scheduler = torch.optim.lr_scheduler.
        return optimizer
        # return [optimizer], [lr_scheduler]

'''
Standard convolutional autoencoder
'''
class CNN(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        #Save model hyperparameters under self.hparams
        self.save_hyperparameters(ignore=[])

    @staticmethod
    def add_args(parent_parser):
        parser = parent_parser.add_argument_group("CNN")

        parser.add_argument()

        return parent_parser

    def forward(self, x):
        pass

    def training_step(self, batch, idx):
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer
