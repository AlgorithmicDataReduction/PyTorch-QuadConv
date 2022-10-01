'''

'''

import torch
import torch.nn as nn

from .quadconv import QuadConvLayer

'''
QuadConvLayer block

Input:
    spatial_dim: space dimension
    num_points_in: number of input points
    num_points_out: number of output points
    in_channels: input feature channels
    out_channels: output feature channels
    filter_seq: complexity of point to filter operation
    use_bias: add bias term to output of layer
    adjoint: downsample or upsample
    activation1:
    activation2:
'''
class QuadConvBlock(nn.Module):

    use_bias = False
    adjoint = False
    activation1 = nn.CELU(alpha=1)
    activation2 = nn.CELU(alpha=1)

    def __init__(self,
            spatial_dim,
            num_points_in,
            num_points_out,
            in_channels,
            out_channels,
            filter_seq,
            **kwargs
        ):
        super().__init__()

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        if self.adjoint:
            conv1_point_num = num_points_out
            conv1_channel_num = out_channels
        else:
            conv1_point_num = num_points_in
            conv1_channel_num = in_channels

        self.conv1 = QuadConvLayer(spatial_dim,
                                    num_points_in = conv1_point_num,
                                    num_points_out = conv1_point_num,
                                    in_channels = conv1_channel_num,
                                    out_channels = conv1_channel_num,
                                    filter_seq = filter_seq,
                                    use_bias = self.use_bias
                                    )
        self.norm1 = nn.BatchNorm1d(conv1_channel_num)

        self.conv2 = QuadConvLayer(spatial_dim,
                                    num_points_in = num_points_in,
                                    num_points_out = num_points_out,
                                    in_channels = in_channels,
                                    out_channels = out_channels,
                                    filter_seq = filter_seq,
                                    use_bias = self.use_bias
                                    )
        self.norm2 = nn.BatchNorm1d(out_channels)

        return

    '''
    Forward mode
    '''
    def forward_op(self, data):
        x = data

        x1 = self.conv1(x)
        x1 = self.activation1(self.norm1(x1))
        x1 = x1 + x

        x2 = self.conv2(x1)
        x2 = self.activation2(self.norm2(x2))

        return x2

    '''
    Adjoint mode
    '''
    def adjoint_op(self, data):
        x = data

        x2 = self.conv2(x)
        x2 = self.activation2(self.norm2(x2))

        x1 = self.conv1(x2)
        x1 = self.activation1(self.norm1(x1))
        x1 = x1 + x2

        return x1

    '''
    Apply operator
    '''
    def forward(self, data):
        if self.adjoint:
            output = self.adjoint_op(data)
        else:
            output = self.forward_op(data)

        return output

################################################################################

'''
QuadConvLayer + Pooling block

Input:
    spatial_dim: space dimension
    num_points_in: number of input points
    num_points_out: number of output points
    in_channels: input feature channels
    out_channels: output feature channels
    adjoint: downsample or upsample
    quad_type: quadrature type
    mlp_mode: ?
    use_bias: add bias term to output of layer
    activation1:
    activation2:
'''
class PoolQuadConvBlock(nn.Module):

    adjoint = False
    mlp_mode = 'share_in'
    quad_type = 'newton_cotes'
    composite_quad_order = 2
    use_bias = True
    activation1 = nn.CELU(alpha=1)
    activation2 = nn.CELU(alpha=1)

    def __init__(self,
            spatial_dim,
            num_points_in,
            num_points_out,
            in_channels,
            out_channels,
            filter_seq,
            **kwargs,
        ):
        super().__init__()

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # channel flexibility can be added later
        assert in_channels == out_channels

        # make sure the user sets this as no default is provided
        assert spatial_dim > 0

        layer_lookup = { 1 : (nn.MaxPool1d),
                         2 : (nn.MaxPool2d),
                         3 : (nn.MaxPool3d),
        }

        Pool = layer_lookup[spatial_dim]

        if self.adjoint:
            self.resample = nn.Upsample(scale_factor=2)
        else:
            self.resample = Pool(2)

        self.conv1 = QuadConvLayer(spatial_dim,
                                    num_points_in = num_points_in,
                                    num_points_out = num_points_in,
                                    in_channels = in_channels,
                                    out_channels = in_channels,
                                    filter_seq = self.filter_seq,
                                    use_bias = self.use_bias
                                    )
        self.batchnorm1 = nn.InstanceNorm1d(in_channels)

        self.conv2 = QuadConvLayer(spatial_dim,
                                    num_points_in = num_points_in,
                                    num_points_out = num_points_in,
                                    in_channels = in_channels,
                                    out_channels = out_channels,
                                    filter_seq = self.filter_seq,
                                    use_bias = self.use_bias
                                    )
        self.batchnorm2 = nn.InstanceNorm1d(out_channels)

        return

    '''
    Forward mode
    '''
    def forward_op(self, data):
        x = data

        x1 = self.conv1(x)
        x1 = self.activation1(self.batchnorm1(x1))

        x2 = self.conv2(x1)
        x2 = self.activation2(self.batchnorm2(x2) + x)

        return self.resample(x2)

    '''
    Adjoint mode
    '''
    def adjoint_op(self, data):
        x = self.resample(data)

        x1 = self.conv1(x)
        x1 = self.activation1(self.batchnorm1(x1))

        x2 = self.conv2(x1)
        x2 = self.activation2(self.batchnorm2(x2) + x)

        return x2

    '''
    Apply operator
    '''
    def forward(self, data):
        if self.adjoint:
            output = self.adjoint_op(data)
        else:
            output = self.forward_op(data)

        return output
