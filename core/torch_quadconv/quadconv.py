'''
'''

import torch
import torch.nn as nn
import torch_scatter

from .utils.misc import Sin

'''
Quadrature convolution operator.

Input:
    spatial_dim: spatial dimension of input data
    in_points: number of input points
    out_points: number of output points
    in_channels: input feature channels
    out_channels: output feature channels
    filter_seq: number of features at each filter stage
    filter_mode: type of point to filter operation
    bias: whether or not to use bias
    output_same: whether or not to use the input points as the output points
    cache: whether or not to cache the evaluation indices
'''
class QuadConv(nn.Module):

    def __init__(self,*,
            spatial_dim,
            in_points,
            out_points,
            in_channels,
            out_channels,
            filter_seq,
            filter_mode = 'single',
            weight_activation = nn.Sigmoid,
            bias = False,
            output_same = False,
            cache = True
        ):
        super().__init__()

        #validate spatial dim
        assert spatial_dim > 0

        #set attributes
        self.spatial_dim = spatial_dim
        self.in_points = in_points
        self.out_points = out_points
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.output_same = output_same

        self.cache = cache
        self.cached = False

        #decay parameter
        self.decay_param = (self.in_points/16)**2

        #initialize filter
        self._init_filter(filter_seq, filter_mode)

        #quadrature weights activation
        self.weight_activation = weight_activation()

        #bias
        if bias:
            bias = torch.empty(1, self.out_channels, self.out_points)
            self.bias = nn.Parameter(nn.init.xavier_uniform_(bias, gain=2), requires_grad=True)
        else:
            self.bias = None

        return

    '''
    Initialize the layer filter.

    Input:
        filter_seq: mlp feature sequence
        filter_mode: type of filter operation
    '''
    def _init_filter(self, filter_seq, filter_mode):

        #single mlp
        if filter_mode == 'single':

            mlp_spec = (self.spatial_dim, *filter_seq, self.in_channels*self.out_channels)

            self.H = self._create_mlp(mlp_spec)
            self.H.append(nn.Unflatten(1, (self.in_channels, self.out_channels)))

        #mlp for each output channel
        elif filter_mode == 'share_in':

            mlp_spec = (self.spatial_dim, *filter_seq, self.in_channels)

            self.filter = nn.ModuleList()
            for j in range(self.out_channels):
                self.filter.append(self._create_mlp(mlp_spec))

            self.H = lambda z: torch.cat([module(z) for module in self.filter]).reshape(-1, self.channels_in, self.channels_out)

        #mlp for each input and output channel pair
        elif filter_mode == 'nested':

            mlp_spec = (self.spatial_dim, *filter_seq, 1)

            self.filter = nn.ModuleList()
            for i in range(self.in_channels):
                for j in range(self.out_channels):
                    self.filter.append(self._create_mlp(mlp_spec))

            self.H = lambda z: torch.cat([module(z) for module in self.filter]).reshape(-1, self.in_channels, self.out_channels)

        else:
            raise ValueError(f'core::modules::quadconv: Filter mode {filter_mode} is not supported.')

        #multiply by bump function
        self.G = lambda z: self._bump(z)*self.H(z)

        return

    '''
    Build an mlp.

    Input:
        mlp_channels: sequence of channels
    '''
    def _create_mlp(self, mlp_channels):

        #linear layer settings
        activation = Sin()
        bias = False

        #build mlp
        mlp = nn.Sequential()

        for i in range(len(mlp_channels)-2):
            mlp.append(nn.Linear(mlp_channels[i], mlp_channels[i+1], bias=bias))
            mlp.append(activation)

        mlp.append(nn.Linear(mlp_channels[-2], mlp_channels[-1], bias=bias))

        return mlp

    '''
    Calculate bump vector norm.

    Input:
        z: evaluation locations, [out_points, in_points, spatial_dim]
    '''
    def _bump_arg(self, z):
        return torch.linalg.vector_norm(z, dim=(2), keepdims = True)**4

    '''
    Calculate bump function.

    Input:
        z: evaluation locations, [num_points, spatial_dim]
    '''
    def _bump(self, z):

        bump_arg = torch.linalg.vector_norm(z, dim=(1), keepdims = False)**4
        bump = torch.exp(1-1/(1-self.decay_param*bump_arg))

        return bump.reshape(-1, 1, 1)

    '''
    Compute indices associated with non-zero filters.

    Input:
        handler: MeshHandler or PointCloudHandler object
    '''
    def _compute_eval_indices(self, handler):

        #get output points
        input_points = handler.input_points
        output_points = input_points if self.output_same else handler.output_points

        #check
        assert input_points.shape[0] == self.in_points, f'{input_points.shape[0]} != {self.in_points}'
        assert output_points.shape[0] == self.out_points, f'{output_points.shape[0]} != {self.out_points}'

        #determine indices
        locs = output_points.unsqueeze(1) - input_points.unsqueeze(0)
        #locs = (output_points.repeat_interleave(input_points.shape[0], dim=0) - input_points.repeat(output_points.shape[0], 1)).reshape(output_points.shape[0], input_points.shape[0], self.spatial_dim)

        bump_arg = self._bump_arg(locs)

        tf_vec = (bump_arg <= 1/self.decay_param).squeeze()
        idx = torch.nonzero(tf_vec, as_tuple=False)

        if self.cache:
            self.eval_indices = nn.Parameter(idx, requires_grad=False)
            self.cached = True

        return idx

    '''
    Apply operator via quadrature approximation of convolution with features and learned filter.

    Input:
        handler: MeshHandler or PointCloudHandler object
        features: a tensor of shape (batch size  X num of points X input channels)

    Output: tensor of shape (batch size X num of output points X output channels)
    '''
    def forward(self, handler, features):

        #get evaluation indices
        if self.cached:
            eval_indices = self.eval_indices
        else:
            eval_indices = self._compute_eval_indices(handler)

        #get weights
        weights = handler.weights[eval_indices[:,1]]

        #compute eval locs
        if self.output_same:
            eval_locs = handler.input_points[eval_indices[:,0]] - handler.input_points[eval_indices[:,1]]
        else:
            eval_locs = handler.output_points[eval_indices[:,0]] - handler.input_points[eval_indices[:,1]]
            handler.step()

        #compute filter
        filters = self.G(eval_locs)

        #compute quadrature as weights*filters*features
        values = torch.einsum('n, nij, bin -> bjn',
                                weights,
                                filters,
                                features[:,:,eval_indices[:,1]])

        #setup integral
        integral = values.new_zeros(features.shape[0], self.out_channels, self.out_points)

        #scatter
        torch_scatter.segment_coo(values, eval_indices[:,0].expand(features.shape[0], self.out_channels, -1), integral, reduce="sum")

        #add bias
        if self.bias is not None:
            integral += self.bias

        return integral
