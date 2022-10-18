'''
Learned quadrature convolutions.
'''

import torch
import torch.nn as nn
import torch_scatter

from core.utilities import Sin, newton_cotes_quad

'''
Quadrature convolution operator.

Input:
    spatial_dim: spatial dimension of input data
    num_points_in: number of input points
    num_points_out: number of output points
    in_channels: input feature channels
    out_channels: output feature channels
    filter_seq: number of features at each filter stage
    output_map: maps number of output points to their location
    filter_mode: type of point to filter operation
    use_bias: whether or not to use bias
'''
class QuadConvLayer(nn.Module):

    def __init__(self,*,
            spatial_dim,
            num_points_in,
            num_points_out,
            in_channels,
            out_channels,
            filter_seq,
            output_map = None,
            filter_mode = 'single',
            use_bias = False
        ):
        super().__init__()

        #validate spatial dim
        assert spatial_dim > 0

        #set attributes
        self.spatial_dim = spatial_dim
        self.num_points_in = num_points_in
        self.num_points_out = num_points_out
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.use_bias = use_bias

        if output_map == None:
            self.output_map = lambda x: newton_cotes_quad(spatial_dim, x)[0]
        else:
            self.output_map = output_map

        #decay parameter
        self.decay_param = (self.num_points_in/16)**2

        #bias
        if self.use_bias:
            bias = torch.empty(1, self.out_channels, self.num_points_out)
            self.bias = nn.Parameter(nn.init.xavier_uniform_(bias, gain=2), requires_grad=True)

        #initialize filter
        self.init_filter(filter_seq, filter_mode)

    '''
    Initialize the layer filter.

    Input:
        filter_seq: mlp feature sequence
        filter_mode: type of filter operation
    '''
    def init_filter(self, filter_seq, filter_mode):
        #NOTE: If we were actually gonne use this one, this would be a bad way to
        #use it as it has a bunch of redundant memory usage.
        if filter_mode == 'static':
            self.G = lambda z: nn.Parameter(torch.randn(self.out_channels, self.in_channels), requires_grad=True).expand(self.idx.shape[0], -1, -1)

        elif filter_mode == 'single':

            mlp_spec = (self.spatial_dim, *filter_seq, self.in_channels*self.out_channels)

            self.G = self.create_mlp(mlp_spec)

            self.G.append(nn.Unflatten(1, (self.in_channels, self.out_channels)))

        elif filter_mode == 'share_in':
            self.filter = nn.ModuleList()

            mlp_spec = (self.spatial_dim, *filter_seq, self.in_channels)

            for j in range(self.out_channels):
                self.filter.append(self.create_mlp(mlp_spec))

            self.G = lambda z: torch.cat(module(z) for module in self.filter).reshape(-1, self.channels_in, self.channels_out)

        elif filter_mode == 'nested':
            self.filter = nn.ModuleList()

            mlp_spec = (self.spatial_dim, *filter_seq, 1)

            for i in range(self.in_channels):
                for j in range(self.out_channels):
                    self.filter.append(self.create_mlp(mlp_spec))

            self.G = lambda z: torch.cat(module(z) for module in self.filter).reshape(-1, self.channels_in, self.channels_out)

        else:
            raise ValueError(f'core::modules::quadconv: Filter mode {filter_mode} is not supported.')

        return

    '''
    Build an mlp.

    Input:
        mlp_channels: sequence of channels
    '''
    def create_mlp(self, mlp_channels):
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
    Compute indices associated with non-zero filters.

    Input:
        nodes: quadrature nodes
        weight_map: maps nodes to quadrature weights
    '''
    def cache(self, nodes, weight_map):
        #output locations
        if self.num_points_in != self.num_points_out:
            output_locs = self.output_map(self.num_points_out)
        else:
            output_locs = nodes

        #determine indices
        locs = (output_locs.repeat_interleave(nodes.shape[0], dim=0) - nodes.repeat(output_locs.shape[0], 1)).view(output_locs.shape[0], nodes.shape[0], self.spatial_dim)

        bump_arg = torch.linalg.vector_norm(locs, dim=(2), keepdims = True)**4

        tf_vec = (bump_arg <= 1/self.decay_param).squeeze()
        idx = torch.nonzero(tf_vec, as_tuple=False)

        self.eval_locs = nn.Parameter(locs[tf_vec, :], requires_grad=False)
        self.eval_indices = nn.Parameter(idx, requires_grad=False)

        #learn the weights
        if weight_map == None:
            weights = torch.zeros(nodes.shape[0])
            self.quad_weights = nn.Parameter(weights, requires_grad=True)

        #weights are specified
        else:
            weights = weight_map(nodes)

            self.quad_weights = nn.Parameter(weights, requires_grad=False)

        return output_locs

    '''
    Apply operator via quadrature approximation of convolution with features and learned filter.

    Input:
        features:  a tensor of shape (batch size  X num of points X input channels)

    Output: tensor of shape (batch size X num of output points X output channels)

    NOTE: THIS WOULD ALL BE A LOT EASIER IF WE WERE DOING CHANNELS_LAST BUT PYTORCH DOESN'T LIKE THAT
    '''
    def forward(self, features):
        integral = torch.zeros(features.shape[0], self.out_channels, self.num_points_out, device=features.device)

        #compute filter
        filters = self.G(self.eval_locs)

        #compute quadrature as weights*filters*features
        values = torch.einsum('n, bni, nij -> bnj',
                                self.quad_weights[self.eval_indices[:,1]],
                                features[:,:,self.eval_indices[:,1]].reshape(features.shape[0], -1, self.in_channels),
                                filters)
        values = values.reshape(features.shape[0], self.out_channels, -1)

        #both of the following are valid
        torch_scatter.segment_coo(values, self.eval_indices[:,0].expand(features.shape[0], self.out_channels, -1), integral, reduce="sum")

        #add bias
        if self.use_bias:
            integral += self.bias

        return integral
