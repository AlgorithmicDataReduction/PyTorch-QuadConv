'''
Quadrature based convolutions.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.integrate import newton_cotes

from core.FastGL.glpair import glpair

'''
Quadrature based convolution operator.

Input:
    point_dim: space dimension
    channels_in: input feature channels
    channels_out: output feature channels
    mlp_channels: convolution kernel MLP feature sequence
    use_bias: add bias term to output of layer
'''
class QuadConvLayer(nn.Module):
    def __init__(self,
                    point_dim,
                    channels_in,
                    channels_out,
                    mlp_channels,
                    use_bias = False
                    ):
        super().__init__()

        #can only handle 2D right now
        if point_dim > 2:
            raise RuntimeError('Point dimension must be less than 2')

        #set hyperparameters
        self.point_dim = point_dim
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.use_bias = use_bias
        self.quad_set_flag = False

        #setup mlps
        self.weight_func = nn.ModuleList()

        mlp_spec = (point_dim, *mlp_channels, 1)

        for i in range(channels_in):
            for j in range(channels_out):
                self.weight_func.append(self.create_mlp(mlp_spec))

    '''
    Create convolution kernel MLP

    Input:
        mlp_channels: MLP feature sequence
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
    Get Gaussian quadrature weights and nodes.

    Input:
        N: number of points
    '''
    def gauss_quad(self, N):
        quad_weights = torch.zeros(N)
        quad_nodes = torch.zeros(N)

        for i in range(N):
            _, quad_weights[i], quad_nodes[i] = glpair(N, i+1)

        return quad_weights, quad_nodes

    '''
    Set quadrature weights and nodes

    The decay parameter here is used to ensure that the MLP decays to 0 as the norm of the input goes to \infty.
    This parameter should likely be set / controlled elsewhere in the code.

    Input:
        N: number of points
    '''
    def set_quad(self, N):

        if N>1e3: raise RuntimeError('Number of quadrature points exceeds the user set limit.')

        self.N = N
        self.decay_param = (N/4)**4

        quad_weights, quad_nodes = self.gauss_quad(N)

        self.quad_weights = nn.Parameter(quad_weights, requires_grad=False)
        self.quad_nodes = nn.Parameter(quad_nodes, requires_grad=False)

        self.quad_set_flag = True

        return

    '''
    Set output locations.

    Input:
        locs: number of output points
    '''
    def set_output_locs(self, locs):
        if isinstance(locs, int):
            _, mesh_nodes = self.gauss_quad(locs)

        else:
            mesh_nodes = locs

        node_list = [mesh_nodes]*self.point_dim

        output_locs = torch.dstack(torch.meshgrid(*node_list, indexing='xy')).reshape(-1, self.point_dim)

        self.output_locs = nn.Parameter(output_locs, requires_grad=False)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(1, self.channels_out, output_locs.shape[0]))

        return

    '''
    Get quadrature nodes and weights
    '''
    def get_quad_mesh(self):

        if not self.quad_set_flag: raise RuntimeError('Mesh not yet set')

        node_list = [self.quad_nodes]*self.point_dim

        mesh_nodes = torch.meshgrid(*node_list, indexing='xy')

        mesh_nodes = torch.dstack(mesh_nodes).reshape(-1, self.point_dim)

        weight_list = [self.quad_weights]*self.point_dim

        mesh_weights =  torch.meshgrid(*weight_list, indexing='xy')

        mesh_weights = torch.dstack(mesh_weights).reshape(-1, self.point_dim)

        return mesh_nodes, mesh_weights

    '''
    Evaluate the convolution kernel MLPs

    The essential component of this function is that it returns an array of shape:
            (self.channels_out, self.channels_in, -1)

    Otherwise any method for generating these weights is acceptable (consider test functions etc)

    Input:
        x: the location in \mathbb{R}^dim that you are evaluating each MLP at
    '''
    def eval_MLPs(self, x):
        weights = [module(x) for module in self.weight_func]
        weights = torch.cat(weights).view(self.channels_out, self.channels_in, -1)

        return weights

    '''
    Evaluate the convolution kernel.

    The mesh_weights array is passed into this function for the sake of computational efficiency, but the main purpose
    of this function is to evaluate the convolution kernel which includes the MLPs and the decay function (AKA the Bump function)

    This function uses the sparse_coo_tensor format since the bump function enforces the compact support of the MLP. Further computation
    does not take advantage of this sparsity but should do so in the future.

    The compact support is a function of the self.decay_param

    Input:
        x: the location in \mathbb{R}^dim that you are evaluating each MLP at
        mesh_weights: this function is derived from the self.get_quad_mesh() call and corresponds to the quadrature weights

    '''
    def kernel_func(self, x, mesh_weights):
        # bump_arg = torch.linalg.vector_norm(x, dim=(2), keepdims = True)**4
        #
        # tf_vec = (bump_arg <= 1/self.decay_param).squeeze()
        #
        # x_eval = x[tf_vec,:]
        #
        # weights_sparse = self.eval_MLPs(x_eval)
        #
        # idx = torch.nonzero(tf_vec, as_tuple=False).transpose(0,1)
        #
        # mesh_weights_sparse = mesh_weights.repeat(x.shape[0], 1).reshape(x.shape[0], x.shape[1])[idx[0,:], idx[1,:]].view(1, 1, -1)
        #
        # bump = (np.e*torch.exp(-1/(1-self.decay_param*bump_arg[tf_vec]))).view(1, 1, -1)
        #
        # temp = (weights_sparse*bump*mesh_weights_sparse).view(-1, self.channels_out, self.channels_in)
        #
        # weights = torch.sparse_coo_tensor(idx, temp, [x.shape[0], x.shape[1], self.channels_out, self.channels_in]).coalesce() #might not need this if we figure out sparsity
        #
        # return weights

        weights = self.eval_MLPs(x)
        bump_arg = torch.linalg.vector_norm(x, dim=(2), keepdims = True)**4
        bump = (np.e*torch.exp(-1/(1-self.decay_param*bump_arg))).view(1, 1, -1)
        mesh_weights = mesh_weights.repeat(x.shape[0], 1).reshape(x.shape[0], x.shape[1]).view(1, 1, -1)

        return (weights*bump*mesh_weights)

    '''
    Compute 1D quadrature

    Input:
        features:
        output_locs:
        nodes:
        mesh_weights:
    '''
    def quad(self, features, output_locs, nodes, mesh_weights):
        # eval_locs = (torch.repeat_interleave(output_locs, nodes.shape[0], dim=0)-nodes.repeat(output_locs.shape[0], 1)).view(output_locs.shape[0], nodes.shape[0], self.point_dim)
        #
        # kf = self.kernel_func(eval_locs, mesh_weights)
        #
        # s = eval_locs.shape
        #
        # del(eval_locs)
        #
        # idx = kf.indices()
        #
        # batch_size = features.shape[0]
        #
        # ol =  output_locs.shape[0]
        # il =  features.shape[2]
        #
        # kf_dense = torch.zeros(1, self.channels_out, self.channels_in, ol, il, device=features.device)
        #
        # kf_dense[:,:,:,idx[0,:],idx[1,:]] = (kf.values()).view(1, self.channels_out, self.channels_in, -1)
        #
        # integral = torch.einsum('b...dij, b...dj -> b...i', kf_dense, features.view(batch_size, 1, self.channels_in, il))
        #
        # return integral

        eval_locs = (torch.repeat_interleave(output_locs, nodes.shape[0], dim=0)-nodes.repeat(output_locs.shape[0], 1)).view(output_locs.shape[0], nodes.shape[0], self.point_dim)

        kf = self.kernel_func(eval_locs, mesh_weights)

        s = eval_locs.shape

        del(eval_locs)

        batch_size = features.shape[0]
        ol =  output_locs.shape[0]
        il =  features.shape[2]

        return torch.einsum('b...dij, b...dj -> b...i', kf.view(1, self.channels_out, self.channels_in, ol, il), features.view(batch_size, 1, self.channels_in, il))

    '''
    Compute enitre domain integral

    Input:
        features:
        output_locs:
    '''
    def tensor_prod_quad(self, features, output_locs):
        nodes, weights = self.get_quad_mesh()

        integral = self.quad(features, output_locs, nodes, weights)

        return integral

    '''
    Compute entire domain integral via recursive quadrature

    The idea here is that each of the 1D integrals is computed for all output locations and
    summed into the 2D integral or 3D integral.

    Input:
        features:
        output_locs:
        level:
        nodes:
        weights:
    '''
    def rquad(self, features, output_locs, level=(), nodes=(), weights=()):
        if level == self.point_dim or not level:
            nodes =  self.quad_nodes.unsqueeze(-1)
            weights = self.quad_weights.unsqueeze(-1)
            level = self.point_dim

        if level == 1:
            integral = self.quad(features, output_locs, nodes, weights)

        elif level > 1:
            '''
            Not sure there is a better way to do this.
            '''
            integral = torch.zeros(features.shape[0], self.channels_out, output_locs.shape[0], device=features.device)

            for i in range(self.N):
                this_coord = self.quad_nodes[i].expand(self.N, 1)
                this_weight = self.quad_weights[i]

                integral += self.rquad(features[:,:,(i*self.N):(i+1)*self.N], output_locs, level=level-1,
                                        nodes=torch.hstack([nodes,this_coord]), weights=this_weight*weights)

        return integral

    '''
    Apply operator

    Input:
        features:
        output_locs:
    '''
    def forward(self, features, output_locs=None):
        if output_locs is None:
            output_locs = self.output_locs

        if output_locs.shape[1] != self.point_dim:
            raise RuntimeError('dimension mismatch')

        integral =  self.rquad(features, output_locs)
        # integral = self.tensor_prod_quad(features, output_locs)

        if self.use_bias:
            integral += self.bias

        return integral

'''
Module wrapper around sin function; allows it to operate as a layer.
'''
class Sin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

################################################################################

'''
QuadConvLayer block

Input:
    point_dim: space dimension
    channels_in: input feature channels
    channels_out: output feature channels
    N_in: number of input points
    N_out: number of output points
    mlp_channels: convolution kernel MLP feature sequence
    adjoint: downsample or upsample
    quad_type: quadrature type
    mlp_mode: ?
    use_bias: add bias term to output of layer
    activation1:
    activation2:
'''
class QuadConvBlock(nn.Module):
    def __init__(self,
                    point_dim,
                    channels_in,
                    channels_out,
                    N_in,
                    N_out,
                    mlp_channels,
                    adjoint = False,
                    use_bias = False,
                    activation1 = nn.CELU(alpha=1),
                    activation2 = nn.CELU(alpha=1)
                    ):
        super().__init__()

        self.adjoint = adjoint
        self.activation1 = activation1
        self.activation2 = activation2

        if self.adjoint:
            conv1_point_num = N_out
            conv1_channel_num = channels_out
        else:
            conv1_point_num = N_in
            conv1_channel_num = channels_in

        self.conv1 = QuadConvLayer(point_dim,
                                    channels_in = conv1_channel_num,
                                    channels_out = conv1_channel_num,
                                    mlp_channels = mlp_channels,
                                    use_bias = use_bias
                                    )
        self.batchnorm1 = nn.InstanceNorm1d(conv1_channel_num)

        self.conv2 = QuadConvLayer(point_dim,
                                    channels_in = channels_in,
                                    channels_out = channels_out,
                                    mlp_channels = mlp_channels,
                                    use_bias = use_bias
                                    )
        self.batchnorm2 = nn.InstanceNorm1d(channels_out)


        if N_in and N_out:
            self.conv1.set_quad(conv1_point_num)
            self.conv2.set_quad(N_in)

            self.conv1.set_output_locs(conv1_point_num)
            self.conv2.set_output_locs(N_out)

    '''
    Forward mode
    '''
    def forward_op(self, data):
        x = data

        x1 = self.conv1(x)
        x1 = self.activation1(self.batchnorm1(x1))
        x1 = x1 + x

        x2 = self.conv2(x1)
        x2 = self.activation2(self.batchnorm2(x2))

        return x2

    '''
    Adjoint mode
    '''
    def adjoint_op(self, data):
        x = data

        x2 = self.conv2(x)
        x2 = self.activation2(self.batchnorm2(x2))

        x1 = self.conv1(x2)
        x1 = self.activation1(self.batchnorm1(x1))
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
