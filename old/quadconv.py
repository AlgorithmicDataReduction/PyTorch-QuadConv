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
    kernel_mode: convolution kernel type
    quad_type: quadrature type
    mlp_mode: ?
    use_bias: add bias term to output of layer
'''
class QuadConvLayer(nn.Module):
    def __init__(self,
                    point_dim,
                    channels_in,
                    channels_out,
                    mlp_channels,
                    kernel_mode = 'MLP',
                    quad_type = 'gauss',
                    mlp_mode = 'single',
                    use_bias = False
                    ):
        super().__init__()

        self.decay_param = 100*torch.ones(1)
        self.center = torch.zeros(1, point_dim)
        self.inv_covar_cholesky = torch.diag(torch.ones(point_dim))

        self.use_bias = use_bias
        self.mlp_mode = mlp_mode

        self.quad_type = quad_type
        self.composite_quad_order = 5

        '''
        Kernel modes:

        MLP: This encompasses most use cases for the code, and is meant to be used whenever training a QConv on data.

        MLP::Single : This generates a single MLP for every incoming and outgoing channel.
                      Using for example a 1D Conv operator, this corresponds to:

                        .. math::
                            \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
                            \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{\text{out}_j}, k)
                            \star \text{input}(N_i, k)

                    If :math:'\text{weight}(i,j)' is each a single MLP with a domain in :math:'\mathbb{R}^3' and range :math:'\mathbb{R}'

        MLP::Share-in : Borrowing the example in MLP::Single, this corresponds to:

                        :math:'\text{weight}(i,:)' being a single MLP with domain in  :math:'\mathbb{R}^3' and range :math:'\mathbb{R}^{C_in}'

        MLP::Half-share : This is currently non-functional


        Sinc: This MLP mode was generated specifically for testing the action of the quadrature portion of the code.
                One can use the Sinc mode if attempting to verify a new quadrature method since the Sinc function implements
                the low-pass filter kernel and can then be used with known data to produce the low-passed version of the data
                should the quadrature functionality be an accurate approximation of the integral.

                TODO: This should really be a mode that ingests a kernel function of choice and checks the quadrature function
                that way.

        '''

        if kernel_mode == 'MLP':
            self.weight_func = torch.nn.ModuleList()

            if mlp_mode == 'single':
                mlp_spec = (point_dim, *mlp_channels, 1)

                for i in range(channels_in):
                    for j in range(channels_out):
                        self.weight_func.append(self.create_mlp(mlp_spec))

            elif mlp_mode == 'share_in' or channels_out == 1:
                mlp_spec = (point_dim, *mlp_channels, channels_in)

                for j in range(channels_out):
                    self.weight_func.append(self.create_mlp(mlp_spec))

            elif self.mlp_mode == 'half_share':
                common = 4

                loop = max(int(channels_out),1)

                share = max(int(channels_out/channels_in),1)

                mlp_spec = (point_dim, *mlp_channels, share)

                for j in range(loop):
                    self.weight_func.append(self.create_mlp(mlp_spec))

        elif kernel_mode == 'sinc':
            self.weight_func = [[]] * channels_out
            self.weight_func[0] = lambda x : (25.3*torch.sinc(8*x[:,0])*torch.sinc(4*x[:,1])).unsqueeze(-1)

        self.channels_out = channels_out
        self.channels_in = channels_in
        self.point_dim = point_dim

        self.quad_set_flag = False

        if point_dim > 2:
            raise RuntimeError('Point dimension must be less than 2')

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
    Get Newton-Cotes quadrature weights and nodes.
    This function returns the composite rule, so its required that the order of the quadrature rule divides evenly into N.

    Input:
        N: number of points
        x0: left end point
        x1: right end point
    '''
    def newton_cotes_quad(self, N, x0=0, x1=1):
        rep = int(N/self.composite_quad_order)

        dx = (x1-x0)/(self.composite_quad_order-1)

        weights, _ = newton_cotes(self.composite_quad_order-1, 1)
        weights = torch.tile(dx*weights, rep)

        return weights, torch.linspace(x0, x1, N)

    '''
    Set quadrature weights and nodes

    The decay parameter here is used to ensure that the MLP decays to 0 as the norm of the input goes to \infty.
    This parameter should likely be set / controlled elsewhere in the code.

    Input:
        N: number of points
    '''
    def set_quad(self, N):
        self.N = N

        self.decay_param = (N/4)**4

        if N < 1e3:
            if self.quad_type == 'gauss':
                quad_weights, quad_nodes = self.gauss_quad(N)

            elif self.quad_type == 'newton_cotes':
                quad_weights, quad_nodes = self.newton_cotes_quad(N)

            self.quad_weights = nn.Parameter(quad_weights, requires_grad=False)
            self.quad_nodes = nn.Parameter(quad_nodes, requires_grad=False)

        else:
            raise RuntimeError('Number of quadrature points exceeds the user set limit.')

        self.quad_set_flag = True

    '''
    Set output locations.

    Input:
        locs: number of output points
    '''
    def set_output_locs(self, locs):
        if isinstance(locs, int):
            if self.quad_type == 'gauss':
                _, mesh_nodes = self.gauss_quad(locs)

            elif self.quad_type == 'newton_cotes':
                _, mesh_nodes = self.newton_cotes_quad(locs)

        else:
            mesh_nodes = locs

        node_list = [mesh_nodes]*self.point_dim

        output_locs = torch.dstack(torch.meshgrid(*node_list, indexing='xy')).reshape(-1, self.point_dim)

        self.output_locs = nn.Parameter(output_locs, requires_grad=False)

        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(1, self.channels_out, output_locs.shape[0]))

    '''
    Get output locations

    This function should be much more flexible in the future since changing the quadrature evaluation scheme may be favorable

    Currently this always returns a mesh that corresponds to the (number of points in the 1D scheme) ** (the dimension of the problem)

    i.e. this is always a tensor product schema

    NOTE: Deprecated for now, instead set_output_locs carries out this functionality as well
    '''
    # def get_output_locs(self):
    #     node_list = [self.output_locs]*self.point_dim
    #
    #     mesh_nodes = torch.meshgrid(*node_list, indexing='xy')
    #
    #     mesh_nodes = torch.dstack(mesh_nodes).reshape(-1, self.point_dim)
    #
    #     return mesh_nodes

    '''
    Get quadrature nodes and weights
    '''
    def get_quad_mesh(self):
        if self.quad_set_flag:
            node_list = [self.quad_nodes]*self.point_dim

            mesh_nodes = torch.meshgrid(*node_list, indexing='xy')

            mesh_nodes = torch.dstack(mesh_nodes).reshape(-1, self.point_dim)

            weight_list = [self.quad_weights]*self.point_dim

            mesh_weights =  torch.meshgrid(*weight_list, indexing='xy')

            mesh_weights = torch.dstack(mesh_weights).reshape(-1, self.point_dim)

        else:
            raise RuntimeError('Mesh not yet set')

        return mesh_nodes, mesh_weights

    '''
    Evaluate the convolution kernel MLPs

    See the init function more a more detailed description of the MLP modes.

    The essential component of this function is that it returns an array of shape:
            (self.channels_out, self.channels_in, -1)

    Otherwise any method for generating these weights is acceptable (consider test functions etc)

    Input:
        x: the location in \mathbb{R}^dim that you are evaluating each MLP at
    '''
    def eval_MLPs(self, x):
        if self.mlp_mode == 'single':
            weights = [module(x) for module in self.weight_func]
            weights = torch.cat(weights).reshape(self.channels_out, self.channels_in, -1)

        elif self.mlp_mode == 'share_in' or self.channels_out == 1:
            weights = [module(x) for module in self.weight_func]
            weights = torch.cat(weights).reshape(self.channels_out, self.channels_in, -1)

        elif self.mlp_mode == 'half_share':
            weights = torch.zeros(self.channels_out*self.channels_in, x.shape[0])
            share = max(int(self.channels_out/self.channels_in), 1)

            for j in range(self.channels_out):
                weights[j,j*share:((j+1)*share-1),:] = self.weight_func[j](x).transpose(0,1)

        return weights.reshape(self.channels_out, self.channels_in, -1)

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
        bump_arg = torch.linalg.vector_norm(x, dim=(2), keepdims = True)**4

        tf_vec = (bump_arg <= 1/self.decay_param).squeeze()

        x_eval = x[tf_vec,:]

        weights_sparse = self.eval_MLPs(x_eval)

        idx = torch.nonzero(tf_vec, as_tuple=False).transpose(0,1)

        mesh_weights_sparse = mesh_weights.repeat(x.shape[0], 1).reshape(x.shape[0], x.shape[1])[idx[0,:], idx[1,:]].reshape(1, 1, -1)

        bump = (np.e*torch.exp(-1/(1-self.decay_param*bump_arg[tf_vec]))).reshape(1, 1, -1)

        temp = (weights_sparse*bump*mesh_weights_sparse).reshape(-1, self.channels_out, self.channels_in)

        weights = torch.sparse_coo_tensor(idx, temp, [x.shape[0], x.shape[1], self.channels_out, self.channels_in]).coalesce()

        return weights

    '''
    Compute 1D quadrature

    Input:
        features:
        output_locs:
        nodes:
        mesh_weights:
    '''
    def quad(self, features, output_locs, nodes, mesh_weights):
        eval_locs = (torch.repeat_interleave(output_locs, nodes.shape[0], dim=0)-nodes.repeat(output_locs.shape[0], 1)).view(output_locs.shape[0], nodes.shape[0], self.point_dim)

        kf = self.kernel_func(eval_locs, mesh_weights)

        s = eval_locs.shape

        del(eval_locs)

        idx = kf.indices()

        batch_size = features.shape[0]

        ol =  output_locs.shape[0]
        il =  features.shape[2]

        kf_dense = torch.zeros(1, self.channels_out, self.channels_in, ol, il, device=features.device)

        kf_dense[:,:,:,idx[0,:],idx[1,:]] = (kf.values()).view(1, self.channels_out, self.channels_in, -1)

        integral = torch.einsum('b...dij, b...dj -> b...i', kf_dense, features.view(batch_size, 1, self.channels_in, il))

        return integral

    '''
    Compute enitre domain integral

    Input:
        features:
        output_locs:
    '''
    def tensor_prod_quad(self, features, output_locs):
        mesh_nodes, mesh_weights = self.get_quad_mesh()

        integral = self.quad(features, output_locs, mesh_nodes, mesh_weights)

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
            device = features.device
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

    TODO: This is a specific and possibly bad way of initializing the self.bias parameter

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
                    quad_type = 'gauss',
                    mlp_mode = 'single',
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
                                    kernel_mode = 'MLP',
                                    quad_type = quad_type,
                                    mlp_mode = mlp_mode,
                                    use_bias = use_bias
                                    )
        self.batchnorm1 = torch.nn.InstanceNorm1d(conv1_channel_num)

        self.conv2 = QuadConvLayer(point_dim,
                                    channels_in = channels_in,
                                    channels_out = channels_out,
                                    mlp_channels = mlp_channels,
                                    kernel_mode = 'MLP',
                                    quad_type = quad_type,
                                    mlp_mode = mlp_mode,
                                    use_bias = use_bias,
                                    )
        self.batchnorm2 = torch.nn.InstanceNorm1d(channels_out)


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
