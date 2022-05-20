'''
Quadrature based convolution
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.integrate import newton_cotes

'''
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

        self.bias = []
        self.use_bias = use_bias
        self.mlp_mode = mlp_mode

        self.quad_type = quad_type
        self.composite_quad_order = 5

        self.output_locs = []

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
                    self.weight_func.append( self.create_mlp(mlp_spec) )

        elif kernel_mode == 'sinc':
            self.weight_func = [[]] * channels_out
            self.weight_func[0] = lambda x : (25.3*torch.sinc(8*x[:,0])*torch.sinc(4*x[:,1])).unsqueeze(-1)

        self.quad_weights = []
        self.quad_nodes = []
        self.N = []
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.point_dim = point_dim

        self.quad_set_flag = False

        if point_dim > 2:
            raise RuntimeError('Point dimension must be less than 2')

    '''
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
    '''
    def kernel_func(self, x, mesh_weights):
        bump_arg = torch.linalg.vector_norm(x, dim=(2), keepdims = True)**4

        tf_vec = (bump_arg <= 1/self.decay_param).squeeze()

        x_eval = x[tf_vec,:]

        weights_sparse = self.eval_MLPs(x_eval)

        idx = torch.nonzero(tf_vec, as_tuple=False).transpose(0,1)

        mesh_weights_sparse = mesh_weights.repeat(x.shape[0], 1).reshape(x.shape[0], x.shape[1])[idx[0,:], idx[1,:]].reshape(1, 1, -1)

        bump = (torch.exp(1)*torch.exp(-1/(1-self.decay_param*bump_arg[tf_vec]))).reshape(1, 1, -1)

        temp = (weights_sparse*bump*mesh_weights_sparse).reshape(-1, self.channels_out, self.channels_in)

        weights = torch.sparse_coo_tensor(idx, temp, [x.shape[0], x.shape[1], self.channels_out, self.channels_in]).coalesce()

        return weights

    '''
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

        mlp.append(nn.Linear(mlp_channels[-2], mlp_channels[-1], bias=False))

        return mlp

    '''
    '''
    def set_quad(self, N):
        self.N = N

        self.decay_param = (N/4)**4

        if N < 1e3:
            if self.quad_type == 'gauss':
                self.quad_weights, self.quad_nodes = self.gauss_quad(N)

            elif self.quad_type == 'newton_cotes':
                self.quad_weights, self.quad_nodes = self.newton_cotes_quad(N)

        else:
            raise RuntimeError('Number of quadrature points exceeds the user set limit')

        self.quad_set_flag = True

    '''
    '''
    def gauss_quad(self, N):
        quad_weights = torch.zeros(N)
        quad_nodes = torch.zeros(N)

        for i in range(N):
            _, quad_weights[i], quad_nodes[i] = glpair(N,i+1)

        return quad_weights, quad_nodes

    '''
    '''
    def newton_cotes_quad(self, N, x0=0, x1=1):
        rep = int(N/self.composite_quad_order)

        dx = (x1-x0)/(self.composite_quad_order-1)

        weights, _ = newton_cotes(self.composite_quad_order-1, 1)
        weights = torch.tile(dx*weights, rep)

        return weights, torch.linspace(x0, x1, N)

    '''
    '''
    def set_output_locs(self, locs):
        if isinstance(locs, int):
            print(self.quad_type)
            if self.quad_type == 'gauss':
                _, self.output_locs = self.gauss_quad(locs)

            elif self.quad_type == 'newton_cotes':
                _, self.output_locs = self.newton_cotes_quad(locs)

        else:
            self.output_locs = locs

    '''
    '''
    def get_output_locs(self):
        node_list = [self.output_locs]*self.point_dim

        mesh_nodes = torch.meshgrid(*node_list, indexing='xy')

        mesh_nodes = torch.dstack(mesh_nodes).reshape(-1, self.point_dim)

        return mesh_nodes

    '''
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
    '''
    def quad(self, features, output_locs, nodes, mesh_weights):
        eval_locs = (torch.repeat_interleave(output_locs, nodes.shape[0], dim=0)-nodes.repeat(output_locs.shape[0], 1)).reshape(output_locs.shape[0], nodes.shape[0], self.point_dim)

        kf = self.kernel_func(eval_locs, mesh_weights)

        s = eval_locs.shape

        del(eval_locs)

        idx = kf.indices()

        batch_size = features.shape[0]

        ol =  output_locs.shape[0]
        il =  features.shape[2]

        kf_dense = torch.zeros(1, self.channels_out, self.channels_in, ol, il)

        kf_dense[:,:,:,idx[0,:], idx[1,:]] = (kf.values()).reshape(1, self.channels_out, self.channels_in, -1)

        integral = torch.einsum('b...dij, b...dj  -> b...i', kf_dense, features.reshape(batch_size, 1, self.channels_in, il))

        return integral

    '''
    '''
    def tensor_prod_quad(self, features, output_locs):
        mesh_nodes, mesh_weights = self.get_quad_mesh()

        integral = self.quad(features, output_locs, mesh_nodes, mesh_weights)

        return integral

    '''
    '''
    def rquad(self, features, output_locs, level=(), nodes=() , weights=()):
        if level == self.point_dim or not level:
            nodes =  self.quad_nodes.unsqueeze(-1)
            weights = self.quad_weights.unsqueeze(-1)
            level = self.point_dim

        if level == 1:
            integral = self.quad(features, output_locs, nodes, weights)

        elif level > 1:
            integral = torch.zeros(features.shape[0], self.channels_out, output_locs.shape[0])

            for i in range(self.N):
                this_coord = self.quad_nodes[i].expand(self.N, 1)
                this_weight = self.quad_weights[i]

                integral += self.rquad(features[:,:,(i*self.N):(i+1)*self.N], output_locs, level=level-1 ,
                                        nodes=torch.hstack([nodes,this_coord]), weights=this_weight*weights)

        return integral

    '''
    '''
    def forward(self, features, output_locs=False):
        if isinstance(output_locs, bool) and not output_locs:
            output_locs = self.get_output_locs()

        if output_locs.shape[1] != self.point_dim:
            raise RuntimeError('dimension mismatch')

        integral =  self.rquad(features, output_locs)

        if self.use_bias:
            if not torch.is_tensor(self.bias) and not self.bias:
                self.bias = torch.nn.Parameter(torch.zeros(1, integral.shape[1], integral.shape[2]))

            integral += self.bias

        return integral

'''
'''
class Sin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return torch.sin(x)

################################################################################

'''
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
    '''
    def forward(self, data):
        if self.adjoint:
            output = self.adjoint_op(data)
        else:
            output = self.forward_op(data)

        return output

    '''
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
    '''
    def adjoint_op(self,data):
        x = data

        x2 = self.conv2(x)
        x2 = self.activation2(self.batchnorm2(x2))

        x1 = self.conv1(x2)
        x1 = self.activation1(self.batchnorm1(x1))
        x1 = x1 + x2

        return x1
