import sys

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy as np
from scipy.integrate import newton_cotes

from torch.nn.utils.parametrizations import spectral_norm as spn


class QuadConvLayer(nn.Module):

    def __init__(self, dimension=1, channels_in = 1, channels_out = 1, weight_mlp_chan=(4,8,4), kernel_mode = 'MLP', quad_type='gauss', use_bias = False, mlp_mode = 'single'):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.decay_param = 100*torch.ones(1,device = self.device)
        self.center = torch.zeros(1,dimension, device = self.device)
        self.inv_covar_cholesky = torch.diag(torch.ones(dimension,device = self.device))


        self.bias = []
        self.use_bias = use_bias
        self.mlp_mode = mlp_mode

        self.quad_type = quad_type
        self.composite_quad_order = 5

        self.output_locs = []

        if kernel_mode == 'MLP':

            self.weight_func = torch.nn.ModuleList()

            if mlp_mode == 'single':

                mlp_spec = (dimension, *weight_mlp_chan, 1)

                for i in range(channels_in):
                    for j in range(channels_out):
                        self.weight_func.append( self.create_mlp(mlp_spec) )

            elif mlp_mode == 'share_in' or channels_out == 1:

                mlp_spec = (dimension, *weight_mlp_chan, channels_in)

                for j in range(channels_out):
                    self.weight_func.append( self.create_mlp(mlp_spec) )

            elif self.mlp_mode == 'half_share':

                common = 4

                loop = max(int(channels_out),1)

                share = max(int(channels_out/channels_in),1)

                mlp_spec = (dimension, *weight_mlp_chan, share)

                for j in range(loop):
                    self.weight_func.append( self.create_mlp(mlp_spec) )


        elif kernel_mode == 'sinc':

            self.weight_func = [ [] ] * channels_out

            self.weight_func[0] = lambda x : torch.from_numpy(np.sinc(8*x[:,0]) * np.sinc(4*x[:,1]) * (25.3)).unsqueeze(-1)
            #self.weight_func[1] = lambda x : torch.from_numpy(np.sinc(2*x[:,0]) * np.sinc(10*x[:,1]) * (25.3)).unsqueeze(-1)


        self.quad_weights = []
        self.quad_nodes = []
        self.N = []
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.dimension = dimension

        self.quad_set_flag = False

        if dimension > 2:
            raise RuntimeError('Dimension must be less than 2')


    def eval_MLPs(self,x):


        if self.mlp_mode == 'single':

            weights = [module(x) for module in self.weight_func]

            weights = torch.cat(weights).reshape(self.channels_out , self.channels_in, -1)

        elif self.mlp_mode == 'share_in' or self.channels_out == 1:

            weights = [module(x) for module in self.weight_func]

            weights = torch.cat(weights).reshape(self.channels_out , self.channels_in, -1)

            #weights = torch.zeros(self.channels_out,  self.channels_in, x.shape[0]).to(self.device)

            #for j in range(self.channels_out):
            #    weights[j,:,:] =  self.weight_func[j](x).transpose(0,1)


        elif self.mlp_mode == 'half_share':

            weights = torch.zeros(self.channels_out * self.channels_in, x.shape[0]).to(self.device)

            share = max(int(self.channels_out/self.channels_in),1)

            for j in range(self.channels_out):
                weights[j,j*share:((j+1)*share-1),:] =  self.weight_func[j](x).transpose(0,1)


        return weights.reshape(self.channels_out , self.channels_in, -1)


    def kernel_func(self, x, mesh_weights):

        # x should be (number of points) by (dimension of the points)

        #weights = torch.zeros(self.channels_out, self.channels_in, x.shape[0]).to(self.device)

        #weights = self.eval_MLPs(x)

        #inv_covar =  torch.matmul(self.inv_covar_cholesky, self.inv_covar_cholesky.transpose(0,1)) + 0.00001*torch.diag(torch.ones(self.dimension,device = self.device))

        #arg1 = torch.matmul(x-self.center, inv_covar)

        #arg2 = torch.linalg.vector_norm(arg1 * (x-self.center), dim=1, keepdims=True)**4

        bump_arg = torch.linalg.vector_norm(x, dim=(2), keepdims = True)**4

        tf_vec = (bump_arg <= 1/self.decay_param).squeeze()

        x_eval = x[tf_vec,:]

        weights_sparse = self.eval_MLPs(x_eval)

        idx = torch.nonzero(tf_vec, as_tuple=False).transpose(0,1)

        mesh_weights_sparse = mesh_weights.repeat(x.shape[0],1).reshape(x.shape[0],x.shape[1])[idx[0,:], idx[1,:]].reshape(1,1,-1)

        bump = (np.e*torch.exp(-1/(1 - self.decay_param*bump_arg[tf_vec]))).reshape(1,1,-1)

        temp = (weights_sparse * bump * mesh_weights_sparse).reshape(-1,self.channels_out, self.channels_in)

        #weight_array = []

        #for j in range(self.channels_out):
        #    for i in range(self.channels_in):
        #        weight_array.append(torch.sparse_coo_tensor(idx, temp[:,j,i], [x.shape[0],x.shape[1]]).coalesce())

        weights = torch.sparse_coo_tensor(idx, temp, [x.shape[0],x.shape[1],self.channels_out, self.channels_in]).coalesce()

        return weights


    def create_mlp(self, weight_mlp_chan):

        weight_func = [ [] ]

        linear_settings = {'bias' : False,
                            'device' : self.device,
                            'dtype' : torch.float,
                            }

        activation = sinLayer()

        weight_func = [nn.Linear(weight_mlp_chan[0], weight_mlp_chan[1], **linear_settings)]
        weight_func += [activation]

        for i in range(len(weight_mlp_chan)-3):
            weight_func += [nn.Linear(weight_mlp_chan[i+1],weight_mlp_chan[i+2], **linear_settings)]
            weight_func += [activation]

        weight_func += [nn.Linear(weight_mlp_chan[-2],weight_mlp_chan[-1], **linear_settings)]

        weight_func = nn.Sequential(*weight_func)

        return weight_func


    def set_quad(self, N):

        self.N = N

        self.decay_param = (N/4)**4

        if N < 1e3:

            if self.quad_type == 'gauss':

                self.quad_weights , self.quad_nodes = self.gauss_quad(N)

            elif self.quad_type == 'newton_cotes':

                self.quad_weights , self.quad_nodes = self.newton_cotes_quad(N)

        else:
            raise RuntimeError('Number of quadrature points exceeds the user set limit')


        self.quad_set_flag = True


    def gauss_quad(self, N):

        quad_weights = torch.zeros(N, device=self.device)
        quad_nodes = torch.zeros(N, device=self.device)

        for i in range(N):
            _, quad_weights[i], quad_nodes[i] = glpair(N,i+1)

        return quad_weights, quad_nodes


    def newton_cotes_quad(self, N, x0=0, x1=1):

        rep = int(N/self.composite_quad_order)

        dx = (x1-x0) / (self.composite_quad_order-1)

        weights , _ = newton_cotes(self.composite_quad_order-1,1)

        weights = np.tile(np.float32(dx * weights),rep)

        return torch.from_numpy(weights).to(self.device), torch.linspace(x0,x1,N,device=self.device, dtype=torch.float)


    def set_output_locs(self, locs):

        if isinstance(locs,int):

            if self.quad_type == 'gauss':

                _ , self.output_locs = self.gauss_quad(locs)

            elif self.quad_type == 'newton_cotes':

                _ , self.output_locs = self.newton_cotes_quad(locs)

        else:

            self.output_locs = locs


    def get_output_locs(self):

        node_list = [self.output_locs] * self.dimension

        mesh_nodes = torch.meshgrid(*node_list, indexing='xy')

        mesh_nodes = torch.dstack(mesh_nodes).reshape(-1,self.dimension)

        return mesh_nodes


    def get_quad_mesh(self):

        if self.quad_set_flag:

            node_list = [self.quad_nodes] * self.dimension

            mesh_nodes = torch.meshgrid(*node_list, indexing='xy')

            mesh_nodes = torch.dstack(mesh_nodes).reshape(-1,self.dimension)

            weight_list = [self.quad_weights] * self.dimension

            mesh_weights =  torch.meshgrid(*weight_list, indexing='xy')

            mesh_weights = torch.dstack(mesh_weights).reshape(-1,self.dimension)

        else:
            raise RuntimeError('Mesh not yet set')

        return mesh_nodes, mesh_weights


    def quad(self, features, output_locs, nodes, mesh_weights):

        eval_locs  = (torch.repeat_interleave(output_locs, nodes.shape[0], dim=0) - nodes.repeat(output_locs.shape[0],1)).reshape(output_locs.shape[0], nodes.shape[0], self.dimension)

        kf = self.kernel_func(eval_locs, mesh_weights)

        s = eval_locs.shape

        del(eval_locs)

        idx = kf.indices()

        batch_size = features.shape[0]

        ol =  output_locs.shape[0]
        il =  features.shape[2]

        kf_dense = torch.zeros(1,self.channels_out,self.channels_in,ol,il, device=self.device, dtype=torch.float)

        kf_dense[:,:,:,idx[0,:], idx[1,:]] = (kf.values()).reshape(1,self.channels_out,self.channels_in, -1)

        #integral = torch.zeros(batch_size, self.channels_out, ol, device=self.device)

        #for j in range(self.channels_out):
        #    for i in range(self.channels_in):
        #        integral[:,j,:] += torch.sparse.mm(kf[i + j*self.channels_in], features[:,i,:].reshape(il,batch_size)).reshape(batch_size,ol)

        #for j in range(self.channels_in):

        #    integral += torch.matmul(ewf[:,j,:,:], features[:,j,:].reshape(batch_size, 1, features.shape[2], 1)).reshape(batch_size, self.channels_out, -1)

        #integral = torch.matmul(kf.reshape(1,self.channels_out,self.channels_in,ol,il),
                                #features.reshape(batch_size,1,self.channels_in,il,1)).sum(dim=2).reshape(batch_size,self.channels_out,ol)

        integral = torch.einsum('b...dij, b...dj  -> b...i', kf_dense, features.reshape(batch_size,1,self.channels_in,il))

        return integral


    def tensor_prod_quad(self, features, output_locs):

        mesh_nodes, mesh_weights = self.get_quad_mesh()

        integral = self.quad(features,output_locs, mesh_nodes, mesh_weights)

        return integral


    def rquad(self, features, output_locs, level = (), nodes = () , weights = ()):

        if level == self.dimension or not level:
            nodes =  self.quad_nodes.unsqueeze(-1)
            weights = self.quad_weights.unsqueeze(-1)
            level = self.dimension


        if level == 1:

            integral = self.quad(features, output_locs, nodes, weights)

        elif level > 1:

            integral = torch.zeros(features.shape[0], self.channels_out, output_locs.shape[0], dtype=torch.float, device = self.device)

            for i in range(self.N):

                this_coord = self.quad_nodes[i].expand(self.N,1)
                this_weight = self.quad_weights[i]

                integral += self.rquad( features[:,:,(i*self.N):(i+1)*self.N], output_locs, level = level - 1 ,
                                        nodes = torch.hstack([nodes, this_coord]), weights = this_weight*weights)

        return integral


    def forward(self, features, output_locs=False):

        if isinstance(output_locs, bool) and not output_locs:
            output_locs = self.get_output_locs()

        if output_locs.shape[1] != self.dimension:
            raise RuntimeError('dimension mismatch')

        integral =  self.rquad(features, output_locs)

        if self.use_bias:
            if not torch.is_tensor(self.bias) and not self.bias:
                self.bias = torch.nn.Parameter(torch.zeros(1,integral.shape[1], integral.shape[2],device=self.device, dtype=torch.float))

            integral += self.bias

        return integral


class QuadConvBlock(nn.Module):

    def __init__(self, dimension=1,
                    channels_in=1,
                    channels_out=1,
                    N_in = False,
                    N_out = False,
                    adjoint = False,
                    activation1 = nn.CELU(alpha=1),
                    activation2 = nn.CELU(alpha=1),
                    mlp_chan = (4,8,4),
                    quad_type = 'gauss',
                    use_bias = False,
                    mlp_mode = 'single'
                    ):
        super().__init__()

        # N_out is the number of gaussian output points
        # if false, then the output locations must be specified by the set_output_locs function

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.adjoint = adjoint
        self.activation1 = activation1
        self.activation2 = activation2

        if self.adjoint:
            conv1_point_num = N_out
            conv1_channel_num = channels_out
        else:
            conv1_point_num = N_in
            conv1_channel_num = channels_in

        self.conv1 = QuadConvLayer( dimension,
                                    channels_in = conv1_channel_num,
                                    channels_out = conv1_channel_num,
                                    weight_mlp_chan = mlp_chan,
                                    kernel_mode = 'MLP',
                                    quad_type = quad_type,
                                    use_bias = use_bias,
                                    mlp_mode = mlp_mode
                                    )

        #self.batchnorm1 = torch.nn.BatchNorm1d(conv1_channel_num).to(self.device)
        self.batchnorm1 = torch.nn.InstanceNorm1d(conv1_channel_num, dtype=torch.float, device=self.device)

        self.conv2 = QuadConvLayer( dimension,
                                    channels_in = channels_in,
                                    channels_out = channels_out,
                                    weight_mlp_chan = mlp_chan,
                                    kernel_mode = 'MLP',
                                    quad_type = quad_type,
                                    use_bias = use_bias,
                                    mlp_mode = mlp_mode
                                    )

        #self.batchnorm2 = torch.nn.BatchNorm1d(channels_out).to(self.device)
        self.batchnorm2 = torch.nn.InstanceNorm1d(channels_out, dtype=torch.float, device=self.device)


        if N_in and N_out:
            self.conv1.set_quad(conv1_point_num)
            self.conv2.set_quad(N_in)

            self.conv1.set_output_locs(conv1_point_num)
            self.conv2.set_output_locs(N_out)


    def forward(self,data):

        if self.adjoint:
            output = self.adjoint_op(data)
        else:
            output = self.forward_op(data)

        return output


    def forward_op(self,data):

        x = data

        x1 = self.conv1(x)
        x1 = self.activation1(self.batchnorm1(x1))
        x1 = x1 + x

        x2 = self.conv2(x1)
        x2 = self.activation2(self.batchnorm2(x2))

        return x2


    def adjoint_op(self,data):

        x = data

        x2 = self.conv2(x)
        x2 = self.activation2(self.batchnorm2(x2))

        x1 = self.conv1(x2)
        x1 = self.activation1(self.batchnorm1(x1))
        x1 = x1 + x2


        return x1


class AutoQuadConv(nn.Module):

    def __init__(self,
        feature_dim = (16, 1, 10000),
        dimension = 2,
        compressed_dim = 100,
        point_seq = (100, 20, 5),
        channel_sequence = [8],
        final_activation = nn.Tanh(),
        mlp_chan = (4,8,4),
        quad_type = 'gauss',
        use_bias = False,
        mlp_mode = 'single',
        adjoint_activation = nn.CELU(alpha=1),
        forward_activation = nn.CELU(alpha=1),
        middle_activation = nn.CELU(alpha=1),
        noise_eps = 0.0
    ):
        super().__init__()

        # data shape info
        init_points_in = feature_dim[2]
        self.quad_type = quad_type
        self.noise_eps = noise_eps

        #define the final activation
        self.final_activation = final_activation
        self.forward_activation = forward_activation
        self.adjoint_activation = adjoint_activation
        self.middle_activation = middle_activation

        block_depth = len(point_seq) - 1

        # GPU if its there
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Build the network
        encoder_layer_stack = []
        decoder_layer_stack = []


        for i in range(block_depth):
            encoder_layer_stack += [QuadConvBlock(  dimension,
                                                    channel_sequence[i],
                                                    channel_sequence[i+1],
                                                    N_in=point_seq[i],
                                                    N_out=point_seq[i+1],
                                                    mlp_chan=mlp_chan,
                                                    quad_type=self.quad_type,
                                                    use_bias = use_bias,
                                                    mlp_mode = mlp_mode,
                                                    activation1 = self.forward_activation,
                                                    activation2 = self.forward_activation
                                                    )]


        self.encoder = nn.Sequential(*encoder_layer_stack)

        self.flat = nn.Flatten(start_dim=1,end_dim=-1)
        self.linear_down = spn(nn.Linear(channel_sequence[-1]*(point_seq[-1]**dimension), compressed_dim, dtype=torch.float, device=self.device))
        self.linear_down2 = spn(nn.Linear(compressed_dim, compressed_dim, dtype=torch.float, device=self.device))
        #######
        self.unflat = nn.Unflatten(1,[channel_sequence[-1],point_seq[-1]**dimension])
        self.linear_up2 = spn(nn.Linear(compressed_dim, compressed_dim, dtype=torch.float, device=self.device))
        self.linear_up = spn(nn.Linear(compressed_dim, (point_seq[-1]**dimension)*channel_sequence[-1], dtype=torch.float, device=self.device))


        for i in range(block_depth-1):
            i = i + 1
            decoder_layer_stack += [QuadConvBlock(  dimension,
                                                    channel_sequence[-i],
                                                    channel_sequence[-(i+1)],
                                                    N_in = point_seq[-i],
                                                    N_out = point_seq[-(i+1)],
                                                    adjoint = True,
                                                    mlp_chan = mlp_chan,
                                                    quad_type=self.quad_type,
                                                    use_bias = use_bias,
                                                    mlp_mode = mlp_mode,
                                                    activation1 = self.adjoint_activation,
                                                    activation2 = self.adjoint_activation
                                                    )]


        decoder_layer_stack += [QuadConvBlock(  dimension,
                                                channel_sequence[1],
                                                channel_sequence[0],
                                                N_in = point_seq[1],
                                                N_out = point_seq[0],
                                                adjoint = True,
                                                mlp_chan = mlp_chan,
                                                quad_type = self.quad_type,
                                                use_bias = use_bias,
                                                mlp_mode = mlp_mode,
                                                activation1 = nn.Identity(),
                                                activation2 = self.adjoint_activation
                                                )]

        self.decoder = nn.Sequential(*decoder_layer_stack)


    def encode(self, features):

        output = self.encoder(features)

        compressed = self.linear_down(self.flat(output))

        compressed = self.middle_activation(compressed)

        compressed = self.middle_activation(self.linear_down2(compressed))

        return compressed


    def decode(self, compressed):

        if self.training == True:
            compressed += self.noise_eps * torch.randn_like(compressed[0,:])

        compressed = self.middle_activation(self.linear_up2(compressed))

        unflat_uncompressed = self.unflat(self.linear_up(compressed))

        output = self.final_activation(self.decoder(unflat_uncompressed))


        return output


    def forward(self, features):

        compressed = self.encode(features)

        c1 = self.decode(compressed)

        return c1, compressed


class sinLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,x):
        return torch.sin(x)


def apply_bn(x, bn):

    return bn(x.transpose(1,2)).transpose(1,2)


def x_func(x,f):
    return f(x)
