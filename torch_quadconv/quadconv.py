import torch
import torch.nn as nn

from pykeops.torch import LazyTensor
from pykeops.torch.cluster import grid_cluster, sort_clusters, cluster_ranges_centroids, from_matrix


class QuadConv(nn.Module):

    def __init__(self,*,
            domain,
            range,
            in_channels,
            out_channels,
            filter_seq = [16, 16, 16, 16, 16],
            decay_param = None,
            bias = False,
            output_same = False,
            cache = True
        ):
        super().__init__()

        #validate spatial dim
        assert domain.shape[1] == range.shape[1], "Domain and range must have the same spatial dimension"

        #set attributes
        self.spatial_dim = domain.shape[1]

        self.add_module('domain', domain)
        self.add_module('range', range)

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.output_same = output_same

        self.cache = cache
        self.cached = False
        self.weight_map = None

        #decay parameter
        if decay_param == None:
            self.decay_param = (self.domain.shape[0]/16)**2
        else:
            self.decay_param == decay_param

        #initialize filter
        self._init_filter(filter_seq)

        self._build_weight_map()

        self.cache_ranges()

        #bias
        if bias:
            bias = torch.empty(1, self.out_channels, self.range.shape[0])
            self.bias = nn.Parameter(nn.init.xavier_uniform_(bias, gain=2), requires_grad=True)
        else:
            self.bias = None

        return
    

    def cache_ranges(self):

        eps = 0.1  # Size of our square bins

        alpha = self.decay_param

        x_labels = grid_cluster(self.domain.points, eps)  # class labels
        y_labels = grid_cluster(self.range.points, eps)  # class labels

    
        # Compute one range and centroid per class:
        x_ranges, x_centroids, _ = cluster_ranges_centroids(self.domain.points, x_labels)
        y_ranges, y_centroids, _ = cluster_ranges_centroids(self.range.points, y_labels)


        x, x_labels = sort_clusters(self.domain.points, x_labels)
        y, y_labels = sort_clusters(self.range.points, y_labels)
            

        keep = (((y_centroids[None, :, :]-x_centroids[:, None, :])**2).sum(-1) - 1 / alpha) < 0


        keep = keep
        x_ranges = x_ranges
        y_ranges = y_ranges
                
        ranges_ij = from_matrix(x_ranges, y_ranges, keep)

        self.ranges_ij = ranges_ij

        self.cached = True

        return

    '''
    Initialize the layer filter.

    Input:
        filter_seq: mlp feature sequence
        filter_mode: type of filter operation
    '''
    def _init_filter(self, filter_seq):


        mlp_spec = (self.spatial_dim, *filter_seq, self.in_channels*self.out_channels)

        self._lazy_mlp_list = nn.ParameterList()

        for i in range(len(mlp_spec)-1):
            self._lazy_mlp_list.append(nn.Parameter(torch.randn(mlp_spec[i] , mlp_spec[i+1])))

        return
    

    def _lazy_kernel(self, x, y):

        alpha = self.decay_param

        out = y - x

        domain = (out.sqnorm2() - 1 / alpha).ifelse(1,0)

        bump = (-1 / (1 - alpha * out.sqnorm2())).exp() * domain

        for lazy in self._lazy_mlp_list:

            this_lazy = LazyTensor(lazy.data.view(-1))

            out = this_lazy.matvecmult(out).sin()
        
        return out * bump
    

    def _build_weight_map(self, element_size = 3, dimension = 2):

        self.weight_map = nn.Sequential(
            nn.Linear(element_size * dimension, 8),
            nn.Sigmoid(),
            nn.Linear(8, 8),
            nn.Sigmoid(),
            nn.Linear(8, 8),
            nn.Sigmoid(),
            nn.Linear(8, element_size),
            nn.Sigmoid()
        )

        return
    

    def eval_weight_map(self, domain, element_size = 3, dimension = 2):

        element_list = domain.adjacency

        el_points = domain.points[element_list].reshape(-1, element_size * dimension)

        el_weights =  self.weight_map(el_points)

        weights = torch.zeros(domain.points.shape[0]).to(el_points)

        weights.scatter_add_(0, element_list.reshape(-1), el_weights.reshape(-1))

        return weights


    '''
    Apply operator via quadrature approximation of convolution with features and learned filter.

    Input:
        features: a tensor of shape (batch size  X num of points X input channels)

    Output: tensor of shape (batch size X num of output points X output channels)
    '''
    def forward(self, features):

        #use_cuda = torch.cuda.is_available()
        #dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        #get weights
        weights = self.eval_weight_map(self.domain)

        features = features.permute(0, 2, 1).contiguous()

        f = LazyTensor(features.view(features.shape[0], features.shape[1], 1, features.shape[2]))

        x_i = LazyTensor(self.domain.points.view(self.domain.num_points, 1, self.spatial_dim))   

        y_j = LazyTensor(self.range.points.view(1, self.range.num_points, self.spatial_dim))

        ex_tensor = LazyTensor(torch.ones(1, 1, 1, self.out_channels))

        ex_f = f.keops_kron(ex_tensor, [self.in_channels], [self.out_channels])

        rho_i = LazyTensor(weights.view(self.domain.num_points, 1, 1))

        G_ij = self._lazy_kernel(x_i,  y_j) * rho_i * ex_f

        if self.cached:
            #TODO This is waiting for a KeOps update to support sparse computation on the GPU (currently only CPU)
            #G_ij.ranges = self.ranges_ij
            pass

        integral = G_ij.sum_reduction(axis=1).reshape(features.shape[0], self.range.num_points, self.in_channels, self.out_channels).sum(dim=-2)

        #add bias
        if self.bias is not None:
            integral += self.bias

        return integral