import torch
import torch.nn as nn
from scipy.spatial import Delaunay
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import weakref
from torch_quadconv.utils.quadrature import mfnus
from scipy.spatial import KDTree



class Grid(nn.Module):
    def __init__(self, num_points_per_dim, domain=None):
        """
        # Usage Example
            domain = [(0, 1), (0, 1)]  # 2D grid from (0,0) to (1,1)
            num_points_per_dim = [3, 3]  # 3x3 grid
            grid = Grid(num_points_per_dim, domain)
            print("Initial shape:", grid.shape())
            print("Initial points:", grid.points)
        """
        super(Grid, self).__init__()

        # Validate num_points_per_dim
        for n in num_points_per_dim:
            assert n > 1, "There must be at least two points per dimension"

        self.spatial_dimension = len(num_points_per_dim)
        self.num_points_per_dim = num_points_per_dim
        self.kdtree = None
        self.num_points = torch.prod(torch.tensor(num_points_per_dim))

        # If domain is not provided, assume it's [(0, 1)] * len(num_points_per_dim)
        if domain is None:
            self.domain = [(0, 1)] * self.spatial_dimension
        else:
            self.domain = domain

        # Ensure the inputs are valid
        assert len(self.domain) == self.spatial_dimension, "Domain length must match num_points_per_dim length"
        for d in self.domain:
            assert len(d) == 2 and d[0] < d[1], "Each domain range must be a tuple of two values (min, max)"

        # Generate points
        points = []
        for dim, (min_val, max_val) in enumerate(self.domain):
            points.append(torch.linspace(min_val, max_val, num_points_per_dim[dim]))

        # Create a meshgrid and reshape it to have a list of points
        meshgrid = torch.meshgrid(points)

        # Register points as a parameter of the module
        self.register_buffer('points', torch.stack([g.reshape(-1) for g in meshgrid], dim=-1))

        #self.points = torch.stack([g.reshape(-1) for g in meshgrid], dim=-1)
        self.num_points = self.points.shape[0]

        self.register_buffer('adjacency', torch.from_numpy(Delaunay(self.points).simplices).long())

    def __repr__(self):
        return f"Grid( {self.num_points} points in {self.spatial_dimension} dimensions)"
    
    def display(self, ax=None, **kwargs):
        #displays an image of the mesh as a scatter plot

        #set default plot values 
        for key, value in {'s': 1, 'edgecolor': 'k', 'facecolor' : 'w', 'alpha' : 0.5}.items():
            if key not in kwargs:
                kwargs[key] = value

        if ax is None:
            ax = plt.gca()

        ax.scatter(self.points[:,0], self.points[:,1], **kwargs)
        plt.show()
        
        return ax

    def query(self, range, knn=1):

        if self.spatial_dimension != range.spatial_dimension:
            raise ValueError("Spatial dimensions must match")
        
        if self.domain != range.domain:
            raise ValueError("Domains must match")
        
        if self.kdtree is None:
            self.kdtree = KDTree(self.points)

        return self.kdtree.query(range.points, k=knn)

    def downsample(self, factor):
        assert factor >= 1, "Factor must be greater or equal to 1"
        new_num_points_per_dim = [max(2, n // factor) for n in self.num_points_per_dim]
        # Construct a new Grid object with the new number of points per dimension and the same domain.
        return Grid(new_num_points_per_dim, domain=self.domain)


    def upsample(self, factor):
        assert factor >= 1, "Factor must be greater or equal to 1"
        new_num_points_per_dim = [n * factor for n in self.num_points_per_dim]
        # Construct a new Grid object with the new number of points per dimension and the same domain.
        return Grid(new_num_points_per_dim, domain=self.domain)
    

    def pool_map(self, kernel_size):

        # if the kernel size is an integer, broadcast it into a tuple which the same value repeated for each dimension
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * self.spatial_dimension

        index_map = torch.arange(self.num_points).view(self.num_points_per_dim)

        # Reshape index map so that each dimension has the indices which will be pooled together
        index_tensor = index_map.reshape(*kernel_size, -1).mT.reshape(-1, *kernel_size)

        # Turn the index tensor into a dictionary where the keys are the order of the tensors
        index_dict = { i : index_tensor[i].flatten() for i in range(index_tensor.shape[0]) }

        return index_dict




    @property
    def shape(self):
        return (self.num_points, self.spatial_dimension)



class Mesh(nn.Module):
    def __init__(self, points = None, adjacency = None, domain = None, data_dir = None, multilevel_map=mfnus, parent = None):
        """
        # Usage Example
            domain = [(0, 1), (0, 1)]  # 2D grid from (0,0) to (1,1)
            num_points_per_dim = [3, 3]  # 3x3 grid
            grid = Grid(num_points_per_dim, domain)
            print("Initial shape:", grid.shape())
            print("Initial points:", grid.points)
        """
        super(Mesh, self).__init__()

        self.multilevel_map = multilevel_map

        self.parent = parent

        if data_dir is not None:
            points, adjacency, domain = self.read_path(data_dir)

        self.num_points = points.shape[0]
        self.spatial_dimension = points.shape[1]
        self.kdtree = None

        # If domain is not provided, assume it's [(0, 1)]
        if domain is None:
            self.domain = [(0, 1)] * self.spatial_dimension
        else:
            self.domain = domain

        # Ensure the inputs are valid
        assert len(self.domain) == self.spatial_dimension, "Domain length must match spatial dim"
        for d in self.domain:
            assert len(d) == 2 and d[0] < d[1], "Each domain range must be a tuple of two values (min, max)"

        # Register points as a parameter of the module
        self.register_buffer('points', points)

        if adjacency is None:
            adjacency = torch.from_numpy(Delaunay(self.points).simplices).long()

        self.register_buffer('adjacency', adjacency)

    def __repr__(self):
        return f"Mesh( {self.num_points} points in {self.spatial_dimension} dimensions )"

    def display(self, ax=None, **kwargs):
        #displays an image of the mesh as a scatter plot

        #set default plot values 
        for key, value in {'s': 1, 'edgecolor': 'k', 'facecolor' : 'w', 'alpha' : 0.5}.items():
            if key not in kwargs:
                kwargs[key] = value

        if ax is None:
            ax = plt.gca()

        ax.scatter(self.points[:,0], self.points[:,1], **kwargs)
        plt.show()

        return ax

    def read_path(self, data_dir):

        data_path = Path(data_dir)

        domain = None

        #look for points file
        if data_path.is_dir():
            points_file = data_path.joinpath('points.npy')
        elif data_path.is_file():
            points_file = data_path
            data_path = data_path.parent

        try:
            points = torch.from_numpy(np.float32(np.load(points_file)))

            #look for adjacency file
            adj_files = list(data_path.glob('adj*.npy'))

            if len(adj_files) > 1:
                try:
                    adjacency = torch.from_numpy(np.float32(np.load(adj_files[0])))

                except FileNotFoundError:
                    adjacency = None

                except Exception as e:
                    raise e
            else:
                adjacency = None

        except FileNotFoundError:
            points = None
            adjacency = None

        except Exception as e:
            raise e
        
        return points, adjacency, domain

    def query(self, range, k=1):

        if self.spatial_dimension != range.spatial_dimension:
            raise ValueError("Spatial dimensions must match")
        
        if self.domain != range.domain:
            raise ValueError("Domains must match")
        
        if self.kdtree is None:
            self.kdtree = KDTree(self.points.cpu())

        return self.kdtree.query(range.points.cpu(), k=k)

    def downsample(self, factor=2):
        assert factor >= 1, "Factor must be greater or equal to 1"

        points, weights, elim_map = self.multilevel_map(self.points, appx_ds=factor)

        '''A weak reference to the parent is used to avoid circular references, which would prevent the parent from being garbage collected. 
        This is important because the parent is likely to be a large object that we want to be able to delete when it is no longer needed.
        HOWEVER, this throws an error on the way into the trainer.fit function with lightning, since its an unhashable object'''

        #return Mesh(points = torch.from_numpy(points), domain=self.domain, parent = weakref.proxy(self))
        return Mesh(points = torch.from_numpy(points), domain=self.domain, parent = self)


    def upsample(self):
        
        assert self.parent is not None, "Cannot upsample a mesh without a parent"

        return self.parent
    

    def pool_map(self, factor=2):

        _, _, elim_map = self.multilevel_map(self.points, appx_ds=factor)

        return elim_map

    @property
    def shape(self):
        return (self.num_points, self.spatial_dimension)
