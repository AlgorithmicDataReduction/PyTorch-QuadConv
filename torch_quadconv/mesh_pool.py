

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class Mesh_MaxPool(nn.Module):


    def __init__(self, pool_map, adjoint=False):
        super().__init__()

        self.adjoint = adjoint

        self.pool_map = pool_map

        self.cached = False

        self.index = None
        self.output_shape = None

        return

    def forward(self, data):
        x = data

        if self.index is not None: 
            if (self.output_shape[0] != x.shape[0]):
                self.index = None

        if self.index is None:
             
            if not self.adjoint:
                  
                elim_map = self.pool_map

                for i in elim_map.keys():
                    elim_map[i] = np.append(elim_map[i], i)

                self.index = torch.zeros_like(x, dtype=torch.int64, device=x.device)

                for i,j in enumerate(elim_map):
                    for k in elim_map[j]:
                        self.index[:, :, int(k)] = int(i)

                self.output_shape = (x.shape[0], x.shape[1], len(elim_map))

            elif self.adjoint:
             
                forward_elim_map = self.pool_map

                new_ind = {}
                for i,j in enumerate(forward_elim_map):
                    new_ind[j] = i

                for i in forward_elim_map.keys():
                    forward_elim_map[i] = np.append(forward_elim_map[i], i)

                inv_map = {}
                for k,v in forward_elim_map.items():
                    for vv in v:
                        v_get = inv_map.get(vv,[])
                        inv_map[vv] = v_get+[k]

                self.index = torch.zeros(x.shape[0], x.shape[1], len(inv_map), dtype=torch.int64, device=x.device)

                for i in inv_map:
                    for j in inv_map[i]:
                        self.index[:, :, int(i)] = int(new_ind[j])

                self.output_shape = (x.shape[0], x.shape[1], len(inv_map))
                  

        if not self.adjoint:

            output = torch.zeros(*self.output_shape, device=x.device)
            output.scatter_reduce_(dim=2, index=self.index, src=x, reduce='amax', include_self=False)

        elif self.adjoint:

                output = torch.gather(input=x, dim=2, index=self.index)


        return output
    




class Mesh_AvgPool(nn.Module):


    def __init__(self, pool_map, adjoint=False):
        super().__init__()

        self.adjoint = adjoint

        self.pool_map = pool_map

        self.cached = False

        self.index = None
        self.output_shape = None

        return

    def forward(self, data):
        x = data

        if self.index is not None: 
            if (self.output_shape[0] != x.shape[0]):
                self.index = None

        if self.index is None:
             
            if not self.adjoint:
                  
                elim_map = self.pool_map

                for i in elim_map.keys():
                    elim_map[i] = np.append(elim_map[i], i)

                self.index = torch.zeros_like(x, dtype=torch.int64, device=x.device)

                for i,j in enumerate(elim_map):
                    for k in elim_map[j]:
                        self.index[:, :, int(k)] = int(i)

                self.output_shape = (x.shape[0], x.shape[1], len(elim_map))

            elif self.adjoint:
             
                forward_elim_map = self.pool_map

                new_ind = {}
                for i,j in enumerate(forward_elim_map):
                    new_ind[j] = i

                for i in forward_elim_map.keys():
                    forward_elim_map[i] = np.append(forward_elim_map[i], i)

                inv_map = {}
                for k,v in forward_elim_map.items():
                    for vv in v:
                        v_get = inv_map.get(vv,[])
                        inv_map[vv] = v_get+[k]

                self.index = torch.zeros(x.shape[0], x.shape[1], len(inv_map), dtype=torch.int64, device=x.device)

                for i in inv_map:
                    for j in inv_map[i]:
                        self.index[:, :, int(i)] = int(new_ind[j])

                self.output_shape = (x.shape[0], x.shape[1], len(inv_map))
                  

        if not self.adjoint:

            output = torch.zeros(*self.output_shape, device=x.device)
            output.scatter_reduce_(dim=2, index=self.index, src=x, reduce='mean', include_self=False)

        elif self.adjoint:

                output = torch.gather(input=x, dim=2, index=self.index)


        return output