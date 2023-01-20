import torch
import torch.nn as nn

import numpy as np

'''
Quad Loss function; Computes the MSE Loss and constrains the quadrature weights

Input:
    model_pointer: the model from which to grab the quad weights
'''


class QuadLoss(nn.Module):
    
    def __init__(self,*,
            mesh_pointer,
            order = 0,
        ):
        super().__init__()

        self.mesh = mesh_pointer
        self.order = order # exactness of a number of low degree polys?

    def forward(self, pred, target):

        # Standard MSE component
        loss = nn.functional.mse_loss(pred, target)**(0.5)

        # Get quadrature params, if any
        for w in self.mesh._weights:
            if w.requires_grad:
                loss += 0.1*nn.functional.mse_loss(torch.ones((1), device=w.device), torch.sum(w))**(0.5)

        return loss/pred.shape[0]