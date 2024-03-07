import torch
import torch.nn as nn
import numpy as np


'''
Credit to Vincent Sitzmann for SIREN:
https://github.com/vsitzmann/siren/blob/master/modules.py
'''

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30.):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.out_features = out_features

        self.register_module('linear', nn.Linear(in_features, out_features, bias=bias))    
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
class Siren(nn.Module):
    def __init__(self, mlp_spec, outermost_linear=False, 
                 first_omega_0=1., hidden_omega_0=1.):
        super().__init__()
        
        self.register_module('net', nn.Sequential())
        self.register_module('normalize', nn.InstanceNorm1d(mlp_spec[0]))

        self.net.append(SineLayer(mlp_spec[0], mlp_spec[1], 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(1,len(mlp_spec)-2):
            self.net.append(SineLayer(mlp_spec[i], mlp_spec[i+1], 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(mlp_spec[-2], mlp_spec[-1], bias=True)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / mlp_spec[-2]), 
                                              np.sqrt(6 / mlp_spec[-2]))
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(mlp_spec[-2], mlp_spec[-1], 
                                      is_first=False, omega_0=hidden_omega_0, bias=True))
        

    def forward(self, coords):
        return self.net(coords)