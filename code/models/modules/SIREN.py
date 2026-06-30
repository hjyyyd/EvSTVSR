import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np

import time


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
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
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features[0], 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features[i], hidden_features[i + 1], 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features[-1], out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features[-1]) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features[-1]) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features[-1], out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output     
    
class warp_Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features1=2,out_features2=1, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(Siren(in_features= in_features , out_features=32, hidden_features=hidden_features,
            hidden_layers=hidden_layers, outermost_linear=outermost_linear))

        
        self.net = nn.Sequential(*self.net)

        self.flow_interp_out_layer= nn.Linear(in_features=32, out_features=out_features1, bias=True)
        self.flow_interp_vis_layer= nn.Linear(in_features=32, out_features=out_features2, bias=True)
    
    def forward(self, coords):
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        
        # output = self.net(coords)  # b,q,32
        flow_interp_motion_rep = self.net(coords)  # b,q,32

        flow_interp_flow = self.flow_interp_out_layer(flow_interp_motion_rep)
        flow_interp_vis_map = self.flow_interp_vis_layer(flow_interp_motion_rep)
        flow_interp_vis_map = torch.sigmoid(flow_interp_vis_map)

        return flow_interp_flow, flow_interp_vis_map
    
    
# class Siren_mask(nn.Module):
#     def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
#                  first_omega_0=30, hidden_omega_0=30.):
#         super().__init__()
        
#         self.net = []
#         self.net.append(SineLayer(in_features, hidden_features[0], 
#                                   is_first=True, omega_0=first_omega_0))

#         for i in range(hidden_layers):
#             self.net.append(SineLayer(hidden_features[i], hidden_features[i + 1], 
#                                       is_first=False, omega_0=hidden_omega_0))

#         if outermost_linear:
#             final_linear = nn.Linear(hidden_features[-1], out_features)
            
#             with torch.no_grad():
#                 final_linear.weight.uniform_(-np.sqrt(6 / hidden_features[-1]) / hidden_omega_0, 
#                                               np.sqrt(6 / hidden_features[-1]) / hidden_omega_0)
                
#             self.net.append(final_linear)
#         else:
#             self.net.append(SineLayer(hidden_features[-1], out_features, 
#                                       is_first=False, omega_0=hidden_omega_0))
        
#         self.net = nn.Sequential(*self.net)
#         self.output_layer = nn.Linear(in_features, out_features, bias=bias)
    
#     def forward(self, coords):
#         # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
#         output = self.net(coords)
#         return output     