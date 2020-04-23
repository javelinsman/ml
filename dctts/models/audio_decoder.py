import torch
import torch.nn as nn
import torch.nn.functional as F
from util.layers import CustomConv1d, HighwayConv1d
from .config import DIMENSIONS as D

class AudioDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        d = D['latent']
        self.layers = nn.ModuleList([
#             CustomConv1d(2 * d, d, 1, 1, 'causal'),
            CustomConv1d(d, d, 1, 1, 'causal'),
            HighwayConv1d(d, 3, 1, padding_mode='causal'),
            HighwayConv1d(d, 3, 3, padding_mode='causal'),
            HighwayConv1d(d, 3, 9, padding_mode='causal'),
            HighwayConv1d(d, 3, 27, padding_mode='causal'),
            HighwayConv1d(d, 3, 1, padding_mode='causal'),
            HighwayConv1d(d, 3, 1, padding_mode='causal'),
            CustomConv1d(d, d, 1, 1, 'causal'), nn.ReLU(),
            CustomConv1d(d, d, 1, 1, 'causal'), nn.ReLU(),
            CustomConv1d(d, d, 1, 1, 'causal'), nn.ReLU(),
            CustomConv1d(d, D['F'], 1, 1, 'causal'), nn.Sigmoid(),
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x