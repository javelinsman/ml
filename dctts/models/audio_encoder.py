import torch
import torch.nn as nn
import torch.nn.functional as F
from util.layers import CustomConv1d, HighwayConv1d

from .config import DIMENSIONS as D

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        d = D['latent']
        self.layers = nn.ModuleList([
            CustomConv1d(D['F'], d, 1, dilation=1), nn.ReLU(),
            CustomConv1d(d, d, 1, dilation=1), nn.ReLU(),
            CustomConv1d(d, d, 1, dilation=1),

            HighwayConv1d(d, 3, 1, padding_mode='causal'),
            HighwayConv1d(d, 3, 3, padding_mode='causal'),
            HighwayConv1d(d, 3, 9, padding_mode='causal'),
            HighwayConv1d(d, 3, 27, padding_mode='causal'),
            HighwayConv1d(d, 3, 1, padding_mode='causal'),
            HighwayConv1d(d, 3, 3, padding_mode='causal'),
            HighwayConv1d(d, 3, 9, padding_mode='causal'),
            HighwayConv1d(d, 3, 27, padding_mode='causal'),
            HighwayConv1d(d, 3, 3, padding_mode='causal'),
            HighwayConv1d(d, 3, 3, padding_mode='causal'),
        ])
    
    def forward(self, x):
        for layer in self.layers[:1]:
            x = layer(x)
        return x
