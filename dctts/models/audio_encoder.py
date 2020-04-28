import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm as norm
from util.layers import CustomConv1d, HighwayConv1d
import util.yangyang as mm

from .config import DIMENSIONS as D

class SimpleAudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        d = D['latent']
        self.layers = nn.ModuleList([
            mm.CausalConv1d(D['F'], d, 5, dilation=1), nn.ReLU(),
            mm.CausalConv1d(d, d, 5, dilation=1), nn.ReLU(),
            mm.CausalConv1d(d, d, 5, dilation=1)
            # CustomConv1d(D['F'], d, 5, 1, 'causal'), nn.ReLU(),
            # CustomConv1d(d, d, 5, 1, 'causal'), nn.ReLU(),
            # CustomConv1d(d, d, 5, 1, 'causal')
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class YYAudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        d = D['latent']
        self.layers = nn.ModuleList([
            norm(mm.CausalConv1d(D['F'], d, 1, dilation=1)),
            norm(mm.CausalConv1d(d, d, 1, dilation=1)), nn.ReLU(),
            norm(mm.CausalConv1d(d, d, 1, dilation=1)), nn.ReLU(),
            norm(mm.CausalHighwayConv1d(d, d, 3, dilation=1)),
            norm(mm.CausalHighwayConv1d(d, d, 3, dilation=3)),
            norm(mm.CausalHighwayConv1d(d, d, 3, dilation=9)),
            norm(mm.CausalHighwayConv1d(d, d, 3, dilation=27)),
            norm(mm.CausalHighwayConv1d(d, d, 3, dilation=1)),
            norm(mm.CausalHighwayConv1d(d, d, 3, dilation=3)),
            norm(mm.CausalHighwayConv1d(d, d, 3, dilation=9)),
            norm(mm.CausalHighwayConv1d(d, d, 3, dilation=27)),
            norm(mm.CausalHighwayConv1d(d, d, 3, dilation=3)),
            norm(mm.CausalHighwayConv1d(d, d, 3, dilation=3)),
        ])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        d = D['latent']
        self.layers = nn.ModuleList([
            (CustomConv1d(D['F'], d, 1, 1, 'causal')), nn.ReLU(),
            (CustomConv1d(d, d, 1, 1, 'causal')), nn.ReLU(),
            (CustomConv1d(d, d, 1, 1, 'causal')),
            (HighwayConv1d(d, 3, 1, 'causal')),
            (HighwayConv1d(d, 3, 3, 'causal')),
            (HighwayConv1d(d, 3, 9, 'causal')),
            (HighwayConv1d(d, 3, 27, 'causal')),
            (HighwayConv1d(d, 3, 1, 'causal')),
            (HighwayConv1d(d, 3, 3, 'causal')),
            (HighwayConv1d(d, 3, 9, 'causal')),
            (HighwayConv1d(d, 3, 27, 'causal')),
            (HighwayConv1d(d, 3, 3, 'causal')),
            (HighwayConv1d(d, 3, 3, 'causal')),
        ])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
