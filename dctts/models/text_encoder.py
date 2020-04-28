import torch
import torch.nn as nn
import torch.nn.functional as F
from util.g2p import phoneme_types
from util.layers import CustomConv1d, HighwayConv1d
from torch.nn.utils import weight_norm as norm
from .config import DIMENSIONS as D
import util.yangyang as mm

class SimpleTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        d = D['latent']
        e = D['embedding']
        self.embedding = nn.Embedding(len(phoneme_types), e)
        self.layers = nn.ModuleList([
            mm.Conv1d(e, 2 * d, 5, dilation=1, padding='same'), nn.ReLU(),
            mm.Conv1d(2 * d, 2 * d, 5, dilation=1, padding='same'), nn.ReLU(),
            mm.Conv1d(2 * d, 2 * d, 5, dilation=1, padding='same'),
            # CustomConv1d(e, 2 * d, 5, 1), nn.ReLU(),
            # CustomConv1d(2 * d, 2 * d, 5, 1), nn.ReLU(),
            # CustomConv1d(2 * d, 2 * d, 5, 1),
        ])
    
    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        for layer in self.layers:
            x = layer(x)
        t_att, t_chr = x[:,:D['latent'],:], x[:,D['latent']:,:]
        return t_att, t_chr

class YYTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        d = D['latent']
        e = D['embedding']
        self.embedding = nn.Embedding(len(phoneme_types), e)
        self.layers = nn.ModuleList([
            norm(mm.Conv1d(e, 2 * d, 1, dilation=1)), nn.ReLU(),
            norm(mm.Conv1d(2 * d, 2 * d, 1, dilation=1)),
            norm(mm.HighwayConv1d(2 * d, 2 * d, 3, dilation=1)),
            norm(mm.HighwayConv1d(2 * d, 2 * d, 3, dilation=3)),
            norm(mm.HighwayConv1d(2 * d, 2 * d, 3, dilation=9)),
            norm(mm.HighwayConv1d(2 * d, 2 * d, 3, dilation=27)),
            norm(mm.HighwayConv1d(2 * d, 2 * d, 3, dilation=1)),
            norm(mm.HighwayConv1d(2 * d, 2 * d, 3, dilation=3)),
            norm(mm.HighwayConv1d(2 * d, 2 * d, 3, dilation=9)),
            norm(mm.HighwayConv1d(2 * d, 2 * d, 3, dilation=27)),
            norm(mm.HighwayConv1d(2 * d, 2 * d, 3, dilation=1)),
            norm(mm.HighwayConv1d(2 * d, 2 * d, 3, dilation=1)),
            norm(mm.HighwayConv1d(2 * d, 2 * d, 1, dilation=1)),
            norm(mm.HighwayConv1d(2 * d, 2 * d, 1, dilation=1)),
        ])
    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        for layer in self.layers:
            x = layer(x)
        t_att, t_chr = x[:,:D['latent'],:], x[:,D['latent']:,:]
        return t_att, t_chr

    

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        d = D['latent']
        e = D['embedding']
        self.embedding = nn.Embedding(len(phoneme_types), e)
        self.layers = nn.ModuleList([
            (CustomConv1d(e, 2 * d, 1, 1)), nn.ReLU(),
            (CustomConv1d(2 * d, 2 * d, 1, 1)),
            (HighwayConv1d(2 * d, 3, 1)),
            (HighwayConv1d(2 * d, 3, 3)),
            (HighwayConv1d(2 * d, 3, 9)),
            (HighwayConv1d(2 * d, 3, 27)),
            (HighwayConv1d(2 * d, 3, 1)),
            (HighwayConv1d(2 * d, 3, 3)),
            (HighwayConv1d(2 * d, 3, 9)),
            (HighwayConv1d(2 * d, 3, 27)),
            (HighwayConv1d(2 * d, 3, 1)),
            (HighwayConv1d(2 * d, 3, 1)),
            (HighwayConv1d(2 * d, 1, 1)),
            (HighwayConv1d(2 * d, 1, 1)),
        ])
    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        for layer in self.layers:
            x = layer(x)
        t_att, t_chr = x[:,:D['latent'],:], x[:,D['latent']:,:]
        return t_att, t_chr
