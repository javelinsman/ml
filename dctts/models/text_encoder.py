import torch
import torch.nn as nn
import torch.nn.functional as F
from util.g2p import phoneme_types
from util.layers import CustomConv1d, HighwayConv1d
from .config import DIMENSIONS as D

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        d = D['latent']
        e = D['embedding']
        
        self.embedding = nn.Embedding(len(phoneme_types), e)
        self.layers = nn.ModuleList([
            
            CustomConv1d(e, 2 * d, 1, 1), nn.ReLU(),
            CustomConv1d(2 * d, 2 * d, 1, 1),

            HighwayConv1d(2 * d, 3, 1),
            HighwayConv1d(2 * d, 3, 3),
            HighwayConv1d(2 * d, 3, 9),
            HighwayConv1d(2 * d, 3, 27),
            HighwayConv1d(2 * d, 3, 1),
            HighwayConv1d(2 * d, 3, 3),
            HighwayConv1d(2 * d, 3, 9),
            HighwayConv1d(2 * d, 3, 27),
            HighwayConv1d(2 * d, 3, 1),
            HighwayConv1d(2 * d, 3, 1),
            HighwayConv1d(2 * d, 1, 1),
            HighwayConv1d(2 * d, 1, 1),
        ])
    
    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        for layer in self.layers[:1]:
            x = layer(x)
        return x[:,:D['latent'],:], x[:,D['latent']:,:]
