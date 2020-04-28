import torch
import torch.nn as nn
from torch.nn.utils import weight_norm as norm
import torch.nn.functional as F

class CustomConv1d(nn.Module):
    def __init__(self, d1, d2, kernel_size, dilation, padding_mode='same'):
        super().__init__()
        self.d1, self.d2 = d1, d2
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding_mode = padding_mode
        
        effective_kernel_size = dilation * (kernel_size - 1) + 1
        self.offset = effective_kernel_size - 1

        if self.padding_mode == 'same':
            self.conv = norm(nn.Conv1d(d1, d2, kernel_size, dilation=dilation, padding=1 + (self.offset - 1) // 2))
        elif self.padding_mode == 'causal':
            self.conv = norm(nn.Conv1d(d1, d2, kernel_size, dilation=dilation, padding=self.offset))
        # self.conv_bn = nn.BatchNorm1d(d2)
    
    def forward(self, x):
        return self.forward_raw(x)
        # return self.conv_bn(self.forward_raw(x))
    
    def forward_raw(self, x):
        if self.padding_mode == 'same':
            if self.offset % 2 == 0:
                return self.conv(x)
            else:
                return self.conv(x)[:,:,:-1]
            
        elif self.padding_mode == 'causal':
            if self.offset == 0:
                return self.conv(x)
            else:
                return self.conv(x)[:,:,:-self.offset]

class HighwayConv1d(nn.Module):
    def __init__(self, d, kernel_size=1, dilation=1, padding_mode='same'):
        super().__init__()
        self.d = d 
        self.conv = CustomConv1d(
            d, 2 * d, kernel_size, dilation=dilation, padding_mode=padding_mode
        )
        # self.conv_h1 = CustomConv1d(
        #     d, d, kernel_size, dilation=dilation, padding_mode=padding_mode
        # )
        # self.conv_h2 = CustomConv1d(
        #     d, d, kernel_size, dilation=dilation, padding_mode=padding_mode
        # )

    def forward(self, x):
        # sigH1 = torch.sigmoid(self.conv_h1(x))
        # H2 = self.conv_h2(x)
        H = self.conv(x)
        H1, H2 = H[:,:self.d,:], H[:,self.d:,:]
        sigH1 = torch.sigmoid(H1)
        return sigH1 * F.relu(H2) + (1 - sigH1) * x