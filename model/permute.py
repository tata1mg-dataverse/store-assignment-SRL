import torch.nn as nn
import torch

class Permute(nn.Module):
    def __init__(self, shape):
        super(Permute,self).__init__()
        self._shape = shape
        
    def forward(self, x):
        x = torch.permute(x,self._shape)
        return x
