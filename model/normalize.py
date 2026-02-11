import torch.nn as nn
import torch

class Norms(nn.Module):
    def __init__(self,dim):
        super(Norms,self).__init__()
        self.layer_norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        x = self.layer_norm(x)
        return x
