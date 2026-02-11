import torch
import torch.nn as nn

class Permute(nn.Module):
    def __init__(self, shape):
        super(Permute, self).__init__()
        self._shape = shape
    
    def forward(self, x):
        x = torch.permute(x, self._shape)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list, output_dim: int, dropout: float = 0.0):
        super(MLP, self).__init__()
        # Define input and output layers
        self.input_layer = nn.Linear(input_dim, hidden_layers[0])
        self.output_layer = nn.Linear(hidden_layers[-1], output_dim)
        self.relu = nn.ReLU()
        self.permute_layer = Permute((0, 2, 1))
        self.dropout = nn.Dropout(dropout)
        
        # Define hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_layers[layer_idx-1], hidden_layers[layer_idx]),
                self.permute_layer,
                nn.BatchNorm1d(hidden_layers[layer_idx]),
                self.permute_layer,
                nn.ReLU()
            )
            for layer_idx in range(1, len(hidden_layers))
        ])
        
    def forward(self, x):
        x = self.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

class MLP_V2(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list, output_dim: int, dropout: float = 0.0):
        super(MLP, self).__init__()
        # Define input and output layers
        self.input_layer = nn.Linear(input_dim, hidden_layers[0])
        self.output_layer = nn.Linear(hidden_layers[-1], output_dim)
        self.relu = nn.ReLU()
        self.permute_layer = Permute((0, 2, 1))
        self.dropout = nn.Dropout(dropout)
        
        # Define hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_layers[layer_idx-1], hidden_layers[layer_idx]),
                self.permute_layer,
                nn.BatchNorm1d(hidden_layers[layer_idx]),
                self.permute_layer,
                nn.ReLU(),
                self.dropout
            )
            for layer_idx in range(1, len(hidden_layers))
        ])
        
    def forward(self, x):
        x = self.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x
