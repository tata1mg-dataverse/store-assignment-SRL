import torch
import torch.nn as nn
from .encoder_decoder import EncoderLayer
from .mlp import MLP

class AllocationModel(nn.Module):
    def __init__(
        self, num_scalar_features,
        mlp_output_dim, mlp_hidden_layers,
        transformer_output_dim, num_attention_heads, transformer_ff_dim,
        allocation_output_dim, dropout, num_encoder_layers=1):
        super(AllocationModel, self).__init__()
        
        # Define Model Parameters
        self._dropout = dropout
        
        # Define MLP parameters
        self._mlp_input_dim = num_scalar_features
        self._mlp_output_dim = mlp_output_dim
        self._mlp_hidden_layers = mlp_hidden_layers
        self._mlp_allocation_output_dim = allocation_output_dim
        
        # Define Transformer Encoder parameters
        self._transformer_input_dim = num_scalar_features
        self._d_model = transformer_output_dim
        self._num_heads = num_attention_heads
        self._d_ff = transformer_ff_dim
        
        # Define Allocation Model Layers
        self.linear_transform_sku = nn.Linear(self._transformer_input_dim, 128)
        self.linear_transform_sku_1 = nn.Linear(128, self._d_model)
        self.linear_transform_enc = nn.Linear(self._d_model + self._mlp_output_dim, self._d_model)
        self.MLP_layer = MLP(self._d_model, self._mlp_hidden_layers, self._mlp_allocation_output_dim, dropout=self._dropout)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(self._d_model, self._num_heads, self._d_ff, self._dropout) for _ in range(num_encoder_layers)]
        )
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def generate_mask(self, mask2d):
        return (mask2d != 0).unsqueeze(1).unsqueeze(2)
    
    def forward(self, order_features, mask=None):
        if mask is not None:
            mask = self.generate_mask(mask)
        order_features = self.relu(self.linear_transform_sku(order_features))
        order_features = self.tanh(self.linear_transform_sku_1(order_features))
        for layer in self.encoder_layers:
            order_features = layer(order_features, mask)
        allocation_output = self.MLP_layer(order_features)
        return allocation_output

