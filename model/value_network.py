import torch
import torch.nn as nn
from .encoder_decoder import EncoderLayer
from .mlp import MLP

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, mlp_hidden_layers, max_seq_len, d_model, num_heads, d_ff, dropout,num_encoder_layers=1):
        super(ValueNetwork, self).__init__()
        
        # Define Transformer Encoder Attributes
        self._d_model = d_model
        self._num_heads = num_heads
        self._d_ff = d_ff
        self._dropout = dropout
        
        # Define linear transformation layers
        self._input_dim = input_dim
        self._max_seq_len = max_seq_len
        self.linear_transformation = nn.Linear(self._input_dim, 128)
        self.linear_transformation_1 = nn.Linear(128, self._d_model)
        self.linear_transformation_output = nn.Linear(self._max_seq_len, 8)
        self.linear_transformation_output_1 = nn.Linear(8, 1)
        
        # Define Transformer Encoder Layers
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(self._d_model, self._num_heads, self._d_ff, self._dropout) for _ in range(num_encoder_layers)]
        )
#         self.transformer_encoder_1 = EncoderLayer(self._d_model, self._num_heads, self._d_ff, self._dropout)
#         self.transformer_encoder_2 = EncoderLayer(self._d_model, self._num_heads, self._d_ff, self._dropout)
#         self.transformer_encoder_3 = EncoderLayer(self._d_model, self._num_heads, self._d_ff, self._dropout)
#         self.transformer_encoder_4 = EncoderLayer(self._d_model, self._num_heads, self._d_ff, self._dropout)
#         self.transformer_encoder_5 = EncoderLayer(self._d_model, self._num_heads, self._d_ff, self._dropout)
#         self.transformer_encoder_6 = EncoderLayer(self._d_model, self._num_heads, self._d_ff, self._dropout)
#         self.transformer_encoder_7 = EncoderLayer(self._d_model, self._num_heads, self._d_ff, self._dropout)
#         self.transformer_encoder_8 = EncoderLayer(self._d_model, self._num_heads, self._d_ff, self._dropout)
        
        # Define MLP Head for Regression
        self._mlp_hidden_layers = mlp_hidden_layers
        self._max_seq_len = max_seq_len
        self.MLP_layer_1 = MLP(self._d_model, self._mlp_hidden_layers, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def generate_mask(self, mask2d):
        return (mask2d != 0).unsqueeze(1).unsqueeze(2)
    
    def forward(self, x, mask=None):
        if mask is not None:
            mask = self.generate_mask(mask)
        x = self.relu(self.linear_transformation(x))
        x = self.tanh(self.linear_transformation_1(x))
#         x = self.transformer_encoder_1(x, mask)
#         x = self.transformer_encoder_2(x, mask)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        x = self.MLP_layer_1(x)
        x = x.squeeze(-1)
        x = self.relu(self.linear_transformation_output(x))
        x = self.linear_transformation_output_1(x)
        return x


class ValueNetworkV1(nn.Module):
    def __init__(self, input_dim,sku_emb, mlp_hidden_layers, max_seq_len, d_model, num_heads, d_ff, dropout,num_encoder_layers=1):
        super(ValueNetworkV1, self).__init__()
        
        # Define Transformer Encoder Attributes

        self._d_model = d_model
        self._num_heads = num_heads
        self._d_ff = d_ff
        self._dropout = dropout
        
        self.embeddings = nn.ModuleList([nn.Embedding(cat, size) for cat, size in sku_emb])
        sku_emb_dim = sum(e.embedding_dim for e in self.embeddings)
        

        # Define linear transformation layers
        self._input_dim = input_dim + sku_emb_dim 
        self._max_seq_len = max_seq_len
        self.linear_transformation = nn.Linear(self._input_dim, 128)
        self.linear_transformation_1 = nn.Linear(128, self._d_model)
        self.linear_transformation_output = nn.Linear(self._max_seq_len, 8)
        self.linear_transformation_output_1 = nn.Linear(8, 1)
        
        # Define Transformer Encoder Layers
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(self._d_model, self._num_heads, self._d_ff, self._dropout) for _ in range(num_encoder_layers)]
        )
#         self.transformer_encoder_1 = EncoderLayer(self._d_model, self._num_heads, self._d_ff, self._dropout)
#         self.transformer_encoder_2 = EncoderLayer(self._d_model, self._num_heads, self._d_ff, self._dropout)
#         self.transformer_encoder_3 = EncoderLayer(self._d_model, self._num_heads, self._d_ff, self._dropout)
#         self.transformer_encoder_4 = EncoderLayer(self._d_model, self._num_heads, self._d_ff, self._dropout)
#         self.transformer_encoder_5 = EncoderLayer(self._d_model, self._num_heads, self._d_ff, self._dropout)
#         self.transformer_encoder_6 = EncoderLayer(self._d_model, self._num_heads, self._d_ff, self._dropout)
#         self.transformer_encoder_7 = EncoderLayer(self._d_model, self._num_heads, self._d_ff, self._dropout)
#         self.transformer_encoder_8 = EncoderLayer(self._d_model, self._num_heads, self._d_ff, self._dropout)
        
        # Define MLP Head for Regression
        self._mlp_hidden_layers = mlp_hidden_layers
        self._max_seq_len = max_seq_len
        self.MLP_layer_1 = MLP(self._d_model, self._mlp_hidden_layers, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def generate_mask(self, mask2d):
        return (mask2d != 0).unsqueeze(1).unsqueeze(2)
    
    def forward(self, order_features, cat_features, mask=None):
        x = [e(cat_features[:, :, i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat(x, dim=2)
        x = torch.cat([x, order_features], dim=2)

        if mask is not None:
            mask = self.generate_mask(mask)
        x = self.relu(self.linear_transformation(x))
        x = self.tanh(self.linear_transformation_1(x))
#         x = self.transformer_encoder_1(x, mask)
#         x = self.transformer_encoder_2(x, mask)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        x = self.MLP_layer_1(x)
        x = x.squeeze(-1)
        x = self.relu(self.linear_transformation_output(x))
        x = self.linear_transformation_output_1(x)
        return x
