import torch.nn as nn

class DeepLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_lstm_layers, dropout):
        super(DeepLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_lstm_layers, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        output = self.dropout(output)
        return output, hidden
