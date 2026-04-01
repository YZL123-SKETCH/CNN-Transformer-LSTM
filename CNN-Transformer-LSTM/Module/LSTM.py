import torch
import torch.nn as nn

class LSTM(nn.Module):
    """
    Three-Branch LSTM with Temporal Attention for Time Series Prediction
    Architecture: LSTM -> Temporal Attention -> Feature Fusion -> Output Prediction

    Three Input Branches:
        - rut_seq: Rutting deformation time series
        - temp_seq: Temperature time series
        - load_seq: Traffic load time series
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTM, self).__init__()

        # Hyperparameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout

        # Three-branch LSTM modules for different time series
        self.rut_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.temp_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.load_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Temporal attention layer for time-step weighting
        self.temporal_attention = nn.Linear(hidden_size, 1)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size * 3, output_size)

    def forward_with_attention(self, lstm_layer, x):
        """
        LSTM forward propagation with temporal attention pooling
        Args:
            lstm_layer: specific LSTM branch
            x: input sequence (batch, seq_len, input_size)
        Returns:
            attention-weighted feature vector
        """
        lstm_out, _ = lstm_layer(x)
        attn_weights = torch.softmax(self.temporal_attention(lstm_out), dim=1)
        weighted_feature = torch.sum(attn_weights * lstm_out, dim=1)
        return weighted_feature

    def forward(self, rut_seq, temp_seq, load_seq):
        # Extract attention-weighted features from three branches
        rut_feature = self.forward_with_attention(self.rut_lstm, rut_seq)
        temp_feature = self.forward_with_attention(self.temp_lstm, temp_seq)
        load_feature = self.forward_with_attention(self.load_lstm, load_seq)

        # Concatenate multi-branch features
        concatenated = torch.cat([rut_feature, temp_feature, load_feature], dim=1)

        # Final prediction
        output = self.fc(concatenated)
        return output