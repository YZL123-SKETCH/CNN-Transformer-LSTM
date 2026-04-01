import torch
import torch.nn as nn


class CNN_Transformer_LSTM(nn.Module):
    """
    CNN-Transformer-LSTM Hybrid Model for Time Series Prediction
    Architecture: 1D CNN -> Multi-Head Self-Attention -> LSTM -> Temporal Attention -> Fusion -> Output

    Three input branches: rutting, temperature, load
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, mha_heads=1):
        super(CNN_Transformer_LSTM, self).__init__()

        # Hyperparameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.cnn_out_dim = 64

        # -------------------------- 1D CNN Feature Extractor --------------------------
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, self.cnn_out_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.cnn_out_dim, self.cnn_out_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # -------------------------- Multi-Head Self-Attention --------------------------
        self.mha = nn.MultiheadAttention(
            embed_dim=self.cnn_out_dim,
            num_heads=mha_heads,
            batch_first=False,
            dropout=dropout
        )

        # -------------------------- Three-branch LSTM Modules --------------------------
        self.rut_lstm = nn.LSTM(
            input_size=self.cnn_out_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.temp_lstm = nn.LSTM(
            input_size=self.cnn_out_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.load_lstm = nn.LSTM(
            input_size=self.cnn_out_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # -------------------------- Temporal Attention Layer --------------------------
        self.temporal_attention = nn.Linear(hidden_size, 1)

        # -------------------------- Fully Connected Output --------------------------
        self.fc = nn.Linear(hidden_size * 3, output_size)

    def extract_cnn(self, x):
        """Extract local temporal features using 1D CNN"""
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        x = self.cnn(x)
        return x.transpose(1, 2)  # (B, T, 64)

    def apply_multihead_attention(self, x):
        """Multi-head self-attention for global dependency modeling"""
        x = x.transpose(0, 1)
        attn_output, _ = self.mha(x, x, x)
        return attn_output.transpose(0, 1)

    def apply_temporal_attention(self, lstm_out):
        """Temporal attention: weight and sum over time steps"""
        attn_weights = torch.softmax(self.temporal_attention(lstm_out), dim=1)
        return torch.sum(attn_weights * lstm_out, dim=1)

    def forward(self, rut_seq, temp_seq, load_seq):
        # Step 1: CNN local feature extraction
        rut_feat = self.extract_cnn(rut_seq)
        temp_feat = self.extract_cnn(temp_seq)
        load_feat = self.extract_cnn(load_seq)

        # Step 2: Multi-head attention for global context
        rut_feat = self.apply_multihead_attention(rut_feat)
        temp_feat = self.apply_multihead_attention(temp_feat)
        load_feat = self.apply_multihead_attention(load_feat)

        # Step 3: LSTM temporal learning
        rut_out, _ = self.rut_lstm(rut_feat)
        temp_out, _ = self.temp_lstm(temp_feat)
        load_out, _ = self.load_lstm(load_feat)

        # Step 4: Temporal attention pooling
        rut_rep = self.apply_temporal_attention(rut_out)
        temp_rep = self.apply_temporal_attention(temp_out)
        load_rep = self.apply_temporal_attention(load_out)

        # Step 5: Feature fusion and prediction
        concat_features = torch.cat([rut_rep, temp_rep, load_rep], dim=1)
        output = self.fc(concat_features)

        return output