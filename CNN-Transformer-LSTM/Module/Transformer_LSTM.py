import torch
import torch.nn as nn

class Transformer_LSTM(nn.Module):
    """
    Transformer + LSTM Hybrid Model for Time Series Prediction
    Architecture: Multi-head Self-Attention → LSTM → Temporal Attention Fusion
    Inputs: rut_seq, temp_seq, load_seq (3 time-series branches)
    Output: rutting prediction
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, mha_heads=1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout

        # ---------------------- Multi-Head Self-Attention ----------------------
        self.mha = nn.MultiheadAttention(
            embed_dim=input_size,
            num_heads=mha_heads,
            batch_first=False,
            dropout=dropout
        )

        # ---------------------- Three-branch LSTM ----------------------
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

        # ---------------------- Temporal Attention Layer ----------------------
        self.temporal_attention = nn.Linear(hidden_size, 1)

        # ---------------------- Fully Connected Output ----------------------
        self.fc = nn.Linear(hidden_size * 3, output_size)

    def forward_multihead_attention(self, seq):
        """Multi-head self-attention with dimension transformation"""
        seq = seq.transpose(0, 1)  # (batch, seq_len, dim) → (seq_len, batch, dim)
        attn_out, _ = self.mha(seq, seq, seq)
        attn_out = attn_out.transpose(0, 1)  # back to (batch, seq_len, dim)
        return attn_out

    def forward_temporal_attention(self, lstm_out):
        """Standard temporal attention: weight & sum over time steps"""
        attn_weights = torch.softmax(self.temporal_attention(lstm_out), dim=1)
        weighted_sum = torch.sum(attn_weights * lstm_out, dim=1)
        return weighted_sum

    def forward(self, rut_seq, temp_seq, load_seq):
        # 1. Multi-head self-attention feature extraction
        rut_feat = self.forward_multihead_attention(rut_seq)
        temp_feat = self.forward_multihead_attention(temp_seq)
        load_feat = self.forward_multihead_attention(load_seq)

        # 2. LSTM temporal feature learning
        rut_out, _ = self.rut_lstm(rut_feat)
        temp_out, _ = self.temp_lstm(temp_feat)
        load_out, _ = self.load_lstm(load_feat)

        # 3. Temporal attention pooling
        rut_rep = self.forward_temporal_attention(rut_out)
        temp_rep = self.forward_temporal_attention(temp_out)
        load_rep = self.forward_temporal_attention(load_out)

        # 4. Concatenate three branches & prediction
        concat_feat = torch.cat([rut_rep, temp_rep, load_rep], dim=1)
        output = self.fc(concat_feat)

        return output