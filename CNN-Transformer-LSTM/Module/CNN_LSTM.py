import torch
import torch.nn as nn


class CNN_LSTM(nn.Module):
    """
    CNN-LSTM Hybrid Model for Time Series Prediction
    Architecture: 2-layer 1D CNN + Three-branch LSTM + Fully Connected Layer

    Inputs:
        rut_seq:  Time series sequence of rutting data
        temp_seq: Time series sequence of temperature data
        load_seq: Time series sequence of load data

    Paper-style structure:
        Input → Conv1d → ReLU → Conv1d → ReLU → LSTM → Last Time-step → Concatenate → Output
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, cnn_channels=64):
        super(CNN_LSTM, self).__init__()

        # Hyperparameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.cnn_channels = cnn_channels

        # 1D CNN Modules
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=cnn_channels,
            kernel_size=3,
            padding=1
        )
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(
            in_channels=cnn_channels,
            out_channels=cnn_channels,
            kernel_size=3,
            padding=1
        )
        self.relu2 = nn.ReLU()

        # Three-branch LSTM for different time series
        self.rut_lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.temp_lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.load_lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size * 3, output_size)

    def extract_cnn_features(self, x):
        """
        2-layer 1D CNN feature extraction
        Input shape:  (batch_size, seq_len, input_dim)
        Output shape: (batch_size, seq_len, cnn_channels)
        """
        # Convert to (batch, channels, seq_len) for Conv1d
        x = x.transpose(1, 2)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        # Convert back to (batch, seq_len, channels)
        x = x.transpose(1, 2)
        return x

    def forward(self, rut_seq, temp_seq, load_seq):
        # CNN feature extraction
        rut_feat = self.extract_cnn_features(rut_seq)
        temp_feat = self.extract_cnn_features(temp_seq)
        load_feat = self.extract_cnn_features(load_seq)

        # LSTM forward
        rut_out, _ = self.rut_lstm(rut_feat)
        temp_out, _ = self.temp_lstm(temp_feat)
        load_out, _ = self.load_lstm(load_feat)

        # Get the last time-step output
        rut_last = rut_out[:, -1, :]
        temp_last = temp_out[:, -1, :]
        load_last = load_out[:, -1, :]

        # Concatenate three branches
        concat_features = torch.cat([rut_last, temp_last, load_last], dim=1)
        output = self.fc(concat_features)

        return output
