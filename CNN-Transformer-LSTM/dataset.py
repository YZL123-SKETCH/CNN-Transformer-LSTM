import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataLoader:
    """
    Data Loader for Multi-Variate Time Series Prediction
    Automatically loads, normalizes, and generates sequences for rutting, temperature, and load data
    """
    def __init__(self, filename, split_ratio, cols):
        # Read data
        self.df = pd.read_csv(filename)
        self.cols = cols
        self.split_ratio = split_ratio
        self.split_idx = int(len(self.df) * split_ratio)

        # Extract raw data
        self.data_train = self.df[cols].values[:self.split_idx]
        self.data_test = self.df[cols].values[self.split_idx:]

        # Min-Max Normalization
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        combined_data = np.concatenate([self.data_train, self.data_test], axis=0)
        normalized_data = self.scaler.fit_transform(combined_data)

        # Assign normalized data back
        self.data_train = normalized_data[:self.split_idx]
        self.data_test = normalized_data[self.split_idx:]

        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)

    def get_train_data(self, seq_len):
        """Generate full training sequences"""
        train_rut, train_temp, train_load = [], [], []
        for i in range(self.len_train - seq_len):
            x_rut, x_temp, x_load = self._extract_sequence(i, seq_len)
            train_rut.append(x_rut)
            train_temp.append(x_temp)
            train_load.append(x_load)
        return np.array(train_rut), np.array(train_temp), np.array(train_load)

    def get_test_data(self, seq_len):
        """Generate full test sequences"""
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)
        test_rut = data_windows[:, :, 0]
        test_temp = data_windows[:, :, 1]
        test_load = data_windows[:, :, 2]
        return test_rut, test_temp, test_load

    def generate_train_batch(self, seq_len, batch_size):
        """Yield training batches"""
        i = 0
        while i < (self.len_train - seq_len):
            x_batch_rut, x_batch_temp, x_batch_load, y_batch = [], [], [], []
            for _ in range(batch_size):
                if i >= self.len_train - seq_len:
                    yield (np.array(x_batch_rut),
                           np.array(x_batch_temp),
                           np.array(x_batch_load),
                           np.array(y_batch))
                    i = 0
                x_rut, x_temp, x_load = self._extract_sequence(i, seq_len)
                x_batch_rut.append(x_rut)
                x_batch_temp.append(x_temp)
                x_batch_load.append(x_load)
                y_batch.append(x_rut[-1])
                i += 1
            yield (np.array(x_batch_rut),
                   np.array(x_batch_temp),
                   np.array(x_batch_load),
                   np.array(y_batch))

    def generate_test_batch(self, seq_len, batch_size):
        """Yield test batches"""
        i = 0
        while i < (self.len_test - seq_len):
            x_batch_rut, x_batch_temp, x_batch_load, y_batch = [], [], [], []
            for _ in range(batch_size):
                if i >= self.len_test - seq_len:
                    yield (np.array(x_batch_rut),
                           np.array(x_batch_temp),
                           np.array(x_batch_load),
                           np.array(y_batch))
                    return
                window = self.data_test[i:i+seq_len]
                x_batch_rut.append(window[:, 0])
                x_batch_temp.append(window[:, 1])
                x_batch_load.append(window[:, 2])
                y_batch.append(window[-1, 0])
                i += 1
            yield (np.array(x_batch_rut),
                   np.array(x_batch_temp),
                   np.array(x_batch_load),
                   np.array(y_batch))

    def _extract_sequence(self, i, seq_len):
        """Extract single sequence from training data"""
        window = self.data_train[i:i+seq_len]
        x_rut = window[:, 0]
        x_temp = window[:, 1]
        x_load = window[:, 2]
        return x_rut, x_temp, x_load


