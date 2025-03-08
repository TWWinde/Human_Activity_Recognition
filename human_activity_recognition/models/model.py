import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleRNN(nn.Module):
    def __init__(self, config, n_classes, window_length, rnn_units, rnn_num, rnn_dropout, dense_units, dense_num, dense_dropout, kernel_initializer, return_sequence=False):
        super(SimpleRNN, self).__init__()
        self.rnn_layers = nn.ModuleList()
        for _ in range(rnn_num):
            self.rnn_layers.append(nn.RNN(input_size=6, hidden_size=rnn_units, num_layers=1, batch_first=True, dropout=rnn_dropout))
            self.rnn_layers.append(nn.MaxPool1d(kernel_size=2))
            self.rnn_layers.append(nn.BatchNorm1d(num_features=rnn_units))
        self.rnn_layers.append(nn.RNN(input_size=6, hidden_size=rnn_units, num_layers=1, batch_first=True, dropout=rnn_dropout))
        
        self.dense_layers = nn.ModuleList()
        for _ in range(dense_num):
            self.dense_layers.append(nn.Linear(rnn_units, dense_units))
            self.dense_layers.append(nn.Dropout(dense_dropout))
        
        self.output_layer = nn.Linear(dense_units, n_classes)

    def forward(self, x):
        for layer in self.rnn_layers:
            x = layer(x)
        x = x[:, -1, :]  # Use the last output of the RNN
        for layer in self.dense_layers:
            x = layer(x)
        x = self.output_layer(x)
        return F.softmax(x, dim=1)


class LSTM(nn.Module):
    def __init__(self, input_size, n_classes, window_length, rnn_units, rnn_num, rnn_dropout, dense_units, dense_num, dense_dropout):
        super(LSTM, self).__init__()
        
        self.rnn_layers = nn.ModuleList()
        for i in range(rnn_num):
            self.rnn_layers.append(nn.LSTM(
                input_size=input_size if i == 0 else rnn_units,
                hidden_size=rnn_units,
                num_layers=1,
                batch_first=True,
                dropout=rnn_dropout
            ))
        
        self.dense_layers = nn.ModuleList()
        for _ in range(dense_num):
            self.dense_layers.append(nn.Linear(rnn_units, dense_units))
            self.dense_layers.append(nn.ReLU())
            self.dense_layers.append(nn.Dropout(dense_dropout))
        
        self.output_layer = nn.Linear(dense_units, n_classes)

    def forward(self, x):
        for lstm in self.rnn_layers:
            x, _ = lstm(x)
        
        x = x[:, -1, :]  # 取最后一个时间步的输出

        for layer in self.dense_layers:
            x = layer(x)
        
        x = self.output_layer(x)
        return F.softmax(x, dim=1)



class GRU(nn.Module):
    def __init__(self, n_classes, window_length, rnn_units, rnn_num, rnn_dropout, dense_units, dense_num, dense_dropout, kernel_initializer, return_sequence=False):
        super(GRU, self).__init__()
        self.rnn_layers = nn.ModuleList()
        for _ in range(rnn_num):
            self.rnn_layers.append(nn.GRU(input_size=6, hidden_size=rnn_units, num_layers=1, batch_first=True, dropout=rnn_dropout))
            self.rnn_layers.append(nn.MaxPool1d(kernel_size=2))
            self.rnn_layers.append(nn.BatchNorm1d(num_features=rnn_units))
        self.rnn_layers.append(nn.GRU(input_size=6, hidden_size=rnn_units, num_layers=1, batch_first=True, dropout=rnn_dropout))
        
        self.dense_layers = nn.ModuleList()
        for _ in range(dense_num):
            self.dense_layers.append(nn.Linear(rnn_units, dense_units))
            self.dense_layers.append(nn.Dropout(dense_dropout))
        
        self.output_layer = nn.Linear(dense_units, n_classes)

    def forward(self, x):
        for layer in self.rnn_layers:
            x = layer(x)
        x = x[:, -1, :]  # Use the last output of the GRU
        for layer in self.dense_layers:
            x = layer(x)
        x = self.output_layer(x)
        return F.softmax(x, dim=1)


if __name__ == '__main__':
    logging.info("PyTorch models defined.")