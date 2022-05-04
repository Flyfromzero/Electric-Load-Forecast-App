import numpy as np
import pandas as pd
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.autograd import Variable

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}" " is available.")

class Dataset:
    def __init__(self, PATH):
        self.PATH = PATH

    # read the dataset
    def read_dataset(self):
        dataset = read_csv(self.PATH, header=0, infer_datetime_format=True, parse_dates=['Date & Time'], index_col=['Date & Time'])
        return dataset

    # split a univariate dataset into train/test sets
    def split_dataset(self, data):
        # split into standard weeks
        train, test = data[0: 294], data[294:364]
        return train, test
    
    # normalize the features and target
    def normalize_dataset(self, data, target, features):
        # split the data into features and target
        features_df = data.loc[:,features]
        target_df = data.loc[:,target]
        # scale features and target repectively
        features_scaler = MinMaxScaler()
        features_scaler.fit(features_df)
        target_scaler = MinMaxScaler()
        target_scaler.fit(target_df)
        features_df_scaled = pd.DataFrame(
            features_scaler.transform(features_df),columns=features,index=data.index)
        target_df_scaled = pd.DataFrame(
            target_scaler.transform(target_df),columns=target,index=data.index)
        # merge scaled features and target
        df_scaled = pd.concat([target_df_scaled,features_df_scaled],axis=1)
        return df_scaled, target_scaler # we need the target_scaler to inverse scale the output of model

    # weekly split the dataset
    def weekly_split(self, data):
        data = array(split(data.values, len(data)/7))
        return data

    """
    Given some number of prior days of total daily power consumption,
    predict the next standard week of daily power consumption
    in this case, the input size is 7
    """
    # convert history into inputs and outputs
    def to_supervised(self, data, n_input, n_output):
        # flatten data
        data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
        X, y = list(), list()
        in_start = 0
        # step over the entire history one time step at a time
        for _ in range(len(data)):
            # define the end of the input sequence
            in_end = in_start + n_input
            out_end = in_end + n_output
            # ensure we have enough data for this instance
            if out_end <= len(data):
                X.append(data[in_start:in_end, :])
                y.append(data[in_end:out_end, 0])
            # move along one time step
            in_start += 1
        return array(X), array(y)
    
    # to torch tensor
    def to_tensor(self, data):
        data = Variable(torch.Tensor(data))
        return data

# define cnn-lstm model
class CNNLSTMModel(nn.Module):
    def __init__(self, lstm_hidden_size, lstm_num_layers, in_seq_length, out_seq_length):
        super(CNNLSTMModel, self).__init__()
        # define hyper-parameters of model
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length
        # define convolution layer
        self.conv1d_1 = nn.Conv1d(in_channels=self.in_seq_length, out_channels=64, kernel_size=3)
        self.conv1d_2 = nn.Conv1d(in_channels=64, out_channels=16, kernel_size=3)
        # define pooling layer
        self.maxpool1d = nn.MaxPool1d(kernel_size=2)
        # define relu
        self.relu = nn.ReLU(inplace=True)
        # define flatten layer
        self.flatten= nn.Flatten()
        # define lstm layer
        self.lstm = nn.LSTM(input_size=2, hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_num_layers, batch_first=True)
        # define fully-connect layer
        self.fc1 = nn.Linear(in_features=3200, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=self.out_seq_length)
        
    def forward(self, input):
        batch_size, _, _ = input.size()
        # initial h_0, c_0
        h_0 = Variable(torch.zeros(
            self.lstm_num_layers, batch_size, self.lstm_hidden_size))
        c_0 = Variable(torch.zeros(
            self.lstm_num_layers, batch_size, self.lstm_hidden_size))
        # compute the output
        x = self.conv1d_1(input)
        x = self.relu(x)
        x = self.conv1d_2(x)
        x = self.relu(x)
        x = self.maxpool1d(x)       
        x, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        x = self.flatten(x)
        x = self.fc1(x)
        output = self.fc2(x)
        return output 

class ForecastLoad:
    def __init__(self, model_path, data, target_scaler):
        self.model_path = model_path
        self.data = data
        self.target_scaler = target_scaler

    def forecast(self):
        cnnlstm = CNNLSTMModel(lstm_hidden_size=200, lstm_num_layers=1, 
                                in_seq_length=7, out_seq_length=7)
        cnnlstm.load_state_dict(torch.load(self.model_path))
        cnnlstm.eval()

        input = self.data[-1:, ]
        predicted = cnnlstm(input)
        predicted = predicted.data.numpy()
        predicted = self.target_scaler.inverse_transform(predicted)
        print("predictions:")
        print(predicted)
        return predicted
