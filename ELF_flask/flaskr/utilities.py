from numpy import split
from numpy import array
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

class Dataset:
    def __init__(self, data):
        self.data = data

    def split_dataset(self):
        # split into standard weeks
        train, test = self.data[0: 294], self.data[294:364]
        # restructure into windows of weekly data
        train = array(split(train, len(train)/7))
        test = array(split(test, len(test)/7))
        return train, test

    def to_supervised(self, train, n_input, n_out=7):
        # flatten data
        self.data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
        X, y = list(), list()
        in_start = 0
        # step over the entire history one time step at a time
        for _ in range(len(self.data)):
            # define the end of the input sequence
            in_end = in_start + n_input
            out_end = in_end + n_out
            # ensure we have enough data for this instance
            if out_end <= len(self.data):
                X.append(self.data[in_start:in_end, :])
                y.append(self.data[in_end:out_end, 0])
            # move along one time step
            in_start += 1
        return array(X), array(y)
        
class cnnlstm_Model():
    def train_cnnlstm_model(self, train_x, train_y):
        # define parameters
        verbose, epochs, batch_size = 0, 20, 16
        n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
        # reshape output into [samples, timesteps, features]
        train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
        # define model
        self.model = Sequential()
        self.model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
        self.model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(RepeatVector(n_outputs))
        self.model.add(LSTM(200, activation='relu', return_sequences=True))
        self.model.add(TimeDistributed(Dense(100, activation='relu')))
        self.model.add(TimeDistributed(Dense(1)))
        self.model.compile(loss='mse', optimizer='adam')
        # fit network
        self.model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def forecast(self, history, n_input):
        # flatten data
        data = array(history)
        data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
        # retrieve last observations for input data
        input_x = data[-n_input:, ]
        # reshape into [1, n_input, 1]
        input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
        # forecast the next week
        yhat = self.model.predict(input_x, verbose=0)
        # we only want the vector forecast
        yhat = yhat[0]
        return yhat

# data = read_csv('./data/daily_dataset.csv', header=0, infer_datetime_format=True, parse_dates=['Date & Time'], index_col=['Date & Time'])
# dataset = Dataset(data)

# train, test = dataset.split_dataset()

# train_x, train_y = dataset.to_supervised(train, n_input=14)

# model = cnnlstm_Model()

# model.train_cnnlstm_model(train_x, train_y)

# yhat = model.forecast(test, n_input=14)

# print(yhat)