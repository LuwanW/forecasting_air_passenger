from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
class LSTM_model:
    def __init__(self, data, train, test):

        # x_train, y_train = self.prepare_data(train,12)
        # x_test, y_test = self.prepare_data(test,12)

        features= list(set(list(train.columns)) - {'val'})
        train = train.drop(features, axis=1)
        test_df = test.drop(features, axis=1)
        test_list = test['val'].tolist()

        # data scaling
        scaler = MinMaxScaler()
        scaler.fit(train)
        train = scaler.transform(train)
        # test = scaler.transform(test_df)


        # train model
        n_input = 24
        n_features = 1
        generator = TimeseriesGenerator(train, train, length=n_input, batch_size=6)
        model = Sequential()
        model.add(LSTM(200, activation='relu', input_shape=(n_input, n_features)))
        model.add(Dropout(0.15))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit_generator(generator, epochs=90, verbose=0)
        model.summary()

        # prediction
        forecasting = []
        batch = train[-n_input:].reshape((1, n_input, n_features))
        for i in range(n_input):
            forecasting.append(model.predict(batch)[0])
            batch = np.append(batch[:, 1:, :], [[forecasting[i]]], axis=1)

        forecasting_df = pd.DataFrame(scaler.inverse_transform(forecasting),index=data[-n_input:].index, columns=['forecast'])
        forecasting_df['test'] = test_list

        forecasting_df.plot()
        plt.title("LSTM")

        plt.show()

        # print test result
        print(forecasting_df)

        # print MSE test result
        print('MSE', mean_squared_error(forecasting_df['test'], forecasting_df['forecast']))

    # transform each data to n features with each feature's n adjacent datapoint
    def prepare_data(self, data, n):
        x=[]
        y=[]
        for i in range(len(data)-n-1):
            x_item = data[i:i+n]
            x.append(x_item)
            y.append(data[i+n+1])
        return np.asarray(x),np.asarray(y)