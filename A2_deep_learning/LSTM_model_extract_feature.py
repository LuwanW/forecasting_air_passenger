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
from collections import deque


class LSTM_model:
    def __init__(self, data, train, test):
        n_input = 24
        # fix random seed for reproducibility
        np.random.seed(0)
        # data scaling
        scalar = MinMaxScaler(feature_range=(0, 1))
        np_arr_train = np.asarray(train['value'])
        np_arr_train = np_arr_train.reshape(-1, 1)
        train['value'] = scalar.fit_transform(np_arr_train)


        x_train, y_train = self.prepare_data(train, n_input, scalar)
        x_test, y_test = self.prepare_data(test, n_input, scalar)
        train_list = train['value'].tolist()
        test_list = test['value'].tolist()

        # reshape
        x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))

        # train model
        n_extract_features = 3
        n_all_features = n_input + n_extract_features

        model = Sequential()
        model.add(LSTM(200, activation='relu', input_shape=(1, n_all_features)))
        model.add(Dropout(0.15))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(x_train, y_train, epochs=90, verbose=0)
        model.summary()

        # prediction
        n_prediction = 24
        predict_hist = train_list[-n_input:]

        for i in range(n_prediction):
            test_predict = predict_hist[-n_input:]

            # prepare extracted feature
            extracted_feature = self.add_extract_feature(test,scalar, i)
            add_feature = test_predict + extracted_feature

            print(add_feature)
            add_feature = np.asarray(add_feature)
            add_feature = np.reshape(add_feature, (1, 1, n_all_features))
            prediction = model.predict(add_feature)
            predict_hist.append(prediction.reshape(-1).tolist()[0])

        # scale back
        predict_hist = np.asarray(predict_hist)
        prediction_list = predict_hist[-n_prediction:].reshape(-1, 1)
        scaled_predict_hist = scalar.inverse_transform(prediction_list)
        forecasting_df = pd.DataFrame(scaled_predict_hist, index=test[-n_prediction:].index,
                                      columns=['forecast'])
        forecasting_df['test'] = test_list

        forecasting_df.plot()
        plt.title("LSTM")

        plt.show()

        # print test result
        print(forecasting_df)

        # print MSE test result
        print('MSE', mean_squared_error(forecasting_df['test'], forecasting_df['forecast']))

    # transform each data to n features with each feature's n adjacent datapoint
    def prepare_data(self, train, n, scalar):
        data = train['value'].tolist()
        x = []
        y = []
        for i in range(len(data) - n - 1):
            extracted_feature = self.add_extract_feature(train,scalar, i)
            this_data = data[i:i + n] + extracted_feature
            x.append(this_data)
            y.append(data[i + n])
        return np.asarray(x), np.asarray(y)

    def add_extract_feature(self, data, scalar, i):
        extracted_feature = []
        year = data['Year'].tolist()
        month = data['Month'].tolist()
        id = data['id'].tolist()
        extracted_feature.append(year[i])
        extracted_feature.append(month[i])
        extracted_feature.append(id[i])
        extracted_feature = np.asarray(extracted_feature)
        extracted_feature = extracted_feature.reshape(-1, 1)
        extracted_feature = scalar.transform(extracted_feature).reshape(-1).tolist()
        return extracted_feature
