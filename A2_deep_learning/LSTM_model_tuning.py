from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from collections import deque
import keras_tuner as kt

class LSTM_model:
    def __init__(self, data, train, test, n_window):
        n_prediction = 24

        # data scaling train
        scalar = MinMaxScaler(feature_range=(0, 1))
        data_list= np.asarray(data['#Passengers'])
        data_list = data_list.reshape(-1, 1)
        data_list_scale = scalar.fit(data_list)

        np_arr_train = np.asarray(train['value'])
        np_arr_train = np_arr_train.reshape(-1, 1)
        train['value'] = scalar.transform(np_arr_train)

        # data scaling test
        np_arr_test = np.asarray(test['value'])
        np_arr_test = np_arr_test.reshape(-1, 1)
        test['value'] = scalar.transform(np_arr_test)

        x_train, y_train = self.prepare_train_data(train, n_window, scalar)
        x_test, y_test = self.prepare_test_data(train, test, n_window, scalar, n_prediction)



        # reshape
        x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
        x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))


        # train model
        n_extract_features = 3
        n_all_features = n_window + n_extract_features


        # tune model 0 for original model, 1 for keras tuner, 2 for turned model
        tune = 2

        if tune == 0:
            # original model
            hidden_layer = 3
            n_neurons = 64
            drop_out = 0.3
            model = Sequential()
            model.add(LSTM(n_neurons, activation='relu', input_shape=(1, n_all_features)))
            for i in range(hidden_layer):
                model.add(Dense(n_neurons))
            model.add(Dropout(drop_out))

            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae', 'mape'])
            history = model.fit(x_train, y_train, validation_split=0.33, batch_size=10, epochs=90, verbose=0)
            model.summary()


        elif tune == 1:
            # prepare for model tuning
            def build_model(hp):
                # # tune number of neurons
                model = Sequential()
                model.add(LSTM(hp.Int('input_unit',min_value=32,max_value=128,step=32), activation='relu', input_shape=(1, n_all_features), return_sequences=True))
                # tune hidden layer
                for i in range(hp.Int('n_layers', 0, 3)):
                    model.add(Dense(units=hp.Int('num_of_neurons', min_value=32, max_value=128, step=32),
                    activation = 'relu'))

                # tune dropout rate
                model.add(Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.3,step=0.05)))
                # tune activation function
                model.add(Dense(1,
                                activation=hp.Choice('dense_activation', values=['relu', 'sigmoid'], default='relu')))
                model.compile(loss='mse',  metrics=['mse', 'mae', 'mape'] ,optimizer='adam')
                return model

            tuner = kt.Hyperband(
                    hypermodel=build_model,
                    objective='val_mse',
                    factor=3,
                    max_epochs=100,
                    project_name='hyperband_tuner'
                    )

            stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

            tuner.search(x_train,
                         y_train,
                         epochs=20,
                         validation_split=0.33,
                            callbacks = [stop_early]
                         )
            tuner.results_summary()

            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            model = tuner.hypermodel.build(best_hps)
            history = model.fit(x_train, y_train, validation_split=0.33, batch_size=10, epochs=90, verbose=0)

            # = tuner.get_best_models(num_models=1)[0]
            model.summary()

            # # find minimum MAPE
            # val_mape = history.history['val_mse']
            # best_epoch = val_mape.index(min(val_mape)) + 1
            # print('Best epoch: %d' % (best_epoch,))
            #
            # model = tuner.hypermodel.build(best_hps)




            # Retrain with best epoch
            # history = model.fit(x_train, y_train,epochs=best_epoch, validation_split=0.2)

        elif tune == 2:
            # original model
            hidden_layer = 0
            n_neurons = 32
            drop_out = 0.05
            model = Sequential()
            model.add(LSTM(n_neurons, activation='relu', input_shape=(1, n_all_features)))
            for i in range(hidden_layer):
                model.add(Dense(n_neurons))
            model.add(Dropout(drop_out))

            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae', 'mape'])
            history = model.fit(x_train, y_train, validation_split=0.33, batch_size=10, epochs=90, verbose=0)
            model.summary()


        # summarize history for accuracy
        plt.plot(history.history['mape'])
        plt.plot(history.history['val_mape'])
        plt.title('model mape')
        plt.ylabel('mape')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

        # summarize history for accuracy
        plt.plot(history.history['mse'])
        plt.plot(history.history['val_mse'])
        plt.title('model mse')
        plt.ylabel('mse')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

        predict_test = model.predict(x_test)
        predict_train= model.predict(x_train)


        # scale back test
        predict_test = np.asarray(predict_test).reshape(-1, 1)
        predict_test_scale_back = scalar.inverse_transform(predict_test)
        y_test_scale_back = scalar.inverse_transform(y_test.reshape(-1, 1))


        # scale back train
        predict_train = np.asarray(predict_train).reshape(-1, 1)
        predict_train_scale_back = scalar.inverse_transform(predict_train)
        y_train_scale_back = scalar.inverse_transform(y_train.reshape(-1, 1))

        # plot test result
        forecasting_df = pd.DataFrame(predict_test, index=test[-n_prediction:].index,
                                      columns=['forecast'])
        forecasting_df['test'] = y_test

        forecasting_df.plot()
        plt.title("LSTM")

        plt.show()

        # print test result
        print(forecasting_df)

        # print MSE test result
        print('MSE test: ', mean_squared_error(y_test, predict_test))
        print('MSE train: ', mean_squared_error(y_train, predict_train))





    def tune_model(self, model):
        pass

    # transform each data to n features with each feature's n adjacent datapoint
    def prepare_train_data(self, data_dict, n_window, scalar):
        data = data_dict['value'].tolist()
        x = []
        y = []
        for i in range(len(data) - n_window - 1):
            extracted_feature = self.add_extract_feature(data_dict,scalar, i)
            this_x = data[i:i + n_window] + extracted_feature
            x.append(this_x)
            this_y = data[i + n_window]
            y.append(this_y)
        return np.asarray(x), np.asarray(y)

    def prepare_test_data(self, train, data_dict, n_window, scalar, n_prediction):
        missing_window = train['value'].tolist()[-n_window:]

        data = data_dict['value'].tolist()
        data = missing_window + data
        x = []
        y = []
        for i in range(n_prediction):
            extracted_feature = self.add_extract_feature(data_dict,scalar, i)
            this_x = data[i:i + n_window] + extracted_feature
            x.append(this_x)
            this_y = data[i + n_window]
            y.append(this_y)
        return np.asarray(x), np.asarray(y)

    def add_extract_feature(self, data, scalar, i):
        extracted_feature = []
        year = data['Year'].tolist()
        month = data['Month'].tolist()
        idI = data['id'].tolist()
        extracted_feature.append(year[i])
        extracted_feature.append(month[i])
        extracted_feature.append(idI[i])
        extracted_feature = np.asarray(extracted_feature)
        extracted_feature = extracted_feature.reshape(-1, 1)
        extracted_feature = scalar.transform(extracted_feature).reshape(-1).tolist()
        return extracted_feature
