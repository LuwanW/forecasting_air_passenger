import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
class SARIMA:
    def __init__(self, data, train, test):
        self.data = data
        self.train = train
        self.test = test

        # analysis
        stationary = self.plot_log_and_first_order()
        plot_acf(stationary, lags=50, alpha=0.05)
        # plt.show()
        plot_pacf(stationary, lags=50, alpha=0.05)
        # plt.show()

        # turn to get pdqPDQm
        #model = self.train_model()
        # got Best model:  ARIMA(0,1,1)(2,1,0)[12]

        # forecasting
        self.plot_forecasting()

    def plot_first_order(self):
        diff = self.data[['#Passengers']].diff()
        diff.plot()
        plt.title('First difference')
        plt.show()

    def plot_log_and_first_order(self):
        after_log = np.log(self.data[['#Passengers']])
        one_diff = after_log.diff()
        two_diff = one_diff.diff()

        two_diff.plot()
        plt.title('log+first order')
        plt.show()
        return two_diff

    def train_model(self):

        sarima = pm.auto_arima(self.data,
                                start_p=0, start_q=0,
                                test='adf',
                                max_p=2, max_q=2, m=12,
                                start_P=0,start_Q=0,max_P=2, max_Q=2, seasonal=True,
                                d=1, D=1, trace=True,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True)
        sarima.summary()
        return sarima

    def plot_forecasting(self):
        # Best model: ARIMA(0, 1, 1)(2, 1, 0)[12]
        # model = SARIMAX(self.data['val_Train'],order=(0,1,1),seasonal_order=(2,1,0,12))
        # model = model.fit()
        # self.data['forecasting_sarima'] = model.predict(start=109, end=144, dynamic=True)
        # self.data[['forecasting_sarima','val_Train','val_Test']].plot()
        # plt.show()
        #
        # print('MSE', mean_squared_error(self.data['forecasting_sarima'], self.data['val_Test'] ))
        n_input = 24
        model = SARIMAX(self.train[['val']],order=(0,1,1),seasonal_order=(2,1,0,12))
        model = model.fit()

        forecast= model.predict(start=121, end=144, dynamic=True)
        forecasting_df = pd.DataFrame(forecast.tolist(),index = self.data[-n_input:].index,
                                      columns=['forecast'])
        forecasting_df['test'] = self.test['val'].tolist()
        forecasting_df.plot()
        plt.title("SARIMA")

        plt.show()
        print(forecasting_df)

        print('MSE', mean_squared_error(forecasting_df['forecast'], forecasting_df['test'] ))
