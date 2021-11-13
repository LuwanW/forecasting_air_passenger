import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import Forecasting_Air_Passengers.A2_deep_learning.LSTM_model_tuning as lstm
# dataset https://www.kaggle.com/rakannimer/air-passengers


def data_preprocessing(data, n_window):
    n_test = 24
    n_train = 144 - n_test
    data['Date'] = pd.to_datetime(data['Date'],infer_datetime_format=True)
    dates = pd.date_range(start='1949-01-01', end='1960-12', freq='MS')

    # add feature year, month, year sum
    m = []
    y = []
    for i in dates:
        m.append(i.month)
        y.append(i.year)
    data['Month'] = m
    data['Year'] = y
    data['id'] = [i for i in range(len(dates))]


    train={
        'value':data['#Passengers'][0:n_train],
        'Month': data['Month'][0:n_train],
        'Year':data['Year'][0:n_train],
        'id': data['id'][0:n_train],
    }
    test={
        'value':data['#Passengers'][n_train:144],
        'Month': data['Month'][n_train:144],
        'Year':data['Year'][n_train:144],
        'id': data['id'][n_train:144],

    }
    return data, pd.DataFrame(train), pd.DataFrame(test)





def plot_data(data):

    data.plot(x='Date', y='#Passengers')
    plt.show()

if __name__ == '__main__':
    data = pd.read_csv("D:\\ECE9063_data_analysis\\Forecasting_Air_Passengers\\AirPassengers.csv")
    # preprocessing
    n_window = 24
    data, train, test  = data_preprocessing(data, n_window)
    # plot_data(data)
    data= data.set_index('Date', inplace=False)


    # LSTM
    print('________________LSTM___________________')
    LSTM_model = lstm.LSTM_model(data, train, test, n_window)
