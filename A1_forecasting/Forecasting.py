import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import A1.models.SARIMA as SARIMA
import A1.models.LinearReg as LinearReg
import A1.models.LSTM_model as lstm
# dataset https://www.kaggle.com/rakannimer/air-passengers


def data_preprocessing(data):
    n_test = 24
    n_train = 144 - n_test
    data['Date'] = pd.to_datetime(data['Date'],infer_datetime_format=True)
    dates = pd.date_range(start='1949-01-01', end='1960-12', freq='MS')

    data['Month'] = [i.month for i in dates]
    data['Year'] = [i.year for i in dates]
    data['id'] = [i for i in range(len(dates))]

    train={
        'val':data['#Passengers'][:n_train],
        'Month': data['Month'][:n_train],
        'Year':data['Year'][:n_train],
        'id': data['id'][:n_train]
    }
    test={
        'val':data['#Passengers'][-n_test:],
        'Month': data['Month'][-n_test:],
        'Year':data['Year'][-n_test:],
        'id': data['id'][-n_test:]
    }
    return data, pd.DataFrame(train), pd.DataFrame(test)

def plot_data(data):
    data.plot(x='Date', y='#Passengers')
    plt.show()

if __name__ == '__main__':
    data = pd.read_csv("D:\\ECE9063_data_analysis\\AirPassengers.csv")
    # preprocessing
    data, train, test  = data_preprocessing(data)
    # plot_data(data)
    data= data.set_index('Date', inplace=False)

    # SARIMA
    print('_________________SARIMA_______________________')
    sarima = SARIMA.SARIMA(data, train, test)

    # Linear regression
    print('______________Linear Regression____________')
    LR = LinearReg.LinearReg(data, train, test)

    # LSTM
    print('________________LSTM___________________')
    LSTM_model = lstm.LSTM_model(data, train, test)
