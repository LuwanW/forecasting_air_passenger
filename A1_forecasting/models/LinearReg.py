from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from collections import deque
class LinearReg:
    def __init__(self, data, train, test):
        x_train, y_train = self.prepare_data(train['val'].tolist(),24)

        print(train, test)
        print(x_train, y_train )

        # train model
        lr = LinearRegression()
        lr.fit(x_train, y_train)

        n_input = 24
        forecasting = []
        batch = deque(train['val'].tolist()[-n_input:],maxlen=24)
        for i in range(n_input):
            prediction = lr.predict([batch])
            forecasting.append(prediction)
            batch.append(forecasting[-1])

        forecasting = np.array(forecasting)
        forecasting.reshape(-1,1)
        forecasting_df = pd.DataFrame(forecasting, index=data[-n_input:].index,
                                      columns=['forecast'])
        forecasting_df['test'] = test['val'].tolist()

        forecasting_df.plot()
        plt.title("LinearRegression")
        plt.show()

        # print R2 test result
        print(forecasting_df)
        y_true = forecasting_df['test']
        y_pred = forecasting_df['forecast']
        print("R2,",r2_score(y_true, y_pred))
        # print mean_squared_error test result
        print("MSE", mean_squared_error(y_true,y_pred))


    # transform each data to n features with each feature's n adjacent datapoint
    def prepare_data(self, data, n):
        x=[]
        y=[]
        for i in range(len(data)-n-1):
            x_item = data[i:i+n]
            x.append(np.asarray(x_item))
            y.append(data[i+n])

        return np.asarray(x),np.asarray(y)



