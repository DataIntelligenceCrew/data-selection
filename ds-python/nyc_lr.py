import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import time
import numpy as np
from sklearn.metrics import mean_squared_error
import argparse

class LinearRegressionModel:
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        self.model = linear_model.LinearRegression()
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
        return self.model.score(X, y)
    
    
# Convert the datetime values of tpep_pickup_datetime and tpep_dropoff_datetime to Unix timestamps
def convert_datetime_to_unix(df):
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    pickup_time_list = [time.mktime(date.timetuple()) for date in df['tpep_pickup_datetime']]
    pickup_time = np.array(pickup_time_list).reshape((-1, 1))
    df['tpep_pickup_datetime'] = pickup_time
    dropoff_time_list = [time.mktime(date.timetuple()) for date in df['tpep_dropoff_datetime']]
    dropoff_time = np.array(dropoff_time_list).reshape((-1, 1))
    df['tpep_dropoff_datetime'] = dropoff_time
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('datasize', type=str, help='The dataset to use')
    args = parser.parse_args()
    if args.datasize == '60k':
        data = pd.read_csv('nyc_60k_2021-09_updated.csv')
    elif args.datasize == '1mil':
        data = pd.read_csv('nyc_1mil_2021-09_updated.csv')
    
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    
    train = convert_datetime_to_unix(train)
    test = convert_datetime_to_unix(test)
    
    train_X = train[['PULat', 'PULong', 'DOLat', 'DOLong', 'trip_distance', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count']]
    train_y = train['fare_amount']
    test_X = test[['PULat', 'PULong', 'DOLat', 'DOLong', 'trip_distance', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count']]
    test_y = test['fare_amount']
    
    lr = LinearRegressionModel()
    lr.fit(train_X, train_y)
    print(lr.score(train_X, train_y))
    test_y_predictions = lr.predict(test_X)

    mse = mean_squared_error(test_y, test_y_predictions)
    rmse = np.sqrt(mse)

    print(rmse)