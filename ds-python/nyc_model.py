import numpy as np
import pandas as pd 
import os 
import torch
import torch.nn as nn 
from torch.autograd import Variable
from nyc_taxicabdata import attribute_index


PATH = '/localdisk3/nyc_2021-09_updated.csv'

def load_data():
    df = pd.read_csv(PATH)
    df.columns = attribute_index.keys()
    # print(df.head())
    # print(df.keys())
    df = df.drop(['VendorID', 'RatecodeID', 'store_and_fwd_flag',
       'PULocationID', 'DOLocationID', 'payment_type', 'extra',
       'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge',
       'total_amount', 'congestion_surcharge', 'airport_fee', 'ID'], axis=1)
    # print(df.head())
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    print(df.info())
    # converting dirty data to NaN --> 1
    # need to change this data file and make sure that the actual data is not dirty 
    df['passenger_count'] = df['passenger_count'].fillna(1)
    print(df.info())
    print(df.head())
    


if __name__ == '__main__':
    load_data()
