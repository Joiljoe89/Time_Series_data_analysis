# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 11:31:30 2018

@author: joe
"""

import pandas as pd
from pandas import DataFrame

import datetime
import pandas.io.data
"""
sp500 = pd.io.data.get_data_yahoo('%5EGSPC',start = datetime.datetime(2000,10,1)
,end = datetime.datetime(2014,6,11))

#print sp500
sp500.to_csv('sp500_ohlc.csv')
"""
df1 = pd.read_csv('NodeMCU_V2Data.csv')
#df1 = pd.read_csv('Mosqitto_data_250418.csv', index_col = 'Data', parse_dates=True)

print (df1)
#print (df1.head())