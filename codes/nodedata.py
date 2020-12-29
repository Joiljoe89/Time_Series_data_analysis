# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 10:14:13 2018

@author: joe
"""

import pandas as pd
#from pandas import DataFrame

#import datetime
#import pandas.io.data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
'''
xb_data = pd.read_csv('NodeMCU_V3Data.csv', delim_whitespace=True, header=None,
                      names = ['Sr.No','Date and Time','Id','Temp1','Temp2',
                      'Humidity','Pressure','Temp3','v1','v2'], na_values='?')
'''
#xb_data = pd.read_csv('NodeMCU_V3Data.csv', parse_dates={'Date Time': ['date', 'time']}, date_parser=dateparse)
#xb_data = pd.read_csv('NodeMCU_V3Data.csv', index_col = 'Date Time',parse_dates=True)
xb_data = pd.read_csv('NodeMCU_V3Data.csv', index_col='Date Time', parse_dates=True)
#print (xb_data)
'''
xb1 = xb_data['Date Time']
xb2 = xb1[0:10]
print (xb2)
xb3 = xb2[(xb2>xb2[5])]
print (xb3)
'''
#xb_data = xb_data[0:2000]

#xb_filter = xb_data.filter(like='AMRC$', axis=0)
#xb_filter = xb_data.filter(items=['ID','Temp1'])
#xb_filter = xb_data.filter(regex='e$', axis=1)

xb_filter = (xb_data.loc[xb_data['ID'] == '004X_AMRC2X_ND'])
print(xb_filter)
xb_data = xb_filter[['Temp1','Temp2','Temp3']]

'''
xb1 = xb_data['Temp1']
xb2 = xb1[0:500]
print (xb2)

#xb_diff = xb2.diff()
xb_ma = pd.rolling_mean(xb2,100)
print (xb_ma)   
#plt.plot(xb_ma)
'''
xb_data[['Temp1','Temp2','Temp3']].plot()
plt.show()         


'''
td = plt.figure().gca(projection='3d')
td.scatter(xb_data['Temp1'],xb_data['Temp2'],xb_data['Temp3'])
td.set_xlabel('temp1') 
td.set_ylabel('temp2')
td.set_zlabel('temp3') 
plt.show()   

ax1 = plt.subplot(2,1,1)
xb_data['Temp1'].plot()

#ax2 = plt.subplot(2,1,2)
ax2 = plt.subplot(2,1,2, sharex = ax1)
xb_data['Temp2'].plot()

plt.show() 
'''

#print (xb_data.describe())
print (xb_data.corr())
                