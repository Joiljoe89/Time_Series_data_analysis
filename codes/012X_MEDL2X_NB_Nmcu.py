# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 14:09:04 2018

@author: joe
for node 012X_MEDL2X_NB - T1,T2,Hum,P,T3
"""
import pandas as pd
import matplotlib.pyplot as plt


Nmcu_data = pd.read_csv('NodeMCU_V3Data.csv', index_col='Date', parse_dates=True)
Nmcu_filter = (Nmcu_data.loc[Nmcu_data['ID'] == '022X_A7EG1X_NB'])

#Nmcu_data = Nmcu_filter[['Temp1','Temp2','Temp3','Humidity','Pressure']]
Nmcu_data = Nmcu_filter[['Temp2','Humidity','Pressure','Temp3','v1']]
#Nmcu_data = Nmcu_data[8::]
print(Nmcu_data)
#Nmcu_data[['Temp1','Temp2','Temp3','Humidity']].plot()
Nmcu_data[['Temp2','Temp3','Humidity','v1']].plot()
plt.ylabel('temp3,Temp2,Humidity,v1')
plt.xlabel('Date')
plt.title('022X_A7EG1X_NB')
plt.show()

#print (Nmcu_data.describe())
print (Nmcu_data.corr())