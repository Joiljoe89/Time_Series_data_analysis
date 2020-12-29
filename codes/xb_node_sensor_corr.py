# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 15:31:47 2018

@author: joe
sensor_corr xb node: 415d86f0,415d8755,415d8759,416c44c2,416c44e1

"""
import pandas as pd
import matplotlib.pyplot as plt


xb_data = pd.read_csv('xbiot2.csv', index_col='Date', parse_dates=True)
#Nmcu_filter = (Nmcu_data.loc[Nmcu_data['ID'] == '002X_A40L1X_ND'])
#Nmcu_data = Nmcu_filter[['Temp1','Temp2','Temp3','Humidity','Pressure']]

#normalized_Nmcu_data=(Nmcu_data-Nmcu_data.min())/(Nmcu_data.max()-Nmcu_data.min())
xb_data = xb_data[4900::]
xb_data = xb_data[['Node','c1','c2','c3','c4','c5','c6','c7','c9']]
#Nmcu_data = (Nmcu_data-Nmcu_data.min())/(Nmcu_data.max()-Nmcu_data.min())
#print(xb_data)

xf0 = (xb_data.loc[xb_data['Node'] == '415d86f0'])
x55 = (xb_data.loc[xb_data['Node'] == '415d8755'])
x59 = (xb_data.loc[xb_data['Node'] == '415d8759'])
xc2 = (xb_data.loc[xb_data['Node'] == '416c44c2'])
xe1 = (xb_data.loc[xb_data['Node'] == '416c44e1'])
#print(xe1)
xf0 = xf0[['c3']]
x55 = x55[['c9']]
x59 = x59[['c9']]
xc2 = xc2[['c7']]
xe1 = xe1[['c7']]
#print(xe1)

#print(a4,am,me,a6)
#a= a4[0:1400]
#b= a6[0:1400]

#plt.plot(xf0,'r-', label='415d86f0')
plt.plot(x55,'g-', label='415d8755')
plt.plot(x59,'b-', label='415d8759')
#plt.plot(xc2,'c-', label='416c44c2')
#plt.plot(xe1,'m-', label='416c44e1')

#plt.scatter(a,b)
# Create legend.
plt.legend(loc='upper right', fontsize=14)
plt.xlabel('date', fontsize=16)
plt.ylabel('Pressure/Pa', fontsize=16)
#plt.ylabel('Humidity in percentage', fontsize=16)
#plt.ylabel('Temperature/degree_C', fontsize=16)
#plt.title('Temperature measurement data using DS18B20 sensor from five xbee nodes placed in IIT, Mandi campus (*IoT lab,A7)', fontsize=16)
#plt.title('Humidity measurement data using BME280 and DHT22 sensor from five xbee nodes placed in IIT, Mandi campus (*IoT lab,A7)', fontsize=16)
plt.title('Pressure measurement data using BME280 sensor from five xbee nodes placed in IIT, Mandi campus (*IoT lab,A7)', fontsize=16)
plt.show()  
