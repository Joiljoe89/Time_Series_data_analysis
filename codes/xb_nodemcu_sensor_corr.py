# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 17:38:10 2018

@author: joe
"""

import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

Nmcu_data = pd.read_csv('NodeMCU_V3Data.csv', index_col='Date', parse_dates=True)
xb_data = pd.read_csv('xbiot2.csv', index_col='Date', parse_dates=True)

#Nmcu_filter = (Nmcu_data.loc[Nmcu_data['ID'] == '002X_A40L1X_ND'])
#Nmcu_data = Nmcu_filter[['Temp1','Temp2','Temp3','Humidity','Pressure']]

#normalized_Nmcu_data=(Nmcu_data-Nmcu_data.min())/(Nmcu_data.max()-Nmcu_data.min())
Nmcu_data = Nmcu_data[500::]
Nmcu_data = Nmcu_data[['ID','Temp1','Temp2','Temp3','Humidity','Pressure']]
xb_data = xb_data[500::]
xb_data = xb_data[['Node','c1','c2','c3','c4','c5','c6','c7','c9']]
#Nmcu_data = (Nmcu_data-Nmcu_data.min())/(Nmcu_data.max()-Nmcu_data.min())


a4_002 = (Nmcu_data.loc[Nmcu_data['ID'] == '002X_A40L1X_ND'])
am_003 = (Nmcu_data.loc[Nmcu_data['ID'] == '003X_AMRC1X_ND'])
am_004 = (Nmcu_data.loc[Nmcu_data['ID'] == '004X_AMRC2X_ND'])
od_005 = (Nmcu_data.loc[Nmcu_data['ID'] == '005X_ODIRLX_ND'])
me_012 = (Nmcu_data.loc[Nmcu_data['ID'] == '012X_MEDL2X_NB'])
a6_013 = (Nmcu_data.loc[Nmcu_data['ID'] == '013X_A6EPLX_NS'])
g2_018 = (Nmcu_data.loc[Nmcu_data['ID'] == '018X_G2EG1X_NB'])
g2_019 = (Nmcu_data.loc[Nmcu_data['ID'] == '019X_G2EG2X_NB'])

a4_002 = a4_002[['Pressure']]
am_003 = am_003[['Pressure']]
am_004 = am_004[['Pressure']]
od_005 = od_005[['Pressure']]
me_012 = me_012[['Pressure']]
a6_013 = a6_013[['Pressure']]
g2_018 = g2_018[['Pressure']]
g2_019 = g2_019[['Pressure']]

xf0 = (xb_data.loc[xb_data['Node'] == '415d86f0'])
x55 = (xb_data.loc[xb_data['Node'] == '415d8755'])
x59 = (xb_data.loc[xb_data['Node'] == '415d8759'])
xc2 = (xb_data.loc[xb_data['Node'] == '416c44c2'])
xe1 = (xb_data.loc[xb_data['Node'] == '416c44e1'])
#print(xe1)
xf0 = xf0[['c3']]/100
x55 = x55[['c3']]/100
x59 = x59[['c3']]/100
#xc2 = xc2[['c7']]
#xe1 = xe1[['c7']]

plt.plot(xf0,'r--', label='415d86f0')
plt.plot(x55,'g--', label='415d8755')
plt.plot(x59,'b--', label='415d8759')
#plt.plot(xc2,'c-', label='416c44c2')
#plt.plot(xe1,'m-', label='416c44e1')

#plt.plot(a4_002,'r.', label='A4_indoor*')
plt.plot(am_003,'g-', label='AMRC3_indoor')
plt.plot(am_004,'b-', label='AMRC4_indoor')
plt.plot(od_005,'c-', label='OD5_indoor')
plt.plot(me_012,'m-', label='MED12_outdoor')
plt.plot(a6_013,'y-', label='A6_13_outdoor_solar')
plt.plot(g2_018,'k-', label='G2_18_indoor')
plt.plot(g2_019,'r-', label='G2_19_indoor')
#plt.scatter(a,b)
# Create legend.
plt.legend(loc='upper right', fontsize=16)
plt.xlabel('date', fontsize=18)
plt.ylabel('Temperature/degree_C', fontsize=18)
plt.title('Temperature measurement data using DS18B20 sensor from seven nodes placed in IIT, Mandi campus (*IoT lab,A7)', fontsize=16)
plt.show()  
