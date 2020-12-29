# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 19:12:37 2018

@author: joe
"""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({'font.size': 18})

Nmcu_data = pd.read_csv('NodeMCU_V3Data.csv', index_col='Date', parse_dates=True)

Nmcu_data = Nmcu_data[500::]
Nmcu_data = Nmcu_data[['ID','Temp1','Temp2','Temp3','Humidity','Pressure']]

a4_002 = (Nmcu_data.loc[Nmcu_data['ID'] == '002X_A40L1X_ND'])
am_003 = (Nmcu_data.loc[Nmcu_data['ID'] == '003X_AMRC1X_ND'])
am_004 = (Nmcu_data.loc[Nmcu_data['ID'] == '004X_AMRC2X_ND'])
od_005 = (Nmcu_data.loc[Nmcu_data['ID'] == '005X_ODIRLX_ND'])
me_012 = (Nmcu_data.loc[Nmcu_data['ID'] == '012X_MEDL2X_NB'])
a6_013 = (Nmcu_data.loc[Nmcu_data['ID'] == '013X_A6EPLX_NS'])
g2_018 = (Nmcu_data.loc[Nmcu_data['ID'] == '018X_G2EG1X_NB'])
g2_019 = (Nmcu_data.loc[Nmcu_data['ID'] == '019X_G2EG2X_NB'])

print(am_003[1632:1633])
print(am_003[1795:1796])

a4_002_p = a4_002[['Pressure']]
am_003_p = am_003[['Pressure']]
am_004_p = am_004[['Pressure']]
od_005_p = od_005[['Pressure']]
me_012_p = me_012[['Pressure']]
a6_013_p = a6_013[['Pressure']]
g2_018_p = g2_018[['Pressure']]
g2_019_p = g2_019[['Pressure']]
#print(od_005)

a4_002_h = a4_002[['Humidity']]
am_003_h = am_003[['Humidity']]
am_004_h = am_004[['Humidity']]
od_005_h = od_005[['Humidity']]
me_012_h = me_012[['Humidity']]
a6_013_h = a6_013[['Humidity']]
g2_018_h = g2_018[['Humidity']]
g2_019_h = g2_019[['Humidity']]

a4_002_t = a4_002[['Temp3']]
am_003_t = am_003[['Temp3']]
am_004_t = am_004[['Temp3']]
od_005_t = od_005[['Temp3']]
me_012_t = me_012[['Temp3']]
a6_013_t = a6_013[['Temp3']]
g2_018_t = g2_018[['Temp3']]
g2_019_t = g2_019[['Temp3']]


#plt.plot(a4_002,'r.', label='A4_indoor*')
#plt.plot(am_003_h,am_003_p,'g.', label='AMRC3_indoor:weather controlled')
#plt.plot(am_004_h,am_004_p,'b.', label='AMRC4_indoor')
#plt.plot(od_005_h,od_005_p,'c.', label='OD5_indoor')
#plt.plot(me_012_t,me_012_p,'m.', label='MED12_outdoor:direct exposure to sunlight')
#plt.plot(a6_013_t,a6_013_p,'y.', label='A6_13_outdoor:under shade of solar panel')
#plt.plot(g2_018_t,g2_018_p,'k.', label='G2_18_outdoor:under shade')
#plt.plot(g2_019_t,g2_019_p,'r.', label='G2_19_outdoor:under shade')

td = plt.figure().gca(projection='3d')
#td.scatter(me_012_t[1020:1021],me_012_h[1020:1021],me_012_p[1020:1021],c='b',s=500,marker='^')
#td.scatter(me_012_t[1160:1161],me_012_h[1160:1161],me_012_p[1160:1161],c='b',s=500,marker='o')
td.scatter(am_003_t[1632:1633],am_003_h[1632:1633],am_003_p[1632:1633],c='y',s=500,marker='^')
td.scatter(am_003_t[1795:1796],am_003_h[1795:1796],am_003_p[1795:1796],c='y',s=500,marker='o')

td.scatter(am_003_t,am_003_h,am_003_p,c='g')
#td.scatter(am_004_t,am_004_h,am_004_p,c='b')
#td.scatter(od_005_t,od_005_h,od_005_p,c='c')
#td.scatter(me_012_t,me_012_h,me_012_p,c='m')
#td.scatter(a6_013_t,a6_013_h,a6_013_p,c='y')
#td.scatter(g2_018_t,g2_018_h,g2_018_p,c='k')
#td.scatter(g2_019_t,g2_019_h,g2_019_p,c='r')

td.set_xlabel('temp') 
td.set_ylabel('humidity')
td.set_zlabel('pressure') 
td.set_title('MED12_outdoor:direct exposure to sunlight')
'''
# Create legend.
plt.legend(loc='upper right', fontsize=16)
plt.xlabel('Temperature/degree_C', fontsize=18)
plt.ylabel('Pressure/hPa', fontsize=16)
#plt.ylabel('Humidity in percentage', fontsize=16)
#plt.ylabel('Temperature/degree_C', fontsize=16)
plt.title('Temperature measurement data using DS18B20 sensor from seven nodes placed in IIT, Mandi campus (*IoT lab,A7)', fontsize=16)
#plt.title('Humidity measurement data using BME280 sensor from seven nodes placed in IIT, Mandi campus (*IoT lab,A7)', fontsize=16)
#plt.title('Pressure measurement data using BME280 sensor from seven nodes placed in IIT, Mandi campus (*IoT lab,A7)', fontsize=16)
'''
plt.show()  