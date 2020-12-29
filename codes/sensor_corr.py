# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 19:40:58 2018

@author: joe

sensor correlation
"""

import pandas as pd
import matplotlib.pyplot as plt


Nmcu_data = pd.read_csv('NodeMCU_V3Data.csv', index_col='Date', parse_dates=True)
a4 = (Nmcu_data.loc[Nmcu_data['ID'] == '002X_A40L1X_ND'])
am = (Nmcu_data.loc[Nmcu_data['ID'] == '003X_AMRC1X_ND'])
me = (Nmcu_data.loc[Nmcu_data['ID'] == '012X_MEDL2X_NB'])
a6 = (Nmcu_data.loc[Nmcu_data['ID'] == '013X_A6EPLX_NS'])

a4 = a4[['Temp3']]
am = am[['Temp3']]
me = me[['Temp2']][1::]
a6 = a6[['Temp3']]

print(a4,am,me,a6)
a= a4[0:1400]
b= a6[0:1400]
#plt.plot(a4,'r.', label='A4')
#plt.plot(am,'g.', label='AMRC')
#plt.plot(me,'b.', label='MED')
#plt.plot(a6,'y.', label='A6')
plt.scatter(a,b)
# Create legend.
plt.legend(loc='upper right')
plt.xlabel('a4')
plt.ylabel('a6')

plt.show()  

