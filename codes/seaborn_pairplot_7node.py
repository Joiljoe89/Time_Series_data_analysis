# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 12:18:14 2018

@author: joe
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", color_codes=True)

Nmcu_data = pd.read_csv('NodeMCU_V3Data.csv', index_col='Date', parse_dates=True)
Nmcu_data = Nmcu_data[10::]
Nmcu_data = Nmcu_data[['ID','Temp3','Humidity','Pressure']]
print(Nmcu_data)
a4_002 = (Nmcu_data.loc[Nmcu_data['ID'] == '002X_A40L1X_ND'])
am_003 = (Nmcu_data.loc[Nmcu_data['ID'] == '003X_AMRC1X_ND'])
am_004 = (Nmcu_data.loc[Nmcu_data['ID'] == '004X_AMRC2X_ND'])
od_005 = (Nmcu_data.loc[Nmcu_data['ID'] == '005X_ODIRLX_ND'])
me_012 = (Nmcu_data.loc[Nmcu_data['ID'] == '012X_MEDL2X_NB'])
a6_013 = (Nmcu_data.loc[Nmcu_data['ID'] == '013X_A6EPLX_NS'])
g2_018 = (Nmcu_data.loc[Nmcu_data['ID'] == '018X_G2EG1X_NB'])
g2_019 = (Nmcu_data.loc[Nmcu_data['ID'] == '019X_G2EG2X_NB'])
#g2_018 = g2_018[['Temp3','Humidity','Pressure']]
#g2_018=(g2_018-g2_018.min())/(g2_018.max()-g2_018.min()) #normalization
sns.pairplot(a6_013, kind="reg",hue='ID',size=5)

plt.show()
