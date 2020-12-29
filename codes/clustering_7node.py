# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 18:22:00 2018

@author: joe
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

Nmcu_data = pd.read_csv('NodeMCU_V3Data.csv', index_col='Date', parse_dates=True)

Nmcu_data = Nmcu_data[['ID','Temp3','Humidity','Pressure']]
Nmcu_data = (Nmcu_data.loc[Nmcu_data['ID'] == '019X_G2EG2X_NB'])
Nmcu_data = Nmcu_data[10::]
features = ['Temp3','Humidity','Pressure']
# Separating out the features
x = Nmcu_data.loc[:, features].values
# Separating out the target
y = Nmcu_data.loc[:,['ID']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)
print(x)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(x)
y_kmeans = kmeans.predict(x)
'''
plt.scatter(x[:,0],x[:,1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0],centers[:,1], c='black', s=200, alpha = .5)
'''
centers = kmeans.cluster_centers_

td = plt.figure().gca(projection='3d')
#td.scatter(x[:,0],x[:,1],x[:,2],c='y',s=500,marker='^')
#td.scatter(x[:,0],x[:,1],x[:,2],c='y',s=500,marker='o')
td.scatter(x[:,0],x[:,1],x[:,2], c=y_kmeans, s=25, cmap='viridis')
td.scatter(centers[:,0],centers[:,1],centers[:,2], c='r', s=200, alpha = .5)
td.set_xlabel('temp') 
td.set_ylabel('humidity')
td.set_zlabel('pressure') 
td.set_title('3D plot of K-Means Clustering (4 clusters) for node G2_19_outdoor:under shade (019X_G2EG2X_NB)')
plt.show()

'''
a4_002 = (Nmcu_data.loc[Nmcu_data['ID'] == '002X_A40L1X_ND'])
am_003 = (Nmcu_data.loc[Nmcu_data['ID'] == '003X_AMRC1X_ND'])
am_004 = (Nmcu_data.loc[Nmcu_data['ID'] == '004X_AMRC2X_ND'])
od_005 = (Nmcu_data.loc[Nmcu_data['ID'] == '005X_ODIRLX_ND'])
me_012 = (Nmcu_data.loc[Nmcu_data['ID'] == '012X_MEDL2X_NB'])
a6_013 = (Nmcu_data.loc[Nmcu_data['ID'] == '013X_A6EPLX_NS'])
g2_018 = (Nmcu_data.loc[Nmcu_data['ID'] == '018X_G2EG1X_NB'])
g2_019 = (Nmcu_data.loc[Nmcu_data['ID'] == '019X_G2EG2X_NB'])
'''




'''
Nmcu_data['Temp3'] = StandardScaler().fit_transform(Nmcu_data[['Temp3']])
Nmcu_data['Humidity'] = StandardScaler().fit_transform(Nmcu_data[['Humidity']])
Nmcu_data['Pressure'] = StandardScaler().fit_transform(Nmcu_data[['Pressure']])

print(Nmcu_data)

am_003 = (Nmcu_data.loc[Nmcu_data['ID'] == '012X_MEDL2X_NB'])
#sns.pairplot(am_003, kind="reg",hue='ID',size=5)
plt.plot(am_003['Temp3'],am_003['Humidity'],'.')
plt.show()

'''