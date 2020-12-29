# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:16:03 2018

@author: joe
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import seaborn as sns
sns.set(style="ticks", color_codes=True)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

Nmcu_data = pd.read_csv('NodeMCU_V3Data.csv', index_col='Date', parse_dates=True)
Nmcu_data = Nmcu_data[10::]
Nmcu_data = Nmcu_data[['ID','Temp3','Humidity','Pressure']]
#Nmcu_data = (Nmcu_data.loc[Nmcu_data['ID'] == '019X_G2EG2X_NB'])
Nmcu_data = Nmcu_data[10800::]

features = ['Temp3', 'Humidity', 'Pressure']
# Separating out the features
x = Nmcu_data.loc[:, features].values
# Separating out the target
y = Nmcu_data.loc[:,['ID']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)
#print(x)

#PCA
model = PCA(n_components=2)
X_2D = model.fit(x)
#a = X_2D[:,0]
#b = X_2D[:,1]
#sns.lmplot(a,b,hue='ID')
print("pca_components:",model.components_)
print("variance",model.explained_variance_)
#dimensionality reduction
X_pca = model.transform(x)
print("original shape:   ", x.shape)
print("transformed shape:", X_pca.shape)

a = X_pca[:,0]
b = X_pca[:,1]
plt.plot(a,b,'.')
# Create legend.
plt.legend(loc='upper right', fontsize=16)
plt.xlabel('Principle component 1 (standardized data)', fontsize=18)
plt.ylabel('Principle component 2 (standardized data)', fontsize=16)
plt.title('PCA of seven nodes data together placed in IIT, Mandi campus (*IoT lab,A7)', fontsize=16)
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

#scaler = StandardScaler()
#scaler.fit(a4_002['Temp3'])
#X_scaled_array = scaler.transform(a4_002['Temp3'])
#X_scaled = pd.DataFrame(X_scaled_array, columns = X.columns)
#print(X_scaled_array.sample(5))

pca = PCA(n_components=2)
pca.fit(a4_002[['Temp3','Humidity','Pressure']])
X_2D = pca.transform(a4_002)

a = X_2D[:,0]
b = X_2D[:,1]
sns.lmplot(a,b,hue='ID')
'''