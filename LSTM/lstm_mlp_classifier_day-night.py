# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 19:22:53 2018

@author: joe
"""

import pandas as pd
import matplotlib.pyplot as plt
##############################################################################
Nmcu_data = pd.read_csv('med_12_5dec.csv', index_col='Date', parse_dates=True)
##############################################################################
from pandas import DataFrame, concat

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
# load dataset
dataset = pd.read_csv('med_12_5dec.csv', header=0, index_col=0)
values = dataset.values
# ensure all data is float
values = values.astype('float32')
a1 = dataset['T'].min()
a2 = dataset['T'].max()
b1 = dataset['H'].min()
b2 = dataset['H'].max()
d1 = dataset['P'].min()
d2 = dataset['P'].max()
print(d1,d2)
# normalize features
dataset['T'] = (dataset['T'] - dataset['T'].min())/(dataset['T'].max() - dataset['T'].min())
dataset['H'] = (dataset['H'] - dataset['H'].min())/(dataset['H'].max() - dataset['H'].min())
dataset['P'] = (dataset['P'] - dataset['P'].min())/(dataset['P'].max() - dataset['P'].min())

values = dataset.values
#***********************************
# frame as supervised learning
n = 2
reframed_t = series_to_supervised(values, n, 1)
reframed_h = series_to_supervised(values, n, 1)
reframed_p = series_to_supervised(values, n, 1)
# drop columns we don't want to predict
reframed_t.drop(reframed_t.columns[[(3*n)+1,(3*n)+2]], axis=1, inplace=True)
reframed_h.drop(reframed_h.columns[[(3*n),(3*n)+2]], axis=1, inplace=True)
reframed_p.drop(reframed_p.columns[[(3*n),(3*n)+1]], axis=1, inplace=True)
print('temp_supervised',reframed_t.head(2))
print('humidity_supervised',reframed_h.head(2))
print('pressure_supervised',reframed_p.head(2))
##############################################################################
values_t = reframed_t.values
values_h = reframed_h.values
values_p = reframed_p.values

# split into train and test sets
l_values = len(values_t)
print(l_values)
cz = 13100
train_t, test_t = values_t[0::], values_t[0:200]
train_h, test_h = values_h[0::], values_h[0:200]
train_p, test_p = values_p[0::], values_p[0:200]
# split into input and outputs
train_X_t, train_y_t = train_t[:, :-1], train_t[:, -1]
test_X_t, test_y_t = test_t[:, :-1], test_t[:, -1]
#######
train_X_h, train_y_h = train_h[:, :-1], train_h[:, -1]
test_X_h, test_y_h = test_h[:, :-1], test_h[:, -1]
#######
train_X_p, train_y_p = train_p[:, :-1], train_p[:, -1]
test_X_p, test_y_p = test_p[:, :-1], test_p[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X_t = train_X_t.reshape((train_X_t.shape[0], 1, train_X_t.shape[1]))
test_X_t = test_X_t.reshape((test_X_t.shape[0], 1, test_X_t.shape[1]))
print(train_X_t.shape, train_y_t.shape, test_X_t.shape, test_y_t.shape)
#####
train_X_h = train_X_h.reshape((train_X_h.shape[0], 1, train_X_h.shape[1]))
test_X_h = test_X_h.reshape((test_X_h.shape[0], 1, test_X_h.shape[1]))
#####
train_X_p = train_X_p.reshape((train_X_p.shape[0], 1, train_X_p.shape[1]))
test_X_p = test_X_p.reshape((test_X_p.shape[0], 1, test_X_p.shape[1]))

##############################################################################
from keras.models import Sequential
from keras.layers import Dense, LSTM
# design network
model_t = Sequential()
model_t.add(LSTM(50, input_shape=(train_X_t.shape[1], train_X_t.shape[2]))) #50 neurons in the first hidden layer 
#The input shape will be 1 time step with 3 features
model_t.add(Dense(1) )#1 neuron in the output layer for predicting
model_t.compile(loss='mae', optimizer='adam')#Mean Absolute Error (MAE) loss function and the efficient Adam version of stochastic gradient descent.
print(model_t.summary())
####
model_h = Sequential()
model_h.add(LSTM(50, input_shape=(train_X_h.shape[1], train_X_h.shape[2]))) #50 neurons in the first hidden layer 
#The input shape will be 1 time step with 3 features
model_h.add(Dense(1) )#1 neuron in the output layer for predicting
model_h.compile(loss='mae', optimizer='adam')
####
model_p = Sequential()
model_p.add(LSTM(50, input_shape=(train_X_p.shape[1], train_X_p.shape[2]))) #50 neurons in the first hidden layer 
#The input shape will be 1 time step with 3 features
model_p.add(Dense(1) )#1 neuron in the output layer for predicting
model_p.compile(loss='mae', optimizer='adam')
# fit network
history_t = model_t.fit(train_X_t, train_y_t, epochs=10, batch_size=32, validation_data=(test_X_t, test_y_t), verbose=2, shuffle=False)
history_h = model_h.fit(train_X_h, train_y_h, epochs=10, batch_size=32, validation_data=(test_X_h, test_y_h), verbose=2, shuffle=False)
history_p = model_p.fit(train_X_p, train_y_p, epochs=10, batch_size=32, validation_data=(test_X_p, test_y_p), verbose=2, shuffle=False)
# plot history
plt.plot(history_t.history['loss'], label='train_temp')
plt.plot(history_t.history['val_loss'], label='test_temp')
plt.plot(history_h.history['loss'], label='train_humidity')
plt.plot(history_h.history['val_loss'], label='test_humidity')
plt.plot(history_p.history['loss'], label='train_pressure')
plt.plot(history_p.history['val_loss'], label='test_pressure')
plt.title('Model loss for MED_12 node using LSTM opt-Adam')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train_temp', 'test_temp', 'train_humidity', 'test_humidity', 'train_pressure', 'test_pressure'], loc='upper left')
plt.show()

##############################################################################
from math import sqrt
from numpy import concatenate
from sklearn.metrics import mean_squared_error
# make a prediction
yhat_t = model_t.predict(test_X_t)
#print(yhat_t)
test_X_t = test_X_t.reshape((test_X_t.shape[0], test_X_t.shape[2]))
# invert scaling for forecast
inv_yhat_t = concatenate((yhat_t, test_X_t[:, 1:]), axis=1)
#inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat_t[:,0] = (inv_yhat_t[:,0]*(a2-a1))+a1
inv_yhat_t = inv_yhat_t[:,0]
# invert scaling for actual
test_y_t = test_y_t.reshape((len(test_y_t), 1))
inv_y_t = concatenate((test_y_t, test_X_t[:, 1:]), axis=1)
#inv_y = scaler.inverse_transform(inv_y)
inv_y_t[:,0] = (inv_y_t[:,0]*(a2-a1))+a1
inv_y_t = inv_y_t[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y_t, inv_yhat_t))
print('Test RMSE temp: %.3f' % rmse) #Root Mean Squared Error (RMSE) that gives error in the same units as the variable itself
############
yhat_h = model_h.predict(test_X_h)
test_X_h = test_X_h.reshape((test_X_h.shape[0], test_X_h.shape[2]))
inv_yhat_h = concatenate((yhat_h, test_X_h[:, 1:]), axis=1)
inv_yhat_h[:,0] = (inv_yhat_h[:,0]*(b2-b1))+b1
inv_yhat_h = inv_yhat_h[:,0]
test_y_h = test_y_t.reshape((len(test_y_h), 1))
inv_y_h = concatenate((test_y_h, test_X_h[:, 1:]), axis=1)
inv_y_h[:,0] = (inv_y_h[:,0]*(b2-b1))+b1
inv_y_h = inv_y_h[:,0]
rmse = sqrt(mean_squared_error(inv_y_h, inv_yhat_h))
print('Test RMSE humidity: %.3f' % rmse)
############
yhat_p = model_p.predict(test_X_p)
test_X_p = test_X_p.reshape((test_X_p.shape[0], test_X_p.shape[2]))
inv_yhat_p = concatenate((yhat_p, test_X_p[:, 1:]), axis=1)
inv_yhat_p[:,0] = (inv_yhat_p[:,0]*(d2-d1))+d1
inv_yhat_p = inv_yhat_p[:,0]
test_y_p = test_y_p.reshape((len(test_y_p), 1))
inv_y_p = concatenate((test_y_p, test_X_p[:, 1:]), axis=1)
inv_y_p[:,0] = (inv_y_p[:,0]*(d2-d1))+d1
inv_y_p = inv_y_p[:,0]
rmse = sqrt(mean_squared_error(inv_y_p, inv_yhat_p))
print('Test RMSE pressure: %.3f' % rmse)
###########################
plt.plot(inv_y_t)
plt.plot(inv_yhat_t)
plt.title('Model loss for MED_12 node using LSTM opt-Adam')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['test_temp', 'predicted_temp'], loc='upper left')
plt.show()
######################

import numpy as np
l = len(test_X_t)
ra = np.zeros((l,3))
for x in range(0,l,1):
    ra[x][0] = inv_yhat_t[x]
    ra[x][1] = inv_yhat_h[x]
    ra[x][2] = inv_yhat_p[x]

print(ra[4:5])
lstm_predict = ra

#############################################################################

#classification
import numpy as np
from keras.models import model_from_json
from keras.optimizers import Adam

#load model
model_architecture = 'med_node_mlp_architecture_n_2n_n.json'
model_weights = 'Med_12_node_day_n_mlp_wt_n_2n_n.h5'

model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)
#train
model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])

#predict
prediction_class = model.predict_classes(lstm_predict[4:5])
print(prediction_class)
if prediction_class==0:
        print('day')
'''
for i in range(0,l,1):
    if prediction_class[i]==0:
        print('day')
    else:
        print('night')
   ''' 
