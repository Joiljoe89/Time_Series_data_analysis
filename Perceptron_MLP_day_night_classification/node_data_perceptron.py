# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 09:03:04 2018

@author: joe

002X_A40L1X_ND,003X_AMRC1X_ND,004X_AMRC2X_ND,005X_ODIRLX_ND,012X_MEDL2X_NB,013X_A6EPLX_NS,018X_G2EG1X_NB,019X_G2EG2X_NB
"""

from __future__ import print_function

import numpy as np
import pandas as pd

Nmcu_data = pd.read_csv('NodeMCU_24_11_master.csv')

Nmcu_data = Nmcu_data[['ID','Date','Time','Temp3','Humidity','Pressure']]
print(Nmcu_data.head())

t5 = (Nmcu_data.loc[Nmcu_data['Time'] >='00:00:00'])
t6 = (t5.loc[t5['Time'] <='10:29:59'])

t1 = (Nmcu_data.loc[Nmcu_data['Time'] >='10:30:00'])
t2 = (t1.loc[t1['Time'] <='17:30:00'])

t3 = (Nmcu_data.loc[Nmcu_data['Time'] >='17:30:01'])
t4 = (t3.loc[t3['Time'] <='23:59:59'])

tt = [t6,t4]

tt = pd.concat(tt)

t2 = t2[['ID','Temp3','Humidity','Pressure']]
tt = tt[['ID','Temp3','Humidity','Pressure']]

t2 = (t2.loc[t2['ID'] == '012X_MEDL2X_NB'])
tt = (tt.loc[tt['ID'] == '012X_MEDL2X_NB'])

#preprocessing- standardize
t2['Temp3'] = (t2['Temp3'] - t2['Temp3'].mean())/t2['Temp3'].std()
t2['Humidity'] = (t2['Humidity'] - t2['Humidity'].mean())/t2['Humidity'].std()
t2['Pressure'] = (t2['Pressure'] - t2['Pressure'].mean())/t2['Pressure'].std()


tt['Temp3'] = (tt['Temp3'] - tt['Temp3'].mean())/tt['Temp3'].std()
tt['Humidity'] = (tt['Humidity'] - tt['Humidity'].mean())/tt['Humidity'].std()
tt['Pressure'] = (tt['Pressure'] - tt['Pressure'].mean())/tt['Pressure'].std()


l_t2 = len(t2)
print('day 10am-6pm:',l_t2)
l_tt = len(tt)
print('night:',l_tt)

tn = [t2, tt]
tn = pd.concat(tn)

c1 = np.zeros(l_t2)
c2 = np.ones(l_tt)
type_label = np.concatenate((c1,c2),axis=0)
features1 = tn[['Temp3','Humidity','Pressure']].as_matrix()
features=(features1-features1.min())/(features1.max()-features1.min())

print('num of label:',type_label.shape)
print('num of features:',features.shape)

#keras library
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop,SGD,Adam
from keras.utils import np_utils
np.random.seed(1671)
from keras import regularizers

#network and training
NB_EPOCH = 500
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 2 #NUM OF O/P = NUM OF DIGITS
OPTIMIZER = Adam()
N_HIDDEN = 4
VALIDATION_SPLIT = 0.1
DROPOUT = 0.2

#DATA: SHUFFLED AND SPLIT BETWEEN TRAIN AND TEST SETS

#split data
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(features, type_label, test_size = 0.10)
print('X_train:',X_train.shape)
print('X_test:',X_test.shape)
print('y_train:',y_train.shape)
print('y_test:',y_test.shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#normalize
#normalized_Nmcu_data=(Nmcu_data-Nmcu_data.min())/(Nmcu_data.max()-Nmcu_data.min())

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

#convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)


#N_HIDDEN hidden layers
#2 outputs
#final stage is softmax
model = Sequential()
model.add(Dense(N_HIDDEN,input_shape=(3,),kernel_regularizer=regularizers.l2(0.0001)))
model.add(Activation('tanh'))
model.add(Dropout(DROPOUT))
model.add(Dense(2*N_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
model.add(Activation('sigmoid'))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()

#objective fn,optimizer
model.compile(loss='categorical_crossentropy',optimizer=OPTIMIZER,metrics=['accuracy'])
#print(X_train)
#print(Y_train)
history = model.fit(X_train,Y_train,batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

#save model
model_json = model.to_json()
open('med_node_mlp_architecture_n_2n_n.json','w').write(model_json)

#save weihts
from keras.models import load_model 
model.save('Med_12_node_day_n_mlp_wt_n_2n_n.h5')
#model = load_model('allnode_day_night_model.h5')

score = model.evaluate(X_test,Y_test,verbose=VERBOSE)
print("Test score",score[0])
print("Test accuracy",score[1])

t1 = np.ones(3)
t1[0] = 23
t1[1] = 45
t1[2] = 900
t1=(t1-features1.min())/(features1.max()-features1.min())
t1 = t1.astype('float32')
t1 = [[t1]]
#t1 = [[28.25,  66.7,  901.55]]
#print(t1.shape)
print(t1)
prediction_class = model.predict_classes(t1)
print(prediction_class)

#Visualization
'''
from keras.utils import plot_model
plot_model(model, to_file='model.png')
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
'''
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy_MED_12 node_N_hidden_4_opt-Adam')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss_MED_12 node_N_hidden_128_opt-Adam')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

'''
#gl = model.get_layer(name = None, index=3)
#print(gl)

#calculate predictions
X_train1 = X_test[1]
X_train1 = [[X_train1]]
#print(X_train1[0])
predictions = model.predict(X_train1)
print(predictions)

p=model.predict_classes(X_train1) #compute category outputs
#p = model.predict_proba(X_train1) #compute class probability
print(p)
'''
