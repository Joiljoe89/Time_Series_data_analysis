# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 16:06:07 2018

@author: joe
002X_A40L1X_ND,003X_AMRC1X_ND,004X_AMRC2X_ND,005X_ODIRLX_ND,012X_MEDL2X_NB,013X_A6EPLX_NS,018X_G2EG1X_NB,019X_G2EG2X_NB
"""

from __future__ import print_function

import numpy as np
import pandas as pd

Nmcu_data = pd.read_csv('NodeMCU_24_11_master.csv')

Nmcu_data = Nmcu_data[['ID','Date','Time','Temp3','Humidity','Pressure','Temp']]
print(Nmcu_data.head())

t5 = (Nmcu_data.loc[Nmcu_data['Time'] >='00:00:00'])
t6 = (t5.loc[t5['Time'] <='10:29:59'])

t1 = (Nmcu_data.loc[Nmcu_data['Time'] >='10:30:00'])
t2 = (t1.loc[t1['Time'] <='17:30:00'])

t3 = (Nmcu_data.loc[Nmcu_data['Time'] >='17:30:01'])
t4 = (t3.loc[t3['Time'] <='23:59:59'])

tt = [t6,t4]
tt = pd.concat(tt)


t2 = t2[['ID','Temp3','Humidity','Pressure','Temp']]
tt = tt[['ID','Temp3','Humidity','Pressure','Temp']]

t2 = (t2.loc[t2['ID'] == '012X_MEDL2X_NB'])
tt = (tt.loc[tt['ID'] == '012X_MEDL2X_NB'])

#preprocessing- standardize
t2['Temp3'] = (t2['Temp3'] - t2['Temp3'].mean())/t2['Temp3'].std()
t2['Humidity'] = (t2['Humidity'] - t2['Humidity'].mean())/t2['Humidity'].std()
t2['Pressure'] = (t2['Pressure'] - t2['Pressure'].mean())/t2['Pressure'].std()
t2['Temp'] = (t2['Temp'] - t2['Temp'].mean())/t2['Temp'].std()

tt['Temp3'] = (tt['Temp3'] - tt['Temp3'].mean())/tt['Temp3'].std()
tt['Humidity'] = (tt['Humidity'] - tt['Humidity'].mean())/tt['Humidity'].std()
tt['Pressure'] = (tt['Pressure'] - tt['Pressure'].mean())/tt['Pressure'].std()
tt['Temp'] = (tt['Temp'] - tt['Temp'].mean())/tt['Temp'].std()

l_t2 = len(t2)
print('day 10am-6pm:',l_t2)
l_tt = len(tt)
print('night:',l_tt)

a = np.zeros((5,4))
b = np.zeros((5,4))

for x in range(2, l_t2-2, 1):
    t = t2[['Temp3','Humidity','Pressure','Temp']][x-2:x+3]
    t1 = t.as_matrix()
    #a = np.concatenate((a,t1),axis=0)
    a = np.append(a,t1, axis=0)
    
print("a:",a.shape)
t2 = np.split(a, 3151)#divide a.shape by 5
t2 = t2[1::]

for x in range(2, l_tt-2, 1):
    t = tt[['Temp3','Humidity','Pressure','Temp']][x-2:x+3]
    t1 = t.as_matrix()
    #a = np.concatenate((a,t1),axis=0)
    b = np.append(b,t1, axis=0)
    
print("b:",b.shape)
tt = np.split(b, 8110)
tt = tt[1::]
#print(tt)

#tn = [t2, tt]
#tn = pd.concat(tn)
tn = np.concatenate((t2,tt),axis=0)

l_t2 = len(t2)
print('day 10am-6pm:',l_t2)
l_tt = len(tt)
print('night:',l_tt)

c1 = np.zeros(l_t2)
c2 = np.ones(l_tt)
type_label = np.concatenate((c1,c2),axis=0)
#features1 = tn[['Temp3','Humidity','Pressure']].as_matrix()
#features=(features1-features1.min())/(features1.max()-features1.min())

print('num of label:',type_label.shape)
print('num of features:',tn.shape)

#keras library
from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop,SGD,Adam
from keras.utils import np_utils
np.random.seed(1671)
from keras import regularizers
'''
#define the ConvNet
class LeNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        #CONV => RELU => POOL
        model.add(Conv2D(20, kernel_size=3, padding="same",input_shape=INPUT_SHAPE))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        #CONV => RELU => POOL
        model.add(Conv2D(50, kernel_size=3, border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        
        #Flatten => RELU layer
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        # a softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model
'''     
# network and training  
NB_EPOCH = 200
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 2 #NUM OF O/P = NUM OF DIGITS
OPTIMIZER = Adam()
IMG_ROWS, IMG_COLS = 5, 4 #input img dimension
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)
VALIDATION_SPLIT = 0.1
DROPOUT = 0.2

#DATA: SHUFFLED AND SPLIT BETWEEN TRAIN AND TEST SETS

#split data
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(tn, type_label, test_size = 0.10)
print('X_train:',X_train.shape)
print('X_test:',X_test.shape)
print('y_train:',y_train.shape)
print('y_test:',y_test.shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#normalize
#normalized_Nmcu_data=(Nmcu_data-Nmcu_data.min())/(Nmcu_data.max()-Nmcu_data.min())

#we need a n-datasize x [5x3x1] shape as input to the convnet
X_train = X_train[:,:,:, np.newaxis]
X_test = X_test[:,:,:, np.newaxis]

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

#convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

#initialize the optimizer and model
#model = LeNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)

model = Sequential()
#CONV => RELU => POOL
model.add(Conv2D(32, (3,2), padding="same",input_shape=INPUT_SHAPE))
model.add(Activation("tanh"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
#CONV => RELU => POOL
model.add(Conv2D(64, (2,2), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
model.add(Dropout(DROPOUT))
model.add(Conv2D(64, (2,2), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
#Flatten => RELU layer
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))
# a softmax classifier
model.add(Dense(NB_CLASSES))
model.add(Activation("softmax"))
model.summary()


#objective fn,optimizer
model.compile(loss='categorical_crossentropy',optimizer=OPTIMIZER,metrics=['accuracy'])
#print(X_train)
#print(Y_train)
history = model.fit(X_train,Y_train,batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

#save model
model_json = model.to_json()
open('med_node_Lenet_architecture.json','w').write(model_json)

#save weihts
from keras.models import load_model 
model.save('Med_12_node_day_n_convnet_Lenet_wt.h5')
#model = load_model('allnode_day_night_model.h5')

score = model.evaluate(X_test,Y_test,verbose=VERBOSE)
print("Test score",score[0])
print("Test accuracy",score[1])

import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy_MED_12 node_Convnet_Lenet opt-Adam')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss_MED_12 node_Convnet_Lenet opt-Adam')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()