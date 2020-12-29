# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 13:49:02 2018

@author: joe
"""

import numpy as np
import scipy.misc
from keras.models import model_from_json
from keras.optimizers import Adam

#load model
model_architecture = 'med_node_mlp_architecture_n_2n_n.json'
model_weights = 'Med_12_node_day_n_mlp_wt_n_2n_n.h5'

model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)

#load data
t1 = np.ones(3)
t1[0] = 23
t1[1] = 45
t1[2] = 900
#t1=(t1-features1.min())/(features1.max()-features1.min())
t1 = t1.astype('float32')
t1 = [[t1]]
#t1 = [[28.25,  66.7,  901.55]]
#print(t1.shape)
print(t1)

#train
model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])

#predict
prediction_class = model.predict_classes(t1)
print(prediction_class)
