#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:17:00 2020

@author: torstengross
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle
import gzip
import json

import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="3" #specify GPU 
import keras as K
import tensorflow as tf
from keras import backend
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential
from keras.layers import Dense, Dropout

import normalize

SCRIPT_FOLDER = os.path.dirname(__file__)
#DATA_FOLDER = os.path.join(SCRIPT_FOLDER, 'data/')

with open(os.path.join(SCRIPT_FOLDER,'config.json')) as f:
    hyperparas = json.load(f)

layers = hyperparas['layers']
epochs = hyperparas['epochs']
act_func = hyperparas['act_func']
dropout = hyperparas['dropout']
input_dropout = hyperparas['input_dropout']
eta = hyperparas['eta']
norm = hyperparas['norm']

if hyperparas['act_func'] == 'relu':
    hyperparas['act_func'] = tf.nn.relu
    
    
    
# #### Define smoothing functions for early stopping parameter

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test = normalize.split_data(
    norm = 'tanh', test_fold = 0, val_fold = 1)

config = tf.ConfigProto(
         allow_soft_placement=True,
         gpu_options = tf.GPUOptions(allow_growth=True))
set_session(tf.Session(config=config))



model = Sequential()
for i in range(len(layers)):
    if i==0:
        model.add(Dense(layers[i], input_shape=(X_tr.shape[1],), activation=act_func, 
                        kernel_initializer='he_normal'))
        model.add(Dropout(float(input_dropout)))
    elif i==len(layers)-1:
        model.add(Dense(layers[i], activation='linear', kernel_initializer="he_normal"))
    else:
        model.add(Dense(layers[i], activation=act_func, kernel_initializer="he_normal"))
        model.add(Dropout(float(dropout)))
    model.compile(loss='mean_squared_error', optimizer=K.optimizers.SGD(lr=float(eta), momentum=0.5))


# #### run model for hyperparameter selection

# In[8]:


hist = model.fit(X_tr, y_tr, epochs=epochs, shuffle=True, batch_size=64, validation_data=(X_val, y_val))
val_loss = hist.history['val_loss']
model.reset_states()


# #### smooth validation loss for early stopping parameter determination

# In[9]:


average_over = 15
mov_av = moving_average(np.array(val_loss), average_over)
smooth_val_loss = np.pad(mov_av, int(average_over/2), mode='edge')
epo = np.argmin(smooth_val_loss)


# #### determine model performance for methods comparison 

# In[10]:


hist = model.fit(X_train, y_train, epochs=epo, shuffle=True, batch_size=64, validation_data=(X_test, y_test))
test_loss = hist.history['val_loss']


# In[11]:


fig, ax = plt.subplots(figsize=(16,8))
ax.plot(val_loss, label='validation loss')
ax.plot(smooth_val_loss, label='smooth validation loss')
ax.plot(test_loss, label='test loss')
ax.legend()
plt.show()

