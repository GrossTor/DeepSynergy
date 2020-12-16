#!/usr/bin/env python
# coding: utf-8

# ## DeepSynergy
# 
# Author: Kristina Preuer
# 
# This Keras script shows how DeepSynergy was evaluated in one cross validation run (executed 5 times - looping over test folds). In this examples fold 0 is used for testing. The script uses 60% of the data  for training (folds 2, 3, 4) and 20% for validation (fold 1). The parameters are loaded with a separate text file (hyperparameters). Validation loss was used to determine the early stopping parameter. After hyperparameter selection the training and validation data was combined (80% = folds 1, 2, 3, 4) and the remaining 20% (fold 0) of the data were used for testing.
# 
# The original work was done accordingly with binet (https://github.com/bioinf-jku/binet/tree/master/binet). 

# In[1]:


import os, sys

import pandas as pd
import numpy as np
import pickle
import gzip

import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="3" #specify GPU 
import keras as K
import tensorflow as tf
from keras import backend
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential
from keras.layers import Dense, Dropout


# #### Define parameters for this cross-validation run

# In[2]:


hyperparameter_file = 'hyperparameters' # textfile which contains the hyperparameters of the model
data_file = 'data_test_fold0_tanh.p.gz' # pickle file which contains the data (produced with normalize.ipynb)


# #### Define smoothing functions for early stopping parameter

# In[3]:


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# #### Load parameters defining the model

# In[4]:


exec(open(hyperparameter_file).read()) 


# #### Load data 
# tr = 60% of data for training during hyperparameter selection <br>
# val = 20% of data for validation during hyperparameter selection
# 
# train = tr + val = 80% of data for training during final testing <br>
# test = remaining left out 20% of data for unbiased testing 
# 
# splitting and normalization was done with normalize.ipynb

# In[5]:


file = gzip.open(data_file, 'rb')
X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test = pickle.load(file)
file.close()


# #### run set

# In[6]:


config = tf.ConfigProto(
         allow_soft_placement=True,
         gpu_options = tf.GPUOptions(allow_growth=True))
set_session(tf.Session(config=config))


# In[7]:


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


# #### plot performance 

# In[11]:


fig, ax = plt.subplots(figsize=(16,8))
ax.plot(val_loss, label='validation loss')
ax.plot(smooth_val_loss, label='smooth validation loss')
ax.plot(test_loss, label='test loss')
ax.legend()
plt.show()


# In[ ]:




