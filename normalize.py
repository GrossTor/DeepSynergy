#!/usr/bin/env python
# coding: utf-8

# ## Normalization Script
# 
# Author: Kristina Preuer
# 
# This script shows how the data was split and how the features were normalized. The data is then saved in a pickle file. Which will be loaded during the cross validation procedure.

# In[1]:


import numpy as np
import pandas as pd
import pickle 
import gzip


# ##### Define the parameters for data generation: folds for testing and validation and normalization strategy

# In[2]:


# in this example tanh normalization is used
# fold 0 is used for testing and fold 1 for validation (hyperparamter selection)
norm = 'tanh'
test_fold = 0
val_fold = 1


# #### Define nomalization function
# It normalizes the input data X. If X is used for training the mean and the standard deviation is calculated during normalization. If X is used for validation or testing, the previously calculated mean and standard deviation of the training data should be used. If "tanh_norm" is used as normalization strategy, then the mean and standard deviation are calculated twice. The features with a standard deviation of 0 are filtered out. 

# In[3]:


def normalize(X, means1=None, std1=None, means2=None, std2=None, feat_filt=None, norm='tanh_norm'):
    if std1 is None:
        std1 = np.nanstd(X, axis=0)
    if feat_filt is None:
        feat_filt = std1!=0
    X = X[:,feat_filt]
    X = np.ascontiguousarray(X)
    if means1 is None:
        means1 = np.mean(X, axis=0)
    X = (X-means1)/std1[feat_filt]
    if norm == 'norm':
        return(X, means1, std1, feat_filt)
    elif norm == 'tanh':
        return(np.tanh(X), means1, std1, feat_filt)
    elif norm == 'tanh_norm':
        X = np.tanh(X)
        if means2 is None:
            means2 = np.mean(X, axis=0)
        if std2 is None:
            std2 = np.std(X, axis=0)
        X = (X-means2)/std2
        X[:,std2==0]=0
        return(X, means1, std1, means2, std2, feat_filt)        


# #### Load features and labels

# In[4]:


#contains the data in both feature ordering ways (drug A - drug B - cell line and drug B - drug A - cell line)
#in the first half of the data the features are ordered (drug A - drug B - cell line)
#in the second half of the data the features are ordered (drug B - drug A - cell line)
file = gzip.open('X.p.gz', 'rb')
X = pickle.load(file)
file.close()


# In[5]:


#contains synergy values and fold split (numbers 0-4)
labels = pd.read_csv('labels.csv', index_col=0) 
#labels are duplicated for the two different ways of ordering in the data
labels = pd.concat([labels, labels]) 


# #### Define indices for splitting

# In[6]:


#indices of training data for hyperparameter selection: fold 2, 3, 4
idx_tr = np.where(np.logical_and(labels['fold']!=test_fold, labels['fold']!=val_fold))
#indices of validation data for hyperparameter selection: fold 1
idx_val = np.where(labels['fold']==val_fold)


# In[7]:


#indices of training data for model testing: fold 1, 2, 3, 4
idx_train = np.where(labels['fold']!=test_fold)
#indices of test data for model testing: fold 0
idx_test = np.where(labels['fold']==test_fold)


# #### Split data 

# In[8]:


X_tr = X[idx_tr]
X_val = X[idx_val]
X_train = X[idx_train]
X_test = X[idx_test]


# In[9]:


y_tr = labels.iloc[idx_tr]['synergy'].values
y_val = labels.iloc[idx_val]['synergy'].values
y_train = labels.iloc[idx_train]['synergy'].values
y_test = labels.iloc[idx_test]['synergy'].values


# #### Normalize training and validation data for hyperparameter selection

# In[10]:


if norm == "tanh_norm":
    X_tr, mean, std, mean2, std2, feat_filt = normalize(X_tr, norm=norm)
    X_val, mean, std, mean2, std2, feat_filt = normalize(X_val, mean, std, mean2, std2, 
                                                          feat_filt=feat_filt, norm=norm)
else:
    X_tr, mean, std, feat_filt = normalize(X_tr, norm=norm)
    X_val, mean, std, feat_filt = normalize(X_val, mean, std, feat_filt=feat_filt, norm=norm)


# #### Normalize training and test data for methods comparison

# In[11]:


if norm == "tanh_norm":
    X_train, mean, std, mean2, std2, feat_filt = normalize(X_train, norm=norm)
    X_test, mean, std, mean2, std2, feat_filt = normalize(X_test, mean, std, mean2, std2, 
                                                          feat_filt=feat_filt, norm=norm)
else:
    X_train, mean, std, feat_filt = normalize(X_train, norm=norm)
    X_test, mean, std, feat_filt = normalize(X_test, mean, std, feat_filt=feat_filt, norm=norm)


# #### Save data as pickle file

# In[12]:


pickle.dump((X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test), 
            open('data_test_fold%d_%s.p'%(test_fold, norm), 'wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




