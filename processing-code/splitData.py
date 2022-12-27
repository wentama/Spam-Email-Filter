# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 00:09:02 2022

@author: wtl22
"""

import numpy as np
from sklearn.model_selection import train_test_split

data = np.loadtxt('spamTrain1.csv', delimiter=',')
train, test = train_test_split(data, test_size=0.2, stratify=data[:,30], random_state=1)
np.savetxt("train.csv", train, delimiter = ",")
np.savetxt("test.csv", test, delimiter = ",")


#%%
import numpy as np
from sklearn.model_selection import train_test_split

data_new= np.loadtxt('spamTrain3.csv', delimiter=',')
train_new, test_new = train_test_split(data_new, test_size=0.2, random_state=1)
np.savetxt("train_new.csv", train_new, delimiter = ",")
np.savetxt("test_new.csv", test_new, delimiter = ",")

#%%
unique1, counts1 = np.unique(train_new, return_counts=True)
unique2, counts2 = np.unique(test_new, return_counts=True)

#%%
import numpy as np
from sklearn.model_selection import train_test_split

train1DataFilename = 'spamTrain1.csv'
train2DataFilename = 'spamTrain2.csv'

train1Data = np.loadtxt(train1DataFilename,delimiter=',')
train2Data = np.loadtxt(train2DataFilename,delimiter=',')
trainData = np.r_[train1Data,train2Data]

train, test = train_test_split(trainData, test_size=0.1, random_state=1)
np.savetxt("train_3.csv", train, delimiter = ",")
np.savetxt("test_3.csv", test, delimiter = ",")

