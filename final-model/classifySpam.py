# -*- coding: utf-8 -*-
"""
Using the baseline classification algorithm given by Professor Kevin S. Xu

@Edited by: Wen Tao Lin & Mason Leung

The code for data Preprocessing, hyperparameter tuning, and other steps we took are in a separate file
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier


def aucCV(features,labels):
    
    v1_features = np.delete(features, [11], 1)
    
    model = RandomForestClassifier(criterion='gini', n_estimators = 2000, min_samples_split=4, min_samples_leaf=1, max_depth=15, bootstrap=False, random_state=42)
    scores = cross_val_score(model,v1_features,labels,cv=10,scoring='roc_auc')
    
    return scores

def predictTest(trainFeatures,trainLabels,testFeatures):
    
    v1_trainFeatures = np.delete(trainFeatures, [11], 1)
    v1_testFeatures = np.delete(testFeatures, [11], 1)
       
    model = RandomForestClassifier(criterion='gini', n_estimators = 2000, min_samples_split=4, min_samples_leaf=1, max_depth=15, bootstrap=False, random_state=42)
    model.fit(v1_trainFeatures, trainLabels)
    testOutputs = model.predict_proba(v1_testFeatures)[:,1]
    
   
    return testOutputs
    
# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    data = np.loadtxt('spamTrain1.csv',delimiter=',')
    # Randomly shuffle rows of data set then separate labels (last column)
    shuffleIndex = np.arange(np.shape(data)[0])
    np.random.shuffle(shuffleIndex)
    data = data[shuffleIndex,:]
    features = data[:,:-1]
    labels = data[:,-1]
    
    # Evaluating classifier accuracy using 10-fold cross-validation
    print("10-fold cross-validation mean AUC: ",
          np.mean(aucCV(features,labels)))
    
    # Arbitrarily choose all odd samples as train set and all even as test set
    # then compute test set AUC for model trained only on fixed train set
    trainFeatures = features[0::2,:]
    trainLabels = labels[0::2]
    testFeatures = features[1::2,:]
    testLabels = labels[1::2]
    testOutputs = predictTest(trainFeatures,trainLabels,testFeatures)
    print("Test set AUC: ", roc_auc_score(testLabels,testOutputs))
    
    # Examine outputs compared to labels
    sortIndex = np.argsort(testLabels)
    nTestExamples = testLabels.size
    plt.subplot(2,1,1)
    plt.plot(np.arange(nTestExamples),testLabels[sortIndex],'b.')
    plt.xlabel('Sorted example number')
    plt.ylabel('Target')
    plt.subplot(2,1,2)
    plt.plot(np.arange(nTestExamples),testOutputs[sortIndex],'r.')
    plt.xlabel('Sorted example number')
    plt.ylabel('Output (predicted target)')
    