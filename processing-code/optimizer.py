# -*- coding: utf-8 -*-
"""
File containing hyperparameter tunning code and RandomForest classifier testing
"""

#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import decomposition, datasets
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#%%
data = np.loadtxt('train_new.csv',delimiter=',')

X = data[:,:-1]
y = data[:,-1]

 #%%


pipe_svc = make_pipeline(SVC(random_state=1))
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0,10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0]
param_grid = [{'svc__C': param_range,'svc__gamma': param_range,'svc__kernel': ['rbf']}]
gs = GridSearchCV(estimator=pipe_svc,param_grid=param_grid,scoring='accuracy',cv=10,refit=True,n_jobs=-1)
gs = gs.fit(X, y)
print(gs.best_score_)
print(gs.best_params_)

#%%
model = RandomForestClassifier(n_estimators = 2000, min_samples_split=4, min_samples_leaf=1, max_depth=15, bootstrap=False, random_state=42)
sbs = SequentialFeatureSelector(estimator = model, n_features_to_select = 'auto', n_jobs = -1, direction = 'backward')
sbs.fit(X, y)

#%%
train = pd.read_csv('train.csv', header = None)
trainLabels = train[30].values
trainFeatures = train.drop(30, axis=1).values

test = pd.read_csv('test.csv', header = None)
testLabels = test[30].values
testFeatures = test.drop(30, axis=1).values

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestClassifier(random_state = 42)

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

rf_random.fit(trainFeatures, trainLabels)

rf_random.best_params_

#%%
train = pd.read_csv('train_new.csv', header = None)
trainLabels = train[30].values
trainFeatures = train.drop(30, axis=1).values

test = pd.read_csv('test_new.csv', header = None)
testLabels = test[30].values
testFeatures = test.drop(30, axis=1).values

def decisionTree(trainFeatures, trainLabels, testFeatures, testLabels):
    clf = RandomForestClassifier(n_estimators = 2000, min_samples_split=4, min_samples_leaf=1, max_depth=15, bootstrap=False, random_state=42)
    clf.fit(trainFeatures, trainLabels)
    prediction = clf.predict(testFeatures)
    confusionMatrix = confusion_matrix(testLabels, prediction)
    matrix = pd.DataFrame(confusionMatrix)
    ax = plt.axes()
    sns.set(font_scale=1.3)
    plt.figure(figsize=(10,7))
    sns.heatmap(matrix, annot=True, fmt="g", ax=ax, cmap="magma")
    ax.set_title('Confusion Matrix - Random Forest Classifier')
    ax.set_xlabel("Predicted label", fontsize =15)
    #ax.set_xticklabels(['']+labels)
    ax.set_ylabel("True Label", fontsize=15)
    #ax.set_yticklabels(list(labels), rotation = 0)Z          
    
decisionTree(trainFeatures, trainLabels, testFeatures, testLabels)

#%% 
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier

data = np.loadtxt('train_2.csv', delimiter=',')
X = data[:,:-1]
y = data[:,-1]

X_s = np.delete(X, [1, 2, 5, 9, 11, 12, 13, 14, 15, 17, 18, 20, 21, 23, 24], 1)

svm = SVC(kernel='rbf', C=100, gamma=30, random_state=1, probability=True)
bag= BaggingClassifier(base_estimator=svm, n_jobs=-1, random_state=1)

base = [bag]
n_estimators = [10, 50, 100, 500, 1000, 1500, 2000]
max_samples = [1, 2, 5, 10, 15, 20]
max_features = [1, 2, 5, 10, 15]
boole = [True, False]
param_grid = {'base_estimator': base,
              'n_estimators' : n_estimators, 
              'max_samples':max_samples,
              'max_features': max_features,
              'bootstrap' : boole,
              'bootstrap_features' : boole
              }

gs = GridSearchCV(estimator=bag, param_grid=param_grid, scoring='accuracy', cv=10, refit=True, n_jobs=-1)

gs = gs.fit(X_s, y)


              
