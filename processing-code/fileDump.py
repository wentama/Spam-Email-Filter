# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 00:48:08 2022

@author: wtl22
"""

#%%
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#%%
 
 model_1 = RandomForestClassifier(n_estimators = 2000, min_samples_split=4, min_samples_leaf=1, max_depth=15, bootstrap=False, random_state=42)
 model_2 = SVC(kernel='rbf', C=100, gamma=30, random_state=1, probability=True)

 e_model = VotingClassifier(estimators=[('rf', model_1), ('rbf', model_2)], voting='soft')
 e_model = e_model.fit(new_trainFeatures, trainLabels)
 testOutputs = e_model.predict_proba(new_testFeatures)[:,1]

 new_trainFeatures = trainFeatures[:,[ 0,  2,  3,  4,  6,  7,  8, 12, 16, 19, 21, 22, 24, 28, 29]]
 new_testFeatures = testFeatures[:,[ 0,  2,  3,  4,  6,  7,  8, 12, 16, 19, 21, 22, 24, 28, 29]]
 
 v1_trainFeatures = np.delete(trainFeatures, [11], 1)
 v1_testFeatures = np.delete(testFeatures, [11], 1)


 v1_trainFeatures = np.delete(trainFeatures, [11], 1)
 v1_testFeatures = np.delete(testFeatures, [11], 1)
 
 model = RandomForestClassifier(n_estimators = 200, min_samples_split=5, min_samples_leaf=1, max_depth=15, bootstrap=True, random_state=42)
 model.fit(v1_trainFeatures, trainLabels)
 testOutputs = model.predict_proba(v1_testFeatures)[:,1]


v1_trainFeatures = np.delete(trainFeatures, [1, 2, 5, 9, 11, 12, 13, 14, 15, 17, 18, 20, 21, 23, 24], 1)
v1_testFeatures = np.delete(testFeatures, [1, 2, 5, 9, 11, 12, 13, 14, 15, 17, 18, 20, 21, 23, 24], 1)

svm = SVC(kernel='rbf', C=100, gamma=30, random_state=1, probability=True)
svm.fit(v1_trainFeatures,trainLabels)
testOutputs = svm.predict_proba(v1_testFeatures)[:,1]

#%%
v1_trainFeatures = np.delete(trainFeatures, [1, 2, 5, 9, 11, 12, 13, 14, 15, 17, 18, 20, 21, 23, 24], 1)
v1_testFeatures = np.delete(testFeatures, [1, 2, 5, 9, 11, 12, 13, 14, 15, 17, 18, 20, 21, 23, 24], 1)

model = RandomForestClassifier(criterion='gini', n_estimators = 2000, min_samples_split=4, min_samples_leaf=1, max_depth=15, bootstrap=False, random_state=42)
model.fit(trainFeatures, trainLabels)
testOutputs = model.predict_proba(testFeatures)[:,1]
'''
'''
svm = SVC(kernel='rbf', C=100, gamma=30, random_state=1, probability=True)
svm.fit(v1_trainFeatures,trainLabels)
testOutputs = svm.predict_proba(v1_testFeatures)[:,1]

 v2_trainFeatures = np.delete(trainFeatures, [1,2,11,12,13,14,15,16,17,18,20,21,23,24,25], 1)
 v2_testFeatures = np.delete(testFeatures, [1,2,11,12,13,14,15,16,17,18,20,21,23,24,25], 1)
 
 v3_trainFeatures = np.delete(trainFeatures, [1,5,6,11,12,13,15,16,17,18,20,21,23,24,25],1)
 v3_testFeatures = np.delete(testFeatures, [1,5,6,11,12,13,15,16,17,18,20,21,23,24,25],1)
 
 #%%
 new_trainFeatures = np.delete(trainFeatures, [1, 2, 5, 9, 11, 12, 13, 14, 15, 17, 18, 20, 21, 23, 24], 1)
 new_testFeatures = np.delete(testFeatures, [1, 2, 5, 9, 11, 12, 13, 14, 15, 17, 18, 20, 21, 23, 24], 1)
 svm = SVC(kernel='rbf', C=100, gamma=100, random_state=1, probability=True)
 svm.fit(new_trainFeatures,trainLabels)
 testOutputs = svm.predict_proba(new_testFeatures)[:,1]

     
 model = SVC(kernel='rbf', random_state=1, probability=True)
 model.fit(trainFeatures,trainLabels)
     
 # Use predict_proba() rather than predict() to use probabilities rather
 # than estimated class labels as outputs
 testOutputs = model.predict_proba(testFeatures)[:,1]
     
 return testOutputs

#%%
 # svm std
 svm.fit(X_train_std,y_train)
 testOutputs_std = svm.predict_proba(X_test_std)[:,1]
 print("Test set AUC svm_std: ", roc_auc_score(y_test,testOutputs_std))
 
 # svm norm
 svm.fit(X_train_norm, y_train)
 testOutputs_norm = svm.predict_proba(X_test_norm)[:,1]
 print("Test set AUC svm_norm: ", roc_auc_score(y_test,testOutputs_norm))

 # calling the def 
 testOutputs = predictTest(X_train,y_train,X_test)
 print("Test set AUC: ", roc_auc_score(y_test,testOutputs))
 
 # svm std
 svm.fit(X_train_std,y_train)
 testOutputs_std = svm.predict_proba(X_test_std)[:,1]
 print("Test set AUC svm_std: ", roc_auc_score(y_test,testOutputs_std))
 
 # svm norm
 svm.fit(X_train_norm, y_train)
 testOutputs_norm = svm.predict_proba(X_test_norm)[:,1]
 print("Test set AUC svm_norm: ", roc_auc_score(y_test,testOutputs_norm))

 # calling the def 
 testOutputs = predictTest(X_train,y_train,X_test)
 print("Test set AUC: ", roc_auc_score(y_test,testOutputs))
 
 #%%
 rf = RandomForestClassifier(max_depth=10, random_state=1)
 scores = cross_val_score(rf,X,y,cv=10,scoring='roc_auc')
 print("10-fold cross-validation mean AUC rf: ",
       np.mean(scores))
 
 svm_rbf =  SVC(kernel='rbf', random_state=1)
 scores = cross_val_score(svm_rbf,X,y,cv=10,scoring='roc_auc')
 print("10-fold cross-validation mean AUC svm rbf: ",
       np.mean(scores))
 
 svm_linear =  SVC(kernel='linear', random_state=1)
 scores = cross_val_score(svm_linear,X,y,cv=10,scoring='roc_auc')
 print("10-fold cross-validation mean AUC svm linear: ",
       np.mean(scores))
 
 svm_poly =  SVC(kernel='poly', random_state=1)
 scores = cross_val_score(svm_poly,X,y,cv=10,scoring='roc_auc')
 print("10-fold cross-validation mean AUC svm poly: ",
       np.mean(scores))
 
 svm_sigmoid =  SVC(kernel='sigmoid', random_state=1)
 scores = cross_val_score(svm_sigmoid,X,y,cv=10,scoring='roc_auc')
 print("10-fold cross-validation mean AUC svm sigmoid: ",
       np.mean(scores))
 
 tree = DecisionTreeClassifier(criterion='gini', max_depth = 6, random_state=1)
 scores = cross_val_score(tree,X,y,cv=10,scoring='roc_auc')
 print("10-fold cross-validation mean AUC tree: ",
       np.mean(scores))
 
 knn = KNeighborsClassifier(n_neighbors=30, p=2, metric='minkowski')
 scores = cross_val_score(knn,X,y,cv=10,scoring='roc_auc')
 print("10-fold cross-validation mean AUC knn: ",
       np.mean(scores))
 
 lr = LogisticRegression(C=100, solver='lbfgs', multi_class='ovr', random_state = 1)
 scores = cross_val_score(lr,X,y,cv=10,scoring='roc_auc')
 print("10-fold cross-validation mean AUC lr: ",
       np.mean(scores))
 
 pca = PCA()
 lr = LogisticRegression(C=100, solver='lbfgs', multi_class='ovr', random_state = 1)
 X_pca = pca.fit_transform(X)
 lr.fit(X_pca, y)
 scores = cross_val_score(lr,X_pca,y,cv=10,scoring='roc_auc')
 print("10-fold cross-validation mean AUC lr pca: ",
       np.mean(scores))
 
 lda = LDA()
 X_lda = lda.fit_transform(X, y)
 lr = LogisticRegression(C=100, solver='lbfgs', multi_class='ovr', random_state = 1)
 lr.fit(X_lda, y)
 scores = cross_val_score(lr,X_lda,y,cv=10,scoring='roc_auc')
 print("10-fold cross-validation mean AUC lr lda: ",
       np.mean(scores))
 
 #%%
  from sklearn.model_selection import RandomizedSearchCV
  from sklearn.pipeline import make_pipeline
  
  pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
  
  param_grid = [{'svc__C': param_range,'svc__kernel': ['linear']}, {'svc__C': param_range, 'svc__gamma': param_range, 'svc__kernel': ['rbf']}]
  rs = RandomizedSearchCV(estimator=pipe_svc, param_distributions=param_grid, scoring='accuracy', refit=True, n_iter=20,cv=10,random_state=1,n_jobs=-1)
  rs = rs.fit(X, y)
  print(rs.best_score_)
  print(rs.best_params_)
  
  #%%
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
  
  #%%
  import numpy as np
  import matplotlib.pyplot as plt
  from sklearn.model_selection import cross_val_score
  from sklearn.metrics import roc_auc_score
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.feature_selection import SequentialFeatureSelector
  from sklearn.ensemble import RandomForestClassifier


  def aucCV(features,labels):
      model = RandomForestClassifier(criterion='gini', n_estimators = 2000, min_samples_split=4, min_samples_leaf=1, max_depth=15, bootstrap=False, random_state=42)
      scores = cross_val_score(model,features,labels,cv=10,scoring='roc_auc')
      
      return scores

  def predictTest(trainFeatures,trainLabels,testFeatures):
      
      model = RandomForestClassifier(criterion='gini', n_estimators = 2000, min_samples_split=4, min_samples_leaf=1, max_depth=15, bootstrap=False, random_state=42)
      model.fit(trainFeatures, trainLabels)
      testOutputs = model.predict_proba(testFeatures)[:,1]
      
     
      return testOutputs
