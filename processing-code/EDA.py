# -*- coding: utf-8 -*-
"""
File containing the EDA code
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from __future__ import print_function
import time
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

%matplotlib inline

#%%
train = pd.read_csv('train.csv', header = None)
trainLabels = train[30].values
trainFeatures = train.drop(30, axis=1).values

test = pd.read_csv('test.csv', header = None)
testLabels = test[30].values
testFeatures = test.drop(30, axis=1).values

pca = PCA(0.95)
components = pca.fit_transform(trainFeatures)
#labels = [0,1]
fig = px.scatter(components, x=0, y=1, color=trainLabels)
fig.show()

#%%
def decisionTree(trainFeatures, trainLabels, testFeatures, testLabels):
    clf = RandomForestClassifier(criterion='gini', n_estimators = 2000, min_samples_split=4, min_samples_leaf=1, max_depth=15, bootstrap=False, random_state=42)
    clf.fit(trainFeatures, trainLabels)
    prediction = clf.predict(testFeatures)
    confusion_matrix = metrics.confusion_matrix(testLabels, prediction)
    matrix = pd.DataFrame(confusion_matrix)
    ax = plt.axes()
    sns.set(font_scale=1.3)
    plt.figure(figsize=(10,7))
    sns.heatmap(matrix, annot=True, fmt="g", ax=ax, cmap="magma")
    ax.set_title('Confusion Matrix - Random Forest')
    ax.set_xlabel("Predicted label", fontsize =15)
    #ax.set_xticklabels(['']+labels)
    ax.set_ylabel("True Label", fontsize=15)
    #ax.set_yticklabels(list(labels), rotation = 0)
    plt.show()
    
#%%
trainFeatures = np.delete(trainFeatures, [11], 1)
testFeatures = np.delete(testFeatures, [11], 1)

#%%
decisionTree(trainFeatures, trainLabels, testFeatures, testLabels)

#%%
pca = PCA(n_components=3)
pca_result = pca.fit_transform(trainFeatures)
train['pca-one'] = pca_result[:,0]
train['pca-two'] = pca_result[:,1] 
train['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

#%%
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    
    palette=sns.color_palette("hls", 10),
    data=train,
    legend="full",
    alpha=0.3
)
