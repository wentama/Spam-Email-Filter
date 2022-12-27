# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 20:42:09 2022

@author: wtl22
"""
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

data = np.loadtxt('train_2.csv', delimiter=',')

X = data[:,:-1]
y = data[:,-1]

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
evr = np.cumsum(pca.explained_variance_ratio_)
print(evr)

plt.plot(range(1,len(evr)+1),evr)
plt.xticks(range(1,len(evr)+1))
plt.title("Explained variance ratio")
plt.ylabel("Explained variance ratio")
plt.xlabel("n_components")
plt.show()
#%%
X_pca=pd.DataFrame(X_pca)
X_pca.columns=["pc1","pc2"]
X_pca["y"]=y

fig = px.scatter(X_pca, x='pc1', y='pc2',color='y',title="Iris 3D")
fig.update_traces(marker_coloraxis=None)
fig.show()

#%%
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
Xt = pca.fit_transform(X)
plot = plt.scatter(Xt[:,0], Xt[:,1], c=y)
plt.legend(handles=plot.legend_elements()[0], labels=list(data[30]))
plt.show()

#%%
