#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:48:09 2020

@author: tatiana
"""
from sklearn.cluster import SpectralClustering
from sklearn.metrics import accuracy_score
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv("data.tsv", sep="\t")

data_clean = np.array(data.drop(["id", "display", "color", "category", "ExpNo", "Class", 
                        "Label", "User", "Experiment", "X1", "X2", "X3", "Animal"], axis=1))

#y = data["Class"].astype('category').cat.codes
y = np.asarray(data["Class"].astype('category').cat.codes)
clusters = list(dict.fromkeys(list(data["Class"])))

spectral_clustering = SpectralClustering(n_clusters=2,assign_labels="discretize",
                                         random_state=0, n_components=4).fit(data_clean)

X = spectral_clustering.fit_predict(data_clean)

to_plot = np.asarray([X, y])
#%%
ra = []
for i in range(len(y)):
    
    if int(y[i]) == int(X[i]):
        print(i, y[i], X[i])
        ra.append([i, y[i], X[i]])
        
er_s = len(ra)/len(y)

error = accuracy_score(np.asarray(y), np.asarray(X))


#%%
a = to_plot[:,40:]
plt.imshow(a, interpolation='nearest')
plt.xticks([])
plt.yticks([0,1],["SC","Original"])
plt.show()


