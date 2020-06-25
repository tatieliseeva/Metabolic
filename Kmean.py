#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 20:30:22 2020

@author: tatiana
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import MinMaxScaler, normalize

data = pd.read_csv("data.tsv", sep="\t")

data_clean = np.array(data.drop(["id", "display", "color", "category", "ExpNo", "Class", 
                        "Label", "User", "Experiment", "X1", "X2", "X3", "Animal"], axis=1))

#y = data["Class"].astype('category').cat.codes
y = np.asarray(data["Class"].astype('category').cat.codes)
clusters = list(dict.fromkeys(list(data["Class"])))

scaled_X = normalize(data_clean)
#%%
kmeans = KMeans(n_clusters=2, max_iter=100000, random_state=42).fit(data_clean)


X = kmeans.transform(scaled_X)

#%%
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle('KMeans++')

for i in range(len(X)):
    if y[i]==0:
        ax1.scatter(X[i][0], X[i][1], c ='tab:blue')
    elif y[i]==1:
        ax2.scatter(X[i][0], X[i][1], c ='tab:green')
    elif y[i]==2:
        ax3.scatter(X[i][0], X[i][1], c ='tab:red')
    else:
        ax4.scatter(X[i][0], X[i][1], c ='tab:orange')

        
plt.figure()
for i in range(len(X)):
    if y[i]==0:
        plt.scatter(X[i][0], X[i][1], c='tab:blue')
    elif y[i]==1:
        plt.scatter(X[i][0], X[i][1], c='tab:green')
    elif y[i]==2:
        plt.scatter(X[i][0], X[i][1], c='tab:red')
    else:
        plt.scatter(X[i][0], X[i][1], c='tab:orange')
plt.title("all_features")















