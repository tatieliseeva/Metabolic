#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 21:18:17 2020

@author: tatiana
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import MinMaxScaler, normalize

data = pd.read_csv("data.tsv", sep="\t")

data_clean = np.array(data.drop(["id", "display", "color", "category", "ExpNo", "Class", 
                        "Label", "User", "Experiment", "X1", "X2", "X3", "Animal"], axis=1))

#y = data["Class"].astype('category').cat.codes
y = np.asarray(data["Class"].astype('category').cat.codes)
clusters = list(dict.fromkeys(list(data["Class"])))


scaler = MinMaxScaler((0,1))
scaled_X = scaler.fit_transform(data_clean)
#%% clusterisation X axis
pca = PCA(n_components=4)
X = pca.fit_transform(data_clean.T)
 
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle('PCA axis x')
ax1.plot(X.T[0])
ax2.plot(X.T[1], 'tab:orange')
ax3.plot(X.T[2], 'tab:green')
ax4.plot(X.T[3], 'tab:red')

for ax in fig.get_axes():
    ax.label_outer()
#%% clusterisation Y axis 2 clusters
pca = PCA(n_components=2)
#X = pca.fit_transform(data_clean) 
X = pca.fit_transform(scaled_X) 
colors = ["blue", "green", "red", "yellow"]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle('PCA axis y')

pre = 0
W2 = 0
W6 = 0
W8 = 0

for i in range(len(X)):
    if y[i]==0:
        pre += 1
        ax1.scatter(X[i][0], X[i][1],c ='tab:blue')
    elif y[i]==1:
        W2 += 1
        ax2.scatter(X[i][0], X[i][1],c ='tab:green')
    elif y[i]==2:
        W6 += 1
        ax3.scatter(X[i][0], X[i][1],c ='tab:red')
    else:
        W8 += 1
        ax4.scatter(X[i][0], X[i][1],c ='tab:orange')
        
plt.figure(dpi = 75)
for i in range(len(X)):
    if y[i]==0:
        plt.scatter(X[i][0], X[i][1],c='tab:blue', label='Pre')
    elif y[i]==1:
        plt.scatter(X[i][0], X[i][1],c='tab:green', label='W2')
    elif y[i]==2:
        plt.scatter(X[i][0], X[i][1],c='tab:red', label='W6')
    else:
        plt.scatter(X[i][0], X[i][1],c='tab:orange', label='W8')
plt.title("all_features")
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

#%% clusterisation Y axis 3 clusters

pca = PCA(n_components=3)
X = pca.fit_transform(data_clean)  
colors = ["blue", "green", "red", "yellow"]
 

fig = plt.figure()  
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(X[:,0], X[:,1], X[:,2])

for i in range(len(X)):
    if y[i]==0:
        ax.scatter(X[i][0], X[i][1], X[i][2], c='tab:blue', label='Pre')
    elif y[i]==1:
        ax.scatter(X[i][0], X[i][1], X[i][2], c='tab:green', label='W2')
    elif y[i]==2:
        ax.scatter(X[i][0], X[i][1], X[i][2], c='tab:red', label='W6')
    else:
        ax.scatter(X[i][0], X[i][1], X[i][2], c='tab:orange', label='W8')

plt.title("all_features")

#%% plot all signals

for i in range(len(X)):
    if y[i]==0:
        signal = plt.plot(data_clean[i], c='tab:blue', label='Pre')
    elif y[i]==1:
        signal = plt.plot(data_clean[i], c='tab:green', label='W2')
    elif y[i]==2:
        signal = plt.plot(data_clean[i], c='tab:red', label='W6')
    else:
        signal = plt.plot(data_clean[i], c='tab:orange', label='W8')

# allows do not repeat the lebel        
plt.title("Dataset")
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
