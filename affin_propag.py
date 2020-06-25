#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:51:53 2020

@author: tatiana

affinity propagation
"""

import chem_utils
import numpy as np
from scipy.cluster.hierarchy import linkage
import matplotlib.pyplot as plt
from sklearn import metrics
from fcmeans import FCM

chem = chem_utils.chem_shift
data = chem.load_data()
#%%
X, y = chem.X_y(data)
# Affinity propagation
clustering = FCM(n_clusters=3, m=1.9).fit(X.T)
points = clustering.u
## outputs
#fcm_centers = fcm.centers
#fcm_labels  = fcm.u.argmax(axis=1)
X=points
fig = plt.figure()  
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(X[:,0], X[:,1], X[:,2])

for i in range(len(X)):
    ax.scatter(X[i][0], X[i][1], X[i][2], c='tab:orange', label='W8')

plt.title("all_features")
#%%


X=points
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

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle('Fuzzy C-Means')

for i in range(len(X)):
    if y[i]==0:
        ax1.scatter(X[i][0], X[i][1], c ='tab:blue')
    elif y[i]==1:
        ax2.scatter(X[i][0], X[i][1], c ='tab:green')
    elif y[i]==2:
        ax3.scatter(X[i][0], X[i][1], c ='tab:red')
    else:
        ax4.scatter(X[i][0], X[i][1], c ='tab:orange')
        

#%%
print(clustering)
print("___________")
print(y)
#print(clustering.predict(X))
#print(clustering.cluster_centers_)
print(metrics.adjusted_rand_score(y, clustering.u))
#%%
'''
a = to_plot[:,:20]
plt.imshow(a, interpolation='nearest')
plt.xticks([])
plt.yticks([0,1],["Affine","Original"])
plt.show()

a = to_plot[:,20:40]
plt.imshow(a, interpolation='nearest')
plt.xticks([])
plt.yticks([0,1],["Affine","Original"])
plt.show()

a = to_plot[:,40:]
plt.imshow(a, interpolation='nearest')
plt.xticks([])
plt.yticks([0,1],["Affine","Original"])
plt.show()
'''