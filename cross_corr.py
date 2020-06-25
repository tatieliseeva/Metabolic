#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 13:30:04 2020

@author: tatiana
"""

import chem_utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, normalize, RobustScaler

shift = chem_utils.chem_shift

data = shift.load_data()
X, y = shift.X_y(data)
X = normalize(X, norm="l2")
#X=MinMaxScaler(feature_range=(0, 100)).fit_transform(X)
#X=RobustScaler().fit_transform(X)

examples = [0, 3, 9, 4]

for x in examples:
    plt.figure()
    plt.plot(X[x])
    

num = 4
comp = X[num]

comp_to_pre = []
for a in X:
    test=np.correlate(comp, a, mode='valid')
    comp_to_pre.append(test)

plt.figure()
for i in range(len(X)):
    if y[i]==0:
        plt.scatter(comp_to_pre[i], i, c='tab:blue')
    elif y[i]==1:
        plt.scatter(comp_to_pre[i], i, c='tab:green')
    elif y[i]==2:
        plt.scatter(comp_to_pre[i],i, c='tab:red')
    else:
        plt.scatter(comp_to_pre[i], i,c='tab:orange')
plt.title("W6 to all")    
to_plot = np.array(comp_to_pre)

plt.figure()
for i in range (len(y)):
    if y[i]==y[num]:
        plt.plot(to_plot[i], c='tab:blue')
    else:
        plt.plot(to_plot[i], c='tab:green')
plt.draw()

plt.figure()
for i in range (len(y)):
    if y[i]==y[num]:
        plt.plot(to_plot[i], c='tab:blue')
plt.draw()  
    
plt.figure()
for i in range (len(y)):
    if y[i]!=y[num]:
        plt.plot(to_plot[i], c='tab:green')
plt.draw() 
#%%

plt.plot(comp_to_pre[0])       
#%%  

test=np.correlate(X[0], X[1], mode='same')
plt.figure()
#plt.plot(X[0], c ='tab:blue')
plt.plot(X[0], c ='tab:red')
plt.figure()
plt.plot(X[1], c ='tab:blue')
plt.figure()
plt.plot(test, c ='tab:orange')
#%%
# 0 weg
# total sum normalizierung
# cross-correlation
