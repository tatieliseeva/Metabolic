#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 10:34:04 2020

@author: tatiana

feature extraction with NN
"""

import chem_utils
from keras.applications.resnet50 import ResNet50
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

chem = chem_utils.chem_shift

model = ResNet50(weights="imagenet")

data = chem.load_data()
X,y = chem.X_y(data)
scaler = MinMaxScaler((0,255))
X = scaler.fit_transform(X)
clusters = list(dict.fromkeys(list(data["Class"])))
sorted_signal = chem.sorted_signal(X,y)
#%%
labels_colors = {"Pre":"blue", "W2":"orange", "W6":"green", "W8":"red"}
#%%
#feature extraction of a signal with ResNet50 
out_array_thr = []
for lc in labels_colors:

    s_array = sorted_signal[str(lc)]
    
    # modify array for the net input
    out_array = []   
    for x in s_array:
        new_x = []
        for i in range(len(x)):
            new_x.append(x)
    
        end_x = np.resize(np.array(new_x),(224,224))
        end_x = np.stack([end_x]*3, axis=-1)
        out_array.append(end_x)   
    out_array = np.array(out_array)
    
    # feature extraction and plot result
    i = 0
    plt.figure( dpi = 800)
    for x in out_array:
        first_out = model.predict(x[np.newaxis,:,:,:])
        
        y = []
        xx = []
        out1 = []
        
        for yy in first_out[0]:
            if yy > 1/750:
                y.append(yy)
            else:
                y.append(0)
        [xx.append(x) for x in range(len(y))]
        
        [out1.append([xx[i], y[i]]) for i in range(len(y))]
        non_zero_features = []
        for x in out1 :
            if x[1] != 0:
                non_zero_features.append(x)
                
        non_zero_features = np.array(non_zero_features)
        
        plt.scatter(*zip(*non_zero_features),color=labels_colors[lc], label=str(lc), alpha=0.5)
        i += 1
        
        out_array_thr.append(y)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    

sums_points = []
for g in range(1000):
    point_counter = 0
    for i in range(59):
        if out_array_thr[i][g] > 0:
            point_counter += 1
    sums_points.append(point_counter)
    if point_counter >= 58:
        print(g)
#%%
cluster_coords = [488, 696, 703, 733, 868, 904]
for x in cluster_coords:
    total_sum = 0
    for xx in out_array_thr:
        total_sum += xx[x]
    print(x, total_sum)
    
#%%
result_coords = [488, 696, 868, 904]

coords = {}
for x in result_coords:
    coords[x] = {}
    for xx in result_coords:
        coords[x][xx] = []
        for i in range (len(out_array_thr)):
            coords[x][xx].append([out_array_thr[i][x],out_array_thr[i][xx]])
#%%
y = sorted_y
for x in coords:
    for xx in coords:
        XXXx = coords[x][xx]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fig.suptitle(str(x) + "-" + str(xx) )
        

        for i in range(len(XXXx)):
            if y[i]==0:
                ax1.scatter(XXXx[i][0], XXXx[i][1], c ='tab:blue')
            elif y[i]==1:
                ax2.scatter(XXXx[i][0], XXXx[i][1], c ='tab:green')
            elif y[i]==2:
                ax3.scatter(XXXx[i][0], XXXx[i][1], c ='tab:red')
            else:
                ax4.scatter(XXXx[i][0], XXXx[i][1], c ='tab:orange')
        
        plt.figure()
        for i in range(len(XXXx)):
            if y[i]==0:
                plt.scatter(XXXx[i][0], XXXx[i][1], c='tab:blue')
            elif y[i]==1:
                plt.scatter(XXXx[i][0], XXXx[i][1], c='tab:green')
            elif y[i]==2:
                plt.scatter(XXXx[i][0], XXXx[i][1], c='tab:red')
            else:
                plt.scatter(XXXx[i][0], XXXx[i][1], c='tab:orange')
        plt.title(str(x) + "-" + str(xx))
        plt.savefig(str(x) + "-" + str(xx) + ".png")

        
        
