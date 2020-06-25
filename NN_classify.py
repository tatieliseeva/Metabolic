#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:38:45 2020

@author: tatiana

train NN to classify chem shift signal
"""

import chem_utils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

chem = chem_utils.chem_shift

data = chem.load_data()
X,y = chem.X_y(data)
scaler = MinMaxScaler((0,1))
scaled_X = scaler.fit_transform(X)
clusters = list(dict.fromkeys(list(data["Class"])))
sorted_signal = chem.sorted_signal(X,y)
#%%
for s_x in scaled_X:
    plt.plot(s_x)

#%%
for cluster in clusters:
    flat = []
    for y in sorted_signal[cluster]:
        i = 0
        for x in y:
            if x<=0.05:
                i += 1
        flat.append(i)
        
    print(sorted(flat))
    
