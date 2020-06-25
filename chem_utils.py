#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 10:35:01 2020

@author: tatiana

utils chemical shift analysis
"""
import pandas as pd
import numpy as np

class chem_shift:
    
    def __init__(self):
        self.names = ["Pre", "W2", "W6", "W8"]
        self.labels_colors = {"Pre":"blue", "W2":"orange", "W6":"green", "W8":"red"}
        
    def load_data (file="data.tsv"):
        '''
        loads data from .tsv file
         IN: filename, str
         OUT: all data, pandas dataframe
         '''
        return pd.read_csv(file, sep="\t")
    
    def X_y (data):
        '''
        Cleans up the dataframe
        IN: all data, pandas dataframe
        OUT: data and targets, numpy array
        '''
                
        y = np.asarray(data["Class"].astype('category').cat.codes)
        X = np.array(data.drop(["id", "display", "color", "category", "ExpNo", "Class", 
                        "Label", "User", "Experiment", "X1", "X2", "X3", "Animal"], axis=1))
        
        return X[:,20:], y
    
    def sorted_signal(X, y):
        Pre, W2, W6, W8 = [], [], [], []
        for i in range(len(X)):
            if y[i]==0:
                Pre.append(X[i,:])
            elif y[i]==1:
                W2.append(X[i,:])
            elif y[i]==2:
                W6.append(X[i,:])
            else:
                W8.append(X[i,:])
        
        return {"Pre":np.array(Pre), "W2":np.array(W2), "W6":np.array(W6), "W8":np.array(W8)}
    
    
    
    
    '''
    y = np.asarray(data["Class"].astype('category').cat.codes)
    data_clean = np.array(data.drop(["id", "display", "color", "category", "ExpNo", "Class", 
                        "Label", "User", "Experiment", "X1", "X2", "X3", "Animal"], axis=1))
    
    '''
#%%
#y = data["Class"].astype('category').cat.codes
#names = list(data["Class"])
