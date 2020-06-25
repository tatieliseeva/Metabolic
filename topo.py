#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:50:23 2020

@author: tatiana
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import librosa
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy
import plotting


def plot_diagram(signal_array):
    sa = np.asarray(signal_array)
    diagram = VietorisRipsPersistence(homology_dimensions=(0, 3)).fit_transform(sa[np.newaxis,:,:])
    plotting.plot_diagram(diagram[0])
    return

def plot_pers_entr(pers_entr,surface_names=["one signal"]):
    plt.figure()
    
    
    pers_entr = np.asarray(pers_entr)
    
    fig, ax = plt.subplots()
    for i in range(len(surface_names)):
        n = 750
        print(pers_entr[i][0])
        x, y = pers_entr[i][0][0],pers_entr[i][0][1]
        ax.scatter(x, y, s = 280, label=surface_names[i],
                   alpha=0.8)
    ax.set_title("Persistence Entropy - all")
    ax.legend()
    ax.grid(True)
    
    plt.savefig("persistence_entropy_all.png", dpi = 400)
    plt.draw()
    plt.show()
    return

def extract_pers_entr(diagrams, step):
    ppe_p = []
    for x in diagrams:
        f = x[:int(step*len(x))]
        print(len(f))
        f = np.asarray(f)
        ol = VietorisRipsPersistence(homology_dimensions=(0, 3)).fit_transform(f[np.newaxis,:,:])
        pe_d = PersistenceEntropy().fit_transform(ol)
        ppe_p.append(pe_d)  

    return ppe_p
#%%
data = pd.read_csv("data.tsv", sep="\t")

data_clean = np.array(data.drop(["id", "display", "color", "category", "ExpNo", "Class", 
                        "Label", "User", "Experiment", "X1", "X2", "X3", "Animal"], axis=1))

#y = data["Class"].astype('category').cat.codes
y = np.asarray(data["Class"].astype('category').cat.codes)
clusters = ["Pre", "W2", "W6", "W8"]

names = list(data["Class"])
pre_c = names.count("Pre")
W2_c = names.count("W2")
W6_c = names.count("W6")
W8_c = names.count("W8")

s_X=[]
for x in data_clean:
    stretched_x = librosa.core.resample(np.asfortranarray(x), 8192, 1000000)
    s_X.append(stretched_x)

plt.figure()
plt.plot(data_clean[0])
plt.figure()
plt.plot(s_X[0])

#%%
pre, W2, W6, W8 = [], [], [], []
X = s_X
for i in range(len(X)):
    if y[i]==0:
        pre.append(X[i])
    elif y[i]==1:
        W2.append(X[i])
    elif y[i]==2:
        W6.append(X[i])
    else:
        W8.append(X[i])

#%% persistence entropy for every signal
ololo = [pre, W2, W6, W8]
new_all = []
for sett in ololo:
    print(len(sett))
    print("new example")
            
    all_signals = []
    for x in sett:
        l = x.reshape((1000,1000))
        all_signals.append(l)
    new_all.append(np.array(all_signals))

#%%
diagrams = [pre, W2, W6, W8]          
[plot_diagram(x[:10]) for x in diagrams]
#%%
# see if there are enough features to train a machine learning algorythm
# delete each cluster in four, fead stepwise, see results
plt.figure(dpi = 75)
ppe_pdd = []
diagrams = new_all[0]
steps = np.array([1]) 


for step in steps:
    ppe_pdd = extract_pers_entr(new_all[0],step)
X=ppe_pdd
for i in range(len(X)):
    plt.scatter(X[i][0][0], X[i][0][1],c='tab:blue', label='Pre')

for step in steps:
    ppe_pdd = extract_pers_entr(new_all[1],step)
X=ppe_pdd
for i in range(len(X)):
    plt.scatter(X[i][0][0], X[i][0][1],c='tab:green', label='W2')
    
for step in steps:
    ppe_pdd = extract_pers_entr(new_all[2],step)
X=ppe_pdd
for i in range(len(X)):
    plt.scatter(X[i][0][0], X[i][0][1],c='tab:red', label='W6')
    
for step in steps:
    ppe_pdd = extract_pers_entr(new_all[3],step)
X=ppe_pdd
for i in range(len(X)):
    plt.scatter(X[i][0][0], X[i][0][1],c='tab:orange', label='W8')  
plt.title("all_features")
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
    
    