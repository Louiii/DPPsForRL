#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 01:22:15 2020

@author: louisrobinson
"""
from RS import MF
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


def RMSE(y_hat, y):#5
    return np.sqrt(np.sum(np.power(y_hat-y, 2))/len(y_hat))
    
def MAE(y_hat, y):#1
    return np.sum(np.abs(y_hat-y))/len(y_hat)

def confusion_matrix(y_hat, y, th=0.5):
    tp, fp, fn, tn = 0, 0, 0, 0
    a = np.zeros(len(y_hat))
    a[y_hat >= th] = 1
    for yh, y in zip(a, y):
        if yh==1:
            if y==1:
                tp += 1
            else:
                fp += 1
        else:
            if y==1:
                fn += 1
            else:
                tn += 1
    return tp, fp, fn, tn

def recall_precision(y_hat, y):#2,3
    tp, fp, fn, tn = confusion_matrix(y_hat, y)
    return tp/(tp+fn), tp/(tp+fp)

def recall_precision_curve(y_hat, y, n=15):#2,3
    fpr_tpr = []
    for th in np.linspace(0,1,n):#6
        tp, fp, fn, tn = confusion_matrix(y_hat, y, th)
        fpr_tpr.append([tp/(tp+fn), tp/(tp+fp)])
    return np.array(fpr_tpr)

def ROC(y_hat, y, n=15):
    fpr_tpr = []
    for th in np.linspace(0,1,n):#6
        tp, fp, fn, tn = confusion_matrix(y_hat, y, th)
        fpr_tpr.append([fp/(fp+tn), tp/(tp+fn)])
    return np.array(fpr_tpr)

def novelty(item_pop_R):
    ''' item_list = proportion of users that rated an item '''
    # for each user: give a recommendation, produce one long list of items.
    # compute log_2(popularity(item)) / len(list)
    return -sum(np.log2(item_pop_R))/len(item_pop_R)

def catalogCoverage(rs, user, N=1500):
    m = rs.m
    
    obs_items = set([])
    coverage = np.zeros(N)
    for i in tqdm(range(N)):
        recs = rs.recommend(user, list(obs_items))
        obs_items = obs_items.union(recs)
        coverage[i] = len(obs_items)/m
        print(coverage[i])
    return coverage



rs = MF(train=True)# train == True loads Y so that we can test it
rs.load()

''' TEST '''
difference, num = [], 1000
for _ in range(num):
    choice = 1 if np.random.random() < 0.2 else 0 # do positive example:
    [i, j] = rs.Y_coords[choice][rs.test_idxs[choice][np.random.randint(rs.n_te[choice])]]
    F_hat = 1 if rs.F(i,j)>0.5 else 0
    difference.append(np.abs(F_hat-rs.y(i,j)))
print(1-sum(difference)/num)



num = 100000
ys, fs = np.zeros(num), np.zeros(num)
for s in tqdm(range(num)):
    choice = 1 if np.random.random() < 0.2 else 0 # do positive example:
    indices = rs.Y_coords[choice][rs.test_idxs[choice][np.random.randint(rs.n_te[choice])]]
    [i, j], c_idxs = indices[:2], list(indices[2:])
    fs[s], ys[s] = rs.F(i,j), rs.y(i,j)
    
frounded = np.round(fs)

rmse, mae = RMSE(frounded, ys), MAE(frounded, ys)
recall, precision = recall_precision(frounded, ys)
roc = ROC(fs, ys, 20)
pr = recall_precision_curve(fs, ys, 20)
#novelty(p_item_list)
#item_coverage(all_items, predictable_items)

print('(rmse, mae, recall, precision)')
print((rmse, mae, recall, precision))


###############################################################################
###################            Both ROC curves            #####################
###############################################################################

plt.clf()
plt.plot(roc[:,0], roc[:,1], 'c', label='MF')
#plt.plot(roc_tf[:,0], roc_tf[:,1], "#72246C", label='TF-DPP')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('ROC', dpi=400)
plt.show()

###############################################################################
###################     Both Precision-Recall curves      #####################
###############################################################################

plt.clf()
plt.plot(pr[:,0], pr[:,1], 'c', label='MF')
#plt.plot(pr[:,0], pr[:,1], "#72246C", label='TF-DPP')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.savefig('PR', dpi=400)
plt.show()

###############################################################################
###################           Catalog Coverage            #####################
###############################################################################
user, N = 0, 1500
coverage = catalogCoverage(rs, user, N=N)

plt.clf()
plt.plot([i*10 for i in range(1, N+1)], coverage, 'c', label='SVD-PF')
#plt.plot([i*10 for i in range(1, N+1)], coverage_tf, "#72246C", label='TF-DPP')
plt.plot([rs.m, rs.m], [0, 1.05], 'r', label='Total number of items')
plt.xlabel('Number of tracks recommended')
plt.ylabel('Proportion of tracks seen')
plt.ylim([0, 1.05])
plt.legend()
#plt.title('Catelog coverage')
plt.savefig('CatalogCoverage', dpi=300)
plt.show()


###############################################################################
###################                Novelty                #####################
###############################################################################
N = 500
user = 0
m = rs.m

obs_items = set([])
nov = np.zeros(N)
for i in tqdm(range(N)):
    recs = rs.recommend(user, list(obs_items))
    item_pop_R = rs.item_popularity[recs]
    nov[i] = novelty(item_pop_R)
#    print(nov[i])
print(np.mean(nov))# 8.29
print(np.std(nov))# 0.658
    
    