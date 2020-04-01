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
###################                Novelty                #####################
###############################################################################
N = 200
N_recs = 8
m = rs.m
novDPP, divDPP, novIND, divIND = [],[],[],[]
rs.generateDiversityVecs()
n_users = 10
c = 0
for user in np.random.choice(range(rs.n), n_users, replace=False):
	c+=1
	print(str(c)+' of '+str(n_users))
	obs_items = set([])
	for i in tqdm(range(N)):
	    recs = rs.recommendDPP(user, list(obs_items), N_recs)
	    item_pop_R = rs.item_popularity[recs]
	    novDPP.append(novelty(item_pop_R))
	    divDPP.append(np.mean([np.dot(rs.diversity[recs[a], :], rs.diversity[recs[b], :]) for a in range(N_recs) for b in range(a)]))

	obs_items = set([])
	for i in tqdm(range(N)):
	    recs = rs.recommendIndep(user, list(obs_items), N_recs)
	    item_pop_R = rs.item_popularity[recs]
	    novIND.append(novelty(item_pop_R))
	    divIND.append(np.mean([np.dot(rs.diversity[recs[a], :], rs.diversity[recs[b], :]) for a in range(N_recs) for b in range(a)]))

print("Novelty:")
print("DPP: Mean: "+str(np.mean(novDPP))+", std dev: "+str(np.std(novDPP)))
print("IND: Mean: "+str(np.mean(novIND))+", std dev: "+str(np.std(novIND)))

print("Diversity:")
print("DPP: Mean: "+str(np.mean(divDPP))+", std dev: "+str(np.std(divDPP)))
print("IND: Mean: "+str(np.mean(divIND))+", std dev: "+str(np.std(divIND)))
    
    