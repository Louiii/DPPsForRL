#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 22:18:27 2020

@author: louisrobinson
"""
import numpy as np
import sys
sys.path.append('../../')
from DPP import  DualDPP
import json

def load_data():
    ''' [user, item, r] '''
    with open("data.json", 'r') as f: 
    	d = json.load(f)
    return np.array(d, dtype=int)

def makeY(data, n, m):
    all_ = set(range(len(data)))
    pos = set(list(np.nonzero(data[:, -1])[0]))
    neg = list(all_ - pos)
    Y_coords = {0:data[neg, :2], 1:data[list(pos), :2]}
    item_popularity = {i:set([]) for i in range(m)}
    Y_tele = dict()
    for [i, j, r] in data:
        item_popularity[j].add(i)
        if r==1:
            if i in Y_tele:
                Y_tele[i].add(j)
            else:
                Y_tele[i] = set([j])
    item_pop = np.zeros(m)
    for j in range(m): item_pop[j] = len(item_popularity[j])/n
    return Y_tele, Y_coords, item_pop, len(pos), len(neg)

class MF:
    def __init__(self, train=False):
        self.mtx = lambda a, b: np.random.normal(0, 0.05, (a, b))
        
        if train:
            print('loading data...')
            self.dataset = load_data()
            
            self.n = len(set(list(self.dataset[:,0])))# num users, approx 2324
            self.m = len(set(list(self.dataset[:,1])))# num tracks, approx 7886
            # ^ dims of matrix Y (stored in sparse represetation).
            self.du, self.dm = 15, 15# latent dimensions
        
            self.Y_tele, self.Y_coords, self.item_popularity, self.n_pos, self.n_neg = makeY(self.dataset, self.n, self.m)
            self.tr_te_split()
            
            self.U, self.M = self.mtx(self.n, self.du), self.mtx(self.m, self.dm)
            self.mean = self.n_pos/(self.n_neg + self.n_pos)
            self.calc_biases()
        else:
            print('loading model...')
            self.load()
            print('done.')

    def calc_biases(self):
        bu, bi, uc, ic = np.zeros(self.n), np.zeros(self.m), np.zeros(self.n), np.zeros(self.m)
        for [u, i, r] in self.dataset:
            bu[u] += r
            bi[i] += r
            uc[u] += 1
            ic[i] += 1
        self.bu, self.bi = bu/uc, bi/ic
    
    def y(self, i, j):
        if i in self.Y_tele:
            if j in self.Y_tele[i]:
                return 1
        return 0
    
    def save(self, path='model/', UM=None):
        np.save(path+'U', self.U), np.save(path+'M', self.M)
        if UM==None:
            np.save(path+'item_popularity', self.item_popularity)
            np.save(path+'Bu', self.bu), np.save(path+'Bi', self.bi)
            np.save(path+'mean', self.mean)
        
    def load(self, path='model/'):
        self.item_popularity = np.load(path+'item_popularity.npy')
        self.U, self.M = np.load(path+'U.npy'), np.load(path+'M.npy')
        self.bu, self.bi = np.load(path+'Bu.npy'), np.load(path+'Bi.npy')
        self.mean = np.load(path+'mean.npy')
        (self.n, self.du), (self.m, self.dm) = self.U.shape, self.M.shape
        
    def F(self, i, j):
        u_i = self.U[i, :]
        m_j = self.M[j, :]
        return self.mean + self.bu[i] + self.bi[j] + np.dot(u_i, m_j)
        
    def loss(self, i, j):
        return 0.5*(self.F(i, j) - self.y(i, j))**2
        
    def df_loss(self, i, j):
        return self.F(i, j) - self.y(i, j)
        
    def dU(self, i, j):
        return self.M[j, :]

    def dM(self, i, j):
        return self.U[i, :]

    def tr_te_split(self):
        pos_idxs = set(list(np.random.choice(range(self.n_pos), self.n_pos//5, replace=False)))
        tr_pos_idxs = set(range(self.n_pos)) - pos_idxs
        neg_idxs = set(list(np.random.choice(range(self.n_neg), self.n_neg//5, replace=False)))
        tr_neg_idxs = set(range(self.n_neg)) - neg_idxs
        self.train_idxs = {0: list(tr_neg_idxs), 1: list(tr_pos_idxs)}
        self.test_idxs = {0: list(neg_idxs), 1: list(pos_idxs)}
        
        self.n_tr = {0: len(self.train_idxs[0]), 1:len(self.train_idxs[1])}
        self.n_te = {0: len(self.test_idxs[0]), 1:len(self.test_idxs[1])}

    def train(self):
        print_time = 1000
        losses = []
        for t in range(1, int(1e7)):
            alpha = min(0.3, int(1e4)/t)
            
            choice, tot = (1, self.n_pos) if np.random.random() < 0.2 else (0, self.n_neg) # do positive example:
            indices = self.Y_coords[choice][self.train_idxs[choice][np.random.randint(self.n_tr[choice])]]
            [i, j] = indices[:2]
            
            F_loss = self.df_loss(i, j)
            d_U, d_M = self.dU(i, j), self.dM(i, j)
            
            self.U[i, :] -= alpha * d_U * F_loss
            self.M[j, :] -= alpha * d_M * F_loss
            
            losses.append( F_loss )
            if t%print_time==0:
                print('time: '+str(t)+', loss, '+str(sum(np.abs(np.array(losses)))/len(losses)))
                losses = []
                if t%(print_time*10)==0: self.save(UM=True)

    def generateDiversityVecs(self):
        self.diversity = np.array([self.M[j, :]/np.linalg.norm(self.M[j, :]) for j in range(self.M.shape[0])])

    def recommendDPP(self, user, obs_items, n_recs):
        ''' 
        data required: 
            arr: item -> popularity_normalised        
            arr: predictions = generate r_ui for all i
            mat: diversity vectors for all items
        '''
        un_obs_items = list(set(list(range(self.m))) - set(obs_items))
        base_Y, selection = np.array(un_obs_items), []
        if len(obs_items) > 0:
            selection = np.random.choice(obs_items, min(n_recs, len(obs_items)), replace=False)
            base_Y = np.append(base_Y, selection)
        ''' quality is a function of user-context-item '''
        quality = np.array([self.F(user, j)*self.item_popularity[j] for j in base_Y])
        mask = quality > 0# remove any with q < 0
        quality = quality[mask]
        ''' quality is a function of the latent space of items '''
        B = np.array([quality[i] * self.diversity[j, :] for i, j in enumerate(base_Y[mask])])
        # --> sample dual dpp
        C = np.dot(B.T, B)
        dpp = DualDPP(C, B.T)
        selected_items = list( dpp.sample_dual(k=n_recs) )
        return selected_items
    
    def recommendIndep(self, user, obs_items, n_recs):
        ''' 
        data required: 
            arr: item -> popularity_normalised        
            arr: predictions = generate r_ui for all i
        '''
        un_obs_items = list(set(list(range(self.m))) - set(obs_items))
        base_Y, selection = np.array(un_obs_items), []
        if len(obs_items) > 0:
            selection = np.random.choice(obs_items, min(n_recs, len(obs_items)), replace=False)
            base_Y = np.append(base_Y, selection)
        ''' quality is a function of user-context-item '''
        quality = np.array([self.F(user, j)*self.item_popularity[j] for j in base_Y])
        mask = quality > 0# remove any with q < 0
        quality, base_Y = quality[mask], base_Y[mask]
        selected_items = np.random.choice(base_Y, n_recs, p=quality/sum(quality), replace=False)
        return selected_items

#rs = MF(train=True)
#rs.train()
        
#rs = MF()
#print(rs.recommendDPP(0, list(range(30))))

