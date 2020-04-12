#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 13:59:44 2020

@author: louisrobinson
"""

import numpy as np
#from numpy.linalg import det, pinv, multi_dot as dot
import sys
sys.path.append('../')
from Environments.BlockingTask import BlockerTask

from PrintState import maze_record, makeGIF, plotQuality, plotSimilarity



class ESARSA:
    def __init__(self, env, gamma=1):
        self.env = env
        self.N = 4*7# num of states
        self.M = 3*4# action vector length
        self.ac = {(0,0):None,(0,1):2, (0,-1):3, (1,0):0, (-1,0):1}
                #(0,0):[0,0,0,0],(0,1):[0,0,1,0],(0,-1):[0,0,0,1],
                   #(1,0):[1,0,0,0],(-1,0):[0,1,0,0],}
        
        self.L = 16
        self.Ws = np.random.normal(0, 0.05, size=(self.N, self.L))
        self.Wa = np.random.normal(0, 0.05, size=(self.M, self.L))
        self.bh = np.random.normal(0, 0.01, size=(self.L))
        self.bs = np.random.normal(0, 0.01, size=(self.N))
        self.ba = np.random.normal(0, 0.01, size=(self.M))
        
        self.a_acs = env.all_actions[1:]
        self.ac_idxs = [self.ac_idx(a) for a in self.a_acs]
#        print(self.ac_idxs)
        self.max_n_ep = 40
        self.gamma = gamma
        self.bin = 10000
        
        
    def st_idx(self, state):
        return [i*7 + j for (i, j) in state]
    
    def ac_idx(self, action):
        acti = []
        for i, act in enumerate(action):
            a = self.ac[act]
            if a is not None:
                acti.append( a + i * 4 )
        return acti
    
    def F(self, s_idx, a_idx):
        f = -sum(self.bs[s_idx])-sum(self.ba[a_idx])
        exponent = np.sum(self.Ws[s_idx, :], axis=0) + np.sum(self.Wa[a_idx, :], axis=0) + self.bh
#        print(f - sum(np.log(1+np.exp(exponent))))
        return f - sum(np.log(1+np.exp(exponent)))
    
    def policy(self, s_idx, T=1):
        pmf = np.array([self.F(s_idx, a_idx) for a_idx in self.ac_idxs])
#        print(pmf)
        pmf = np.exp(-pmf/T)
#        print(pmf/np.sum(pmf))
#        print(sum(pmf))
        return np.random.choice(range(len(self.ac_idxs)), p=pmf/np.sum(pmf))
    
    def updateParams(self, deltaBeta, a_idx, s_idx):
        self.bs[s_idx] += deltaBeta
        self.ba[a_idx] += deltaBeta
#        print((a_idx, s_idx))
        ex = np.sum(self.Ws[s_idx, :], axis=0) + np.sum(self.Wa[a_idx, :], axis=0) + self.bh
        sig = ex/(1 + ex)
        self.bh += deltaBeta * sig
        self.Wa[a_idx, :] += deltaBeta * sig
        self.Ws[s_idx, :] += deltaBeta * sig
    
    def episode(self, tstart, max_steps):
        state  = self.env.reset_state()
        print(state)
        s_idx = self.st_idx(state)
        ai = self.policy(s_idx)
        a_idx, action = self.ac_idxs[ai], self.a_acs[ai]
        cF = self.F(s_idx, a_idx)
        
        tot_time = tstart
        
        for t in range(1, self.max_n_ep+1):
            tot_time += 1
            beta = 0.0005 * min(1, 50000/tot_time )#1 / tot_time + 0.01
            T = 0.01 * (tot_time/max_steps) + (max_steps-tot_time)/max_steps
            
            next_state, reward, done = self.env.step(state, action)
            ns_idx = self.st_idx(state)
            ai = self.policy(ns_idx, T)
            na_idx, next_action = self.ac_idxs[ai], self.a_acs[ai]
            nF = self.F(ns_idx, na_idx)
#            print((state, action))
            td = reward - self.gamma * nF + cF
            self.updateParams(td * beta, a_idx, s_idx)

            self.total_reward += reward
            if tot_time % self.bin == 0:
                self.rec_reward.append( (tot_time, self.total_reward/self.bin ) )
                self.total_reward = 0
                
            if done:
                self.found = True
                print('\n\n\n\nCompleted episode, t=',str(t),'\n\n\n')
                return t
            
            # update variables
            state, action, s_idx, a_idx, nF = next_state, next_action, ns_idx, na_idx, nF
        print(state)
#        print(self.Wa)
        return self.max_n_ep
    
    def run(self, max_steps):
        self.rec_reward = []
        self.total_reward = 0
        
        t, ep = 0, 0
        while t < max_steps:
            ep += 1
            print('time: '+str(t), flush=True)
            dt = self.episode(t, max_steps)
            t += dt
        return self.rec_reward


env = BlockerTask()
agent = ESARSA(env)
results = agent.run(4e5)
#print(agent.a_idx( ((0,1), (0,0), (0,-1)) ))

#steps = 300000
#env = BlockerTask()
#agent = ESARSA( env )
#rews = agent.run( steps )
