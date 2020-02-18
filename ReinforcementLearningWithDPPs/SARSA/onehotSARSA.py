#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:52:32 2019

@author: louisrobinson
"""
import sys
sys.path.append('../')

import numpy as np
import json

from tqdm import tqdm
from PrintState import maze_record

from Environments.BlockingTask import *


class onehotSARSA:
    def __init__(self, env, α=0.5, γ=1, ε=0.0):
        self.env = env
        self.α = α
        self.γ = γ
        self.Q = np.zeros(28)
        self.ε = ε

        self.bin = 4000
        
    # def convert(self, o):
    #     if isinstance(o, np.int64): return int(o) 
    
    # def indices(self, state):
    #     return np.nonzero(state)[0].tolist()
    def indices(self, state):
        inds = []
        for (i, j) in state:
            inds.append( i*7 + j )
        return inds

    def encode(self, idxs):
        x = np.zeros(self.N)
        x[idxs] = 1
        return x

    def φ(self, state, action):#, ns=True):
        ''' feature representation:
        state  = position of all the agents = ((i1, j1), (i2, j2), (i3, j3))
        action = change in position = ((di1, dj1), (di2, dj2), (di3, dj3))
        ns <- compute next state
        encode ns
        '''
        next_state, reward, done = self.env.step(state, action, searchingPolicy=True)
        # print(state)
        x = self.indices(next_state)
        return x

    def q(self, s, a):
        x = self.φ(s, a)
        return np.sum(self.Q[x])

    def updateQ(self, state, action, update):
        x = self.φ(state, action)
        for i in x: self.Q[i] += update
        
    def policy(self, state, ε=0.0):
        # return self.boltzmann(state, tau=1)
        actions = self.env.valid_actions( state )
        if ε < np.random.rand():
            qa = []
            for a in actions:
                qa.append( (a, self.q( state, a ) ) )
            mx = max(qa, key=lambda x:x[1])
            return mx[0]
        return actions[np.random.randint(len(actions))]

    # def boltzmann(self, state, tau=0.5):
    #     actions = self.env.valid_actions( state )
    #     pmf = np.array([np.exp(self.q(state, a)/tau) for a in actions])
    #     # print(pmf)
    #     i = np.random.choice(range(len(pmf)), p=pmf/np.sum(pmf))
    #     return actions[i]
    
    def episode(self, tstart, epLen=40):
        state  = self.env.reset_state()
        action = self.policy(tuple(state), ε=self.ε)

        for t in range(1, epLen+1):
            next_state, reward, done = self.env.step(state, action)
            next_action = self.policy(state, ε=self.ε)

            TD = (reward + self.γ*self.q(next_state, next_action) - self.q(state, action))
            self.updateQ(state, action, self.α*TD)

            
            self.total_reward += reward
            if (tstart+t) % self.bin == 0:
                # print('average_reward ; ', str(self.total_reward/40000))
                self.rec_reward.append( (tstart+t, self.total_reward/self.bin ) )
                self.total_reward = 0
            
            
            state, action = next_state, next_action
            
            if done: 
                print("YAS")
                return t
        return epLen
        
    
    def run(self, epLen=40, mxsteps=1000):
        """ on-policy TD control for estimating Q """
        self.rec_reward = []
        self.total_reward = 0
        
        t = 0
        while t <= mxsteps :
            print('\r%d' % t, end='', flush=True)

            dt = self.episode(t, epLen)
            t += dt
        return self.rec_reward

if __name__=="__main__":
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    import matplotlib.pyplot as plt

    env = BlockerTask()
    agent = onehotSARSA( env, 0.1 )
    rews = agent.run( mxsteps=200000 )
    [x, y] = list(zip(*rews))
    plt.ylim([-1, -0.6])
    plt.plot(x, y)
    plt.show()

    