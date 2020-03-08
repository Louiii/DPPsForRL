#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:52:32 2019

@author: louisrobinson
"""
import sys
sys.path.append('../../')

import numpy as np
import json

from tqdm import tqdm
from PrintState import maze_record


class SARSA_noBoltzStep:
    def __init__(self, env, α=0.5, γ=1, ε=0.1):
        self.env = env
        self.α = α
        self.γ = γ
        self.Q = dict()
        self.ε = ε

        self.bin = 40000

        self.tau = 10
        
    # def convert(self, o):
    #     if isinstance(o, np.int64): return int(o) 
            
    def q(self, s, a):
        if (s, a) in self.Q:
            return self.Q[(s, a)]
        else:
            return 0
        
    def policy(self, state, ε=0.0):
        actions = self.env.valid_actions( state )
        if ε < np.random.rand():
            qa = []
            for a in actions:
                qa.append( (a, self.q(state, a)) )
            mx = max(qa, key=lambda x:x[1])
            return mx[0]
        return actions[np.random.randint(len(actions))]

    def boltzmann(self, state):
        self.tau *= 0.99999
        actions = self.env.valid_actions( state )
        pmf = np.array([np.exp(self.q(state, a)/self.tau) for a in actions])
        # print(pmf)
        # print('Tau: '+str(self.tau))
        i = np.random.choice(range(len(pmf)), p=pmf/np.sum(pmf))
        return actions[i]
    
    def episode(self, tstart, epLen=40, rec=False):
        state  = self.env.reset_state()
        action = self.policy(state, ε=self.ε)
        # action = self.boltzmann(state)

        for t in range(1, epLen+1):
            next_state, reward, done = self.env.step(state, action)
            next_action = self.policy(state, ε=self.ε)
            # next_action = self.boltzmann(state)

            if rec: maze_record(tstart+t, 'Blocking task, '+str(done)+', t=', next_state, 4, 7, self.env.blockers_state, up=False)

            # perform update
            update = self.α*(reward + self.γ*self.q(next_state, next_action) - self.q(state, action))
            if (state, action) in self.Q:
                self.Q[(state, action)] += update
            else:
                self.Q[(state, action)] = update
            
            
            
            # record learning rate
            self.total_reward += reward
            if (tstart+t) % self.bin == 0:
                self.rec_reward.append( (tstart+t, self.total_reward/self.bin ) )
                self.total_reward = 0
            
            
            state, action = next_state, next_action
            
            if done: return t
        return epLen
        
    
    def run(self, epLen=40, mxsteps=1000, rec_any=True):
        """ on-policy TD control for estimating Q """
        self.rec_reward = []
        self.total_reward = 0
        
        t = 0
        rec = False
        ep = 0
        while t <= mxsteps :
            ep += 1
            if ep%200==0:print(str(t*100/mxsteps)+'%', flush=True)
            
            if t > mxsteps - 200 and rec_any: rec= True
            
            dt = self.episode(t, epLen, rec)
            t += dt
            
            
            
        return self.rec_reward
    