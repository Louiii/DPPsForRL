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


class Qlearning:
    def __init__(self, env, α=0.5, γ=1):
        self.env = env
        self.α = α
        self.γ = γ
        self.Q = dict()

        self.bin = 40000
        
    def convert(self, o):
        if isinstance(o, np.int64): return int(o) 
    
#    def saveQ(self):
#        with open('plots/Qtable.json', 'w') as outfile:
#            outfile.write(json.dumps(list(self.Q.items()), default=self.convert))
#
#    def loadQ(self):
#        try:
#            print('Loading Q-Table...')
#            with open("plots/Qtable.json","r") as f:
#                self.Q = dict()
#                for [[s, a], q] in json.loads(f.read()):
#                    s = tuple([(np.int64(x[0]), np.int64(x[1])) for x in s])
#                    a = tuple([(np.int64(x[0]), np.int64(x[1])) for x in a])
#                    self.Q[(s, a)] = q
#        except IOError:
#            print("No previous Q-Table found, making a new one...")
#            self.Q = dict()
        
    def f(self, state):
        # x = np.zeros(4*7)
        # for (i, j) in state:
        #     ind = i*7 + j
        #     x[ind] = 1
        # return tuple(x)
        return state
            
    def q(self, s):
        x = self.f(s)
        if x in self.Q:
            return self.Q[x]
        else:
            return 0

    def maxAQ(self, state, returnAction=True):
        qa = []
        actions = self.env.valid_actions( state )
        for a in actions:
            ns, r, d = self.env.step(state, a, searchingPolicy=True)
            qa.append( (a, self.q( ns ) ) )
        mx = max(qa, key=lambda x:x[1])
        if returnAction:
            return mx[0]
        return mx[1]

    def policy(self, state, ε=0.0):
        if ε < np.random.rand():
            return self.maxAQ(state)
        actions = self.env.valid_actions( state )
        return actions[np.random.randint(len(actions))]
    
    def episode(self, tstart, epLen=40, rec=False):
        state  = self.env.reset_state()

        for t in range(1, epLen+1):
            action = self.policy(state)
            next_state, reward, done = self.env.step(state, action)
            
            if rec:
                maze_record(tstart+t, 'Blocking task, '+str(done)+', t=', next_state, 4, 7, self.env.blockers_state, up=False)

            max_Q = self.maxAQ(state, returnAction=False)
            # next_action = self.policy(next_state)# either this line or the one above 'if done'
            
            state, next_state = tuple(state), tuple(next_state)
            update = self.α*(reward + self.γ*max_Q - self.q(state))

            x = self.f(state)
            if x in self.Q:
                self.Q[x] += update
            else:
                self.Q[x] = update
            
            
            self.total_reward += reward
            if (tstart+t) % self.bin == 0:
                # print('average_reward ; ', str(self.total_reward/40000))
                self.rec_reward.append( (tstart+t, self.total_reward/self.bin ) )
                self.total_reward = 0
            
            
            state = next_state
            
            if done: return t
        return epLen
        
    
    def run(self, epLen=40, mxsteps=1000, rec_any=True):
        """ on-policy TD control for estimating Q """
        self.rec_reward = []
        self.total_reward = 0
        
        t = 0
        rec = False
        # ep = 0
        while t <= mxsteps :
            # ep += 1
            # if ep%200==0:print(t)
            
            if t > mxsteps - 200 and rec_any: rec= True
            
            dt = self.episode(t, epLen, rec)
            t += dt
            
            
            
        return self.rec_reward
    
    