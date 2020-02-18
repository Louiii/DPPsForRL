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


class SARSA:
    def __init__(self, env, α=0.5, γ=1, ε=0.0):
        self.env = env
        self.α = α
        self.γ = γ
        self.Q = dict()
        self.ε = ε

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
        
    def policy(self, state, ε=0.0):
        actions = self.env.valid_actions( state )
        if ε < np.random.rand():
            qa = []
            for a in actions:
                ns, r, d = self.env.step(state, a, searchingPolicy=True)
                qa.append( (a, self.q( ns ) ) )
            mx = max(qa, key=lambda x:x[1])
            return mx[0]
        return actions[np.random.randint(len(actions))]
    
    def episode(self, tstart, epLen=40, rec=False):
        
        state  = self.env.reset_state()
        action = self.policy(tuple(state), ε=self.ε)
        print(action)

        for t in range(1, epLen+1):
            next_state, reward, done = self.env.step(state, action)
            print(str(action)+', '+str(reward))
            
            if rec:
                maze_record(tstart+t, 'Blocking task, '+str(done)+', t=', next_state, 4, 7, self.env.blockers_state, up=False)

            # next_action = self.policy(next_state)# either this line or the one above 'if done'
            
            state, next_state = tuple(state), tuple(next_state)
            update = self.α*(reward + self.γ*self.q(next_state) - self.q(state))
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
            action = self.policy(state, ε=self.ε)
            
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

class properSARSA:
    def __init__(self, env, α=0.5, γ=1, ε=0.0):
        self.env = env
        self.α = α
        self.γ = γ
        self.Q = dict()
        self.ε = ε

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
        
    def f(self, state, action):
        # x = np.zeros(4*7)
        # for (i, j) in state:
        #     ind = i*7 + j
        #     x[ind] = 1
        # return tuple(x)
        return (state, tuple(action))
            
    def q(self, s, a):
        x = self.f(s, a)
        # print(x)
        if x in self.Q:
            return self.Q[x]
        else:
            return 0
        
    def policy(self, state, ε=0.0):
        actions = self.env.valid_actions( state )
        if ε < np.random.rand():
            qa = []
            for a in actions:
                qa.append( (a, self.q( state, a ) ) )
            mx = max(qa, key=lambda x:x[1])
            return mx[0]
        return actions[np.random.randint(len(actions))]

    def boltzmann(self, state, tau=0.5):
        actions = self.env.valid_actions( state )

        # temp = [self.q(state, a) for a in actions]
        # temp.sort()
        # temp = np.array(temp[-2:])
        # print(temp)
        # print(np.exp(temp/tau))

        pmf = np.array([np.exp(self.q(state, a)/tau) for a in actions])
        i = np.random.choice(range(len(pmf)), p=pmf/np.sum(pmf))
        return actions[i]
    
    def episode(self, tstart, epLen=40, total_steps=20000, rec=False):
        
        state  = self.env.reset_state()
        action = self.boltzmann(state, (total_steps-0.8*tstart)/total_steps)#self.policy(tuple(state), ε=self.ε)
        # print(action)

        for t in range(1, epLen+1):
            next_state, reward, done = self.env.step(state, action)
            # print(str(action)+', '+str(reward))
            
            if rec:
                maze_record(tstart+t, 'Blocking task, '+str(done)+', t=', next_state, 4, 7, self.env.blockers_state, up=False)

            next_action = self.boltzmann(next_state, (total_steps-0.5*(t+tstart))/total_steps)#self.policy(next_state)# either this line or the one above 'if done'
            
            # state, next_state = tuple(state), tuple(next_state)
            update = self.α*(reward + self.γ*self.q(next_state, next_action) - self.q(state, action))
            x = self.f(state, action)
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
            action = next_action
            
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
            
            dt = self.episode(t, epLen, mxsteps, rec)
            t += dt
            
            
            
        return self.rec_reward
    
from Environments.StochasticPolicyTask import *

def run(N, n_corr_states, repeats):
    env = SPT(N, n_corr_states)
    env.retTuple = True
    s = properSARSA(env, α=0.01, γ=0., ε=0.3)
    s.bin = 2000
    rws = s.run(epLen=100, mxsteps=repeats, rec_any=False)
    print(env.correct_states)
    print(rws)
    return list(zip(*rws))

if __name__=="__main__":
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    import matplotlib.pyplot as plt

    # [x, y] = run(5, 2, 1000)
    # plt.plot(x, y)

    [x, y] = run(5, 2, 20000)
    # [x, y] = run(6, 3, 40000)
    # [x, y] = run(8, 3, 30000)
    plt.plot(x, y)
    plt.ylim([0, 5])

    # [x, y] = run(5, 2, 4000)
    # plt.plot(x, y)
    plt.savefig('StochasticPolicyTask')
    plt.show()
    