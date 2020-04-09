#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 16:04:14 2019

@author: louisrobinson
"""
import numpy as np
from numpy.linalg import det, pinv, multi_dot as dot


import sys
sys.path.append('../')
from Environments.BlockingTask import *

from PrintState import maze_record, makeGIF, plotQuality, plotSimilarity


class DeterminantalSARSA:
    def __init__(self, env, ρ=0.9, η0=0.001, α=0):
        self.env = env
        self.ρ  = ρ
        self.η0 = η0
        self.α  = α
        self.βinit = 10
        self.βfrac = 10000
        self.ηstart = 10000
        self.bin = 4000
        self.λ = 1
        
        self.N = 4*7
        K = self.N
        self.V = np.eye(self.N) + np.random.uniform(-0.01, 0.01, size=(self.N, K))
        
        self.max_n_ep = 40
        
        self.found = False
        self.i = 0# counter used for rendering
    
    def indices(self, state):
        return [i*7 + j for (i, j) in state]
            
    # def encode(self, state):
    #     x = np.zeros(self.N)
    #     inds = self.indices(state)
    #     x[inds] = 1
    #     return x
                    
    def φ(self, state, action, ns=True):
        ''' feature representation:
        state  = position of all the agents = ((i1, j1), (i2, j2), (i3, j3))
        action = change in position = ((di1, dj1), (di2, dj2), (di3, dj3))
        ns <- compute next state
        encode ns
        '''
        next_state, reward, done = self.env.step(state, action, searchingPolicy=True)
#        print('φ, ns = ',str(next_state))
        # x = self.encode( next_state )
        x = self.indices(next_state)
        if ns:
            return x, next_state, reward, done
        return x
    
    def boltzmannPolicy(self, state, β):
        ''' compute all possible successor states
            encode all possible successor states
            compute det(L)^beta for all successor states
            sample action from pdf
        '''
        seen = set([])
        encoded_successors = []
        actions = []
        for a in self.env.valid_actions( state ):
            x, ns, _, _ = self.φ( state, a, ns=True )
            ns = str(ns)
            if ns not in seen:
                seen.add( ns )
                encoded_successors.append( x )
                actions.append( a )
        
        dets = []
        for x in encoded_successors:
            V = self.V_rd( x )
            dets.append( det( np.dot(V, V.T) ) )
        dets = np.array(dets)
        
        if β > 2:
            m = max(dets)# BAD - high time compexity for large action spaces! 
            # I've done this to avoid numerical errors, (it still samples the same distribution)
            
            ''' COULD THIS BE SOLVED BY LOGS?? '''

            dets /= m

        pmf = np.power( dets, β )

        ind = np.random.choice(range(len(pmf)), p=pmf/np.sum(pmf))
        return actions[ind]
        
    # def ei_s(self, x): # can be useful for regularisation
    #     idxs = np.reshape( np.argwhere( x == 1 ), (1, -1))[0]
    #     ei = np.zeros((len(idxs), len(x)))
    #     for i, j in enumerate(idxs):
    #         ei[i, j] = 1
    #     return ei
    
    # def V_rd(self, x):
    #     idxs = np.reshape( np.argwhere( x == 1 ), (1, -1))[0]
    #     return np.copy( self.V[idxs, :] )
    # def V_wr(self, x, mat):
    #     idxs = np.reshape( np.argwhere( x == 1 ), (1, -1))[0]
    #     self.V[idxs, :] += mat

    def V_rd(self, x):
        return np.copy( self.V[x, :] )
    
    def V_wr(self, x, mat):
        self.V[x, :] += mat
        
    def episode(self, tstart, rcd=False):
        # initialise variables
        state  = self.env.reset_state()
        action = self.boltzmannPolicy(state, self.β)
        x = self.φ(state, action, ns=False)
        V_x = self.V_rd( x )
        Q_x = self.α + np.log( det(np.dot(V_x, V_x.T)) )

        for t in range(1, self.max_n_ep+1):
            # update β, η
            η = self.η0 * min(1, self.ηstart/(tstart + t) )
            self.β = np.power(self.βinit, (tstart + t)/self.βfrac)
            self.render(rcd, state)
            
            # compute 'next' variables
            next_state, reward, done = self.env.step(state, action)
            next_action = self.boltzmannPolicy(next_state, self.β)
            next_x = self.φ(next_state, next_action, ns=False)
            V_next_x = self.V_rd( next_x )
            Q_next_x = self.α + np.log( det(np.dot(V_next_x, V_next_x.T)) )
            
            # compute error
            TD = reward + self.ρ * Q_next_x - Q_x
            # compute gradient
            grad_Q = 2 * pinv( V_x ).T
            # update parameters, V
            self.V_wr( x, η * TD * grad_Q )# - (V_x - self.ei_s(x)) * η * self.λ )
#            self.λ *= 0.9999
#            self.α += η * TD# grad_α = 1
            
            # record data
            self.total_reward += reward
            if (tstart+t) % self.bin == 0:
                self.rec_reward.append( (t + tstart, self.total_reward/self.bin ) )
                self.total_reward = 0
            
            if done:
                self.render(rcd, next_state)
                self.found = True
                return t
            
            # update variables
            state, action, x, V_x, Q_x = next_state, next_action, next_x, V_next_x, Q_next_x
        return self.max_n_ep

    def render(self, rcd, s):
        if rcd:
            self.i += 1
            maze_record(self.i, None, s, 4, 7, self.env.blockers_state, up=False, dpi=300)
    
    def run_rec_hidden_state(self, max_steps):
        self.rec_reward = []
        self.total_reward = 0
        self.β = 1
        
        t = 0
        rcd = False
        ep = 0
        while t < max_steps:
            ep += 1
            if ep%100==0: 
                print('time: '+str(t))
                
            if ep%10==0:
                plotQuality( self.V, ep, '../plots/temp-plots/temp-plots1/V'+str(ep), self.found, dpi=100, show=False )
                plotSimilarity( self.V, ep, '../plots/temp-plots/temp-plots2/V'+str(ep), self.found, dpi=100, show=False, plotL=True )
                plotSimilarity( self.V, ep, '../plots/temp-plots/temp-plots3/V'+str(ep), self.found, dpi=100, show=False )
            
            dt = self.episode(t, rcd)
            
            t += dt
        
        makeGIF('../plots/temp-plots/temp-plots1', '../plots/changing-quality')
        makeGIF('../plots/temp-plots/temp-plots2', '../plots/changing-L')
        makeGIF('../plots/temp-plots/temp-plots3', '../plots/changing-similarity')
        return self.rec_reward
    
    def run(self, max_steps):
        self.rec_reward = []
        self.total_reward = 0
        self.β = 1
        
        t = 0
        rcd = False
        ep = 0
        while t < max_steps:
            ep += 1
#            if ep%100==0: 
#                print('time: '+str(t))
            print(str(ep)+', time: '+str(t))
                
            if t > max_steps-200:
                rcd = True
                
            dt = self.episode(t, rcd)
            
            t += dt
        makeGIF('plots/temp-plots/temp-plots1', 'plots/DetSARSA-Blocker')
        return self.rec_reward
    
    def run_no_rec(self, max_steps):
        self.rec_reward = []
        self.total_reward = 0
        self.β = 1
        
        t = 0
        ep = 0
        while t < max_steps:
            ep += 1
            # print('time: '+str(t)+', β: '+str(self.β))
            dt = self.episode(t, False)
            
            t += dt
        
        return self.rec_reward

# if __name__=="__main__":
#     env = BlockerTask()
#     d = DeterminantalSARSA(env)
#     d.run_rec_hidden_state(30000)
            