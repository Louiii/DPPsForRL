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
# from Environments.BlockingTask import *
from Environments.StochasticPolicyTask import *

from PrintState import maze_record, makeGIF, plotQuality, plotSimilarity

class Autoregressive:
    def __init__(self, N, TEMP):
        # self.phi = np.c_[np.zeros(N), np.eye(N)] + np.random.uniform(-0.05, 0.05, size=(N, N+1))#np.ones((N, N+1))
        self.phi = np.random.uniform(-0.05, 0.05, size=(N, N+1))
        # self.phi[:, 0] += np.ones(N)*0.1
        # for i, j in [[3,0], [4,0], [1,3], [2,2]]:
        #     self.phi[i, j] =1
        # self.I = np.ones(N)
        self.TEMP=TEMP

        self.rN = self.phi.shape[0]

    def b(self):
        return self.phi[:, 0]

    def W(self):
        return self.phi[:, 1:]

    def d(self, z=None):
        # print(z)
        # print('shape b = (%d), shape W = (%d, %d), shap z = %s'%(self.b().shape[0], 
            # self.W().shape[0], self.W().shape[1], str(z.shape)))
        # return self.I
        # if self.TEMP.toString(self.TEMP.correct_states[0])==self.TEMP.toString(z):
        #     return np.log(self.TEMP.correct_states[1]+0.1*np.ones(len(z)))
        # if self.TEMP.toString(self.TEMP.correct_states[1])==self.TEMP.toString(z):
        #     return np.log(self.TEMP.correct_states[0]+0.1*np.ones(len(z)))
        return self.b() + np.dot(self.W(), z)


    def grad(self, z=None):
        dbdb = np.ones((self.rN, 1))# d/db (b) = [1,..,1]^T
        dWzdW = np.tile(z,(self.rN,1))# d/dW (Wz) = [z,..,z]^T
        # print('dbdb',str(dbdb.shape))
        # print('dWzdW',str(dWzdW.shape))
        # print()
        # print(self.phi)
        return np.c_[dbdb, dWzdW]

class DeterminantalSARSA:
    def __init__(self, env, time_series_model, ρ=0, η0=0.005, α=0):
        self.env = env
        self.ρ  = ρ
        self.η0 = η0
        self.α  = α
        self.βinit = 20
        self.βfrac = 10000
        self.ηstart = 1000
        self.bin = 200
        self.λ = 0.2
        self.d = time_series_model
        
        self.N = self.env.N
        K = self.N
        self.V = np.eye(self.N) + np.random.uniform(-0.01, 0.01, size=(self.N, K))
        
        self.max_n_ep = 1000
        
        self.found = False
        self.i = 0# counter used for rendering
    
    def indices(self, state):
        # return [i*7 + j for (i, j) in state]
        return np.nonzero(state)[0].tolist()
            
    # def encode(self, state):
    #     x = np.zeros(self.N)
    #     inds = self.indices(state)
    #     x[inds] = 1
    #     return x
    def encode(self, idxs):
        x = np.zeros(self.N)
        x[idxs] = 1
        return x
                    
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
    
    # def boltzmannPolicy(self, state, β):
    #     ''' compute all possible successor states
    #         encode all possible successor states
    #         compute det(L)^beta for all successor states
    #         sample action from pdf
    #     '''
    #     def computeDET(action):
    #         x = self.indices(action)
    #         V = self.V_rd( x )
    #         D = np.diag(np.exp(self.d.d(z=action)))
    #         return det( dot([V, D, V.T]) )

    #     a = np.random.randint(2, size=self.N)#self.env.valid_actions()[np.random.randint(self.N)]
    #     det_a = computeDET(a)

    #     # Compute all successor states (from the possible actions).
    #     for i in range(100):
    #         a_new = a.copy()
    #         j = np.random.randint(0, self.N)
    #         a_new[j] = 1-a[j]# flip a random bit
    #         # Select a new state (x_neighbour) uniformly from the successor states 
    #         # (that are also neighbours of x) (where a neighbour of x has two positions 
    #         # in common with x)
    #         det_a_new = computeDET(a_new)

    #         p = ( det_a_new / det_a )**β
    #         if np.random.rand() < p:
    #             a = a_new
    #             det_a = det_a_new
    #             # if i>90 and (self.env.toString(a)==self.env.toString([0,1,0,1,1]) or self.env.toString(a)==self.env.toString([0,1,0,1,1])):
    #             #     print(str(a), str(p))
    #     return a

    def boltzmannPolicy(self, state, β):
        actions = self.env.valid_actions( state )
        def computeDET(action):
            x = self.indices(action)
            V = self.V_rd( x )
            D = np.diag(np.exp(self.d.d(z=action)))
            return det( dot([V, D, V.T]) )
        # print(self.V)
        # print("-+"*25)
        # print([(a, [round(d, 2) for d in np.exp(self.d.d(z=a)).tolist()]) for a in actions])
        pmf = np.array([np.exp(computeDET(a)*10/β) for a in actions])
        # print("="*50)
        # print(list(zip(actions, [round(p, 2) for p in pmf/np.sum(pmf)])))
        i = np.random.choice(range(len(pmf)), p=pmf/np.sum(pmf))
        return actions[i]
        
    def ei_s(self, x): # can be useful for regularisation
        # idxs = np.reshape( np.argwhere( x == 1 ), (1, -1))[0]
        ei = np.zeros((len(x), self.N))
        for i, j in enumerate(x):
            ei[i, j] = 1
        return ei
    
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
        state  = self.env.reset_state()
        action = self.boltzmannPolicy(state, self.β)
        x = self.indices(action)#self.φ(state, action, ns=False)

        for t in range(1, self.max_n_ep+1):
            η = self.η0 * min(1, self.ηstart/(tstart + t) )
            # print(η, self.β)
            self.β = self.βinit#np.power(self.βinit, (tstart + t)/self.βfrac)
            self.render(rcd, state)
            
            # Observe reward r_{t+1} and next state s_{t+1}
            next_state, reward, done = self.env.step(state, action)
            # print(reward)
            # Choose the next action a_{t+1}
            next_action = self.boltzmannPolicy(next_state, self.β)

            # print(next_action)

            next_x = self.indices(next_action)#self.φ(next_state, next_action, ns=False)

            V_x = self.V_rd( x )
            V_next_x = self.V_rd( next_x )

            Dt  = np.diag(np.exp(self.d.d(z=self.encode(x))))# z_t = x_t-1
            Dt1 = np.diag(np.exp(self.d.d(z=self.encode(next_x))))
            
            Q_x = self.α + np.log( det(dot([V_x, Dt, V_x.T])) )
            Q_next_x = self.α + np.log( det(dot([V_next_x, Dt1, V_next_x.T])) )

            print(x)
            if x==[]:
                print('\n\n\n\nFUCK\n\n\n')
                print(Q_x)
            
            TD = reward + self.ρ * Q_next_x - Q_x
            # print(V_x)
            invV = pinv( V_x )
            # print('louis')
            # print(invV.shape)
            # print('\n\n')
            grad_Q = 2 * invV.T
            
            # self.α += η * TD# grad_α = 1
            # V update
#            print(self.α)
#            self.λ *= 0.9999
            # print(self.V)
            # self.V_wr( x, η * TD * grad_Q )#
            self.V_wr( x, η * TD * grad_Q - (V_x - self.ei_s(x)) * η * self.λ )
            
            # V(x) = V(x) + η*TD*grad_Q - A(x)*η*λ, 
            # where A(x) = V(x) - I(x), (I(x) is the rows of the identity 
            # matrix indexed by 1’s in x), where λ may be a function of the 
            # timestep.
            
            d_wrt_phi = self.d.grad(z=self.encode(x))
            Q_wrt_d = np.diag(np.dot(invV, V_x))
            # print('Q_wrt_d')
            # print(Q_wrt_d)
            # dVV = np.zeros((self.N, self.N))
            # for i, iv in enumerate(x): dVV[iv, iv] = dVxVx[i]
            # dVV = np.diagonal(dVxVx)
            # print('dVV')
            # print(dVV)

            # print('='*50)
            # print(self.d.phi)
            # print('-'*50)
            # print(η * TD * np.dot( dVV, grad_d ))
            # print('dVV shape '+ str(dVV.shape))
            # print('grad_d shape '+ str(grad_d.shape))
            self.d.phi += η * TD * np.outer( Q_wrt_d, d_wrt_phi[0,:] )
            # self.d.phi += η * TD * np.dot( dVV, grad_d )
        

            self.total_reward += reward
            if (tstart+t) % self.bin == 0:
                self.rec_reward.append( (t + tstart, self.total_reward/self.bin ) )
                self.total_reward = 0
                
            if done:
                self.render(rcd, next_state)
                self.found = True
#                print('\n\n\n\nCompleted episode, t=',str(t),'\n\n\n')
                return t
            
            
            state = next_state
            action = next_action
            x = next_x
            
        return self.max_n_ep

    def render(self, rcd, s):
        if rcd:
            self.i += 1
            maze_record(self.i, 'Blocking task', s, 4, 7, self.env.blockers_state, up=False, dpi=100)
    
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
            print('time: '+str(t)+', β: '+str(self.β))
            dt = self.episode(t, False)
            
            t += dt
        
        return self.rec_reward


def run(N, n_corr_states, repeats):
    env = SPT(N, n_corr_states)
    print(env.correct_states)
    tsm = Autoregressive(N, env)
    d = DeterminantalSARSA(env, tsm)
    rws = d.run_no_rec(repeats)
    print(env.correct_states)
    return list(zip(*rws))

if __name__=="__main__":
    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
    import matplotlib.pyplot as plt

    [x, y] = run(5, 2, 5000)
    # [x, y] = run(6, 3, 7000)
    # [x, y] = run(8, 3,10000)

    plt.plot(x, y)
    plt.ylim([0, 10])
    plt.savefig("detSARSA-StochasticPolicyTask")
    plt.show()
#     env = BlockerTask()
#     d = DeterminantalSARSA(env)
#     d.run_rec_hidden_state(30000)
            