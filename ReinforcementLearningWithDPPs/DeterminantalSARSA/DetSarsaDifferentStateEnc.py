#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 10:46:03 2019

@author: louisrobinson
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 16:04:14 2019

@author: louisrobinson
"""

import numpy as np
from numpy.linalg import det, pinv, multi_dot as dot
from tqdm import tqdm
import json

import sys
sys.path.append('../')
from Environments.BlockingTask import *

from PrintState import maze_record, makeGIF
from SARSA import SARSA

OPTIMAL_SCORE = -0.578922

                
class DetSARSA_variant:
    def __init__(self, env, ρ=0.9, η0=0.001, α=0):
        self.env = env
        self.ρ  = ρ
        self.η0 = η0
        self.α  = α
        self.βinit = 10
        self.βfrac = 10000# 10000
        self.ηstart = 10000
        self.bin = 4000
        
        self.N = 4*7
        K = self.N
        self.V = np.eye(self.N) + np.random.uniform(-0.01, 0.01, size=(self.N, K))
        
        
        self.found = False

    def indices(self, state):
        inds = []
        for (i, j) in state:
            inds.append( i*7 + j )
        return inds
            
    def encode(self, state):
        x = np.zeros(self.N)
        inds = self.indices(state)
        x[inds] = 1
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
        x = self.encode( next_state )
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
        ###############################################
#        encoded_successors = []
#        actions = []
#        for a in self.env.valid_actions( state ):
#            x, ns, _, _ = self.φ( state, a, ns=True )
#            encoded_successors.append( x )
#            actions.append( a )
        ###############################################
        
        dets = []
        for x in encoded_successors:
            V = self.V_rd( x )
            dets.append( det( np.dot(V, V.T) ) )
        dets = np.array(dets)
        
        if β > 2:
            m = max(dets)# BAD - high time compexity for large action spaces! 
            # I've done this to avoid numerical errors, (it still samples the same distribution)
            dets /= m

        pmf = np.power( dets, β )

        ind = np.random.choice(range(len(pmf)), p=pmf/np.sum(pmf))
        return actions[ind]
        
    def V_rd(self, x):
        idxs = np.reshape( np.argwhere( x == 1 ), (1, -1))[0]
        return np.copy( self.V[idxs, :] )
    
    def V_wr(self, x, mat):
        idxs = np.reshape( np.argwhere( x == 1 ), (1, -1))[0]
        self.V[idxs, :] += mat
    
    def episode(self, tstart, rcd=False):
        state  = self.env.reset_state()
        # action = self.boltzmannPolicy(state, self.β)
        # ns, reward, done = self.env.step( state, action )
        # x = self.φ( ns )
        x = self.encode( state )

        for t in range(1, 41):
            η = self.η0 * min(1, self.ηstart/(tstart + t) )
            self.β = np.power(self.βinit, (tstart + t)/self.βfrac)
            
            action = self.boltzmannPolicy(state, self.β)
            # Observe reward r_{t+1} and next state s_{t+1}
            next_state, reward, done = self.env.step(state, action)
            # Choose the next action a_{t+1}
#            next_action = self.boltzmannPolicy(next_state, self.β)

#            next_x, _, reward, done = self.φ(next_state, next_action, ns=True)
#            next_x = self.φ(next_state, next_action, ns=False)
            next_x = self.encode( next_state )
            
            
            if rcd: maze_record(tstart+t, 'Blocking task', next_state, 4, 7, self.env.blockers_state)
            
            V_x = self.V_rd( x )
            V_next_x = self.V_rd( next_x )
            
            Q_x = self.α + np.log( det( np.dot(V_x, V_x.T) ) )
            Q_next_x = self.α + np.log( det( np.dot(V_next_x, V_next_x.T) ) )
            
            TD = reward + self.ρ * Q_next_x - Q_x
            grad_Q = 2 * pinv( V_x ).T
            
            # V update
            self.V_wr( x, η * TD * grad_Q )
            
            # self.α += η * TD
            
            self.total_reward += reward
            if (tstart+t) % self.bin == 0:
                self.rec_reward.append( (t + tstart, self.total_reward/self.bin ) )
                self.total_reward = 0
                
            if done:
                self.found = True
#                print('\n\n\n\nCompleted episode, t=',str(t),'\n\n\n')
                return t
            
            
            state = next_state
#            action = next_action
            x = next_x
            
        return 40
    
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
            
        
#    def run(self, max_steps, interval=10000):
#        rec_reward = []
#        total_reward = 0
#        
#        
#        state  = self.env.reset_state()
#        action = self.boltzmannPolicy(state, 1)
##        print('action = ', str(action))
#        x, next_state, reward, done = self.φ(state, action)
#
#        for t in tqdm(range(max_steps)):
#            η = self.η0 * min(1, 10000/(t + 1) )
#            β = np.power(10, t/10000)
#            
#            
#            
##            if t%100:
##                print(self.V)
##            if state == ((2,0), (2,3), (2,6)): print('reached!')
##            next_state, reward, done = self.env.step(state, action)
#            
#            total_reward += reward
#            if (t+1) % 4000 == 0:
#                rec_reward.append( (t, total_reward/4000 ) )
#                total_reward = 0
#                
#                
#            if t > max_steps - 300:#t % interval == 0: 
#                maze_record(t, 'Blocking task', state, 4, 7, self.env.blockers_state)
#
#            if done:
#                print('\n\n\n\n\n\nCompleted episode, t=',str(t),'\n\n\n\n\n\n')
#                # break
#                state  = self.env.reset_state()
#                action = self.boltzmannPolicy(state, β)
#                x, next_state, reward, done = self.φ(state, action)
#                
#            else:
#                next_action = self.boltzmannPolicy(next_state, β)
#                
#                
#                
#                if len([0 for (s,_) in next_state if s==2])==3: 
#                    st = sorted([s for (_,s) in next_state])
##                    print('t = ',str(t),', state = ', str(st))
#                    if st==[0, 3, 6]:
#                        print('!!!! ONE MOVE AWAY')
#                        print('             ----> action = ', str(next_action), ', blocker state = ', str(self.env.blockers_state))
#                
##                print('next_state = ', str(next_state))
##                print('next_action = ', str(next_action))
#                next_x, next_state, reward, done = self.φ(next_state, next_action)
#                
#                
#                V_x = self.V_rd( x )
#                V_next_x = self.V_rd( next_x )
#                
#                Q_x = self.α + np.log(la.det(np.dot(V_x, V_x.T)))
#                Q_next_x = self.α + np.log(la.det(np.dot(V_next_x, V_next_x.T)))
#                
#                TD = reward + self.ρ * Q_next_x - Q_x
#                grad = 2 * la.pinv( self.V_rd( x ) ).T
#                
#                # V update
#                self.V_wr(x, η * TD * grad)
#                
#                self.α += η * TD
#                
#                state = next_state
#                action = next_action
#                x = next_x
#            
#        return rec_reward
            

#tests = [(( ((2,0), (2,3), (2,6)), ((0,0),(0,0),(0,1)) ), (((2, 0), (2, 3), (2, 6)), -1, False) ),
#         (( ((2,0), (2,3), (2,6)), ((0,0),(0,0),(1,0)) ), (((2, 0), (2, 3), (3, 6)), 1, True) )
#         ]

#for q, a in tests:
#    env.blockers_state = 6
#    my_a = env.step(*q)
#    print('passed test!!') if my_a==a else print('Failed test!!\nmy result: ', str(my_a), ',\ntrue result: ', str(a))
# import matplotlib.pyplot as plt

# def plotQuality(V, ep, fname, found, dpi=200, show=True):
#     L = np.dot( V, V.T )
#     S = np.zeros( L.shape )
#     for i in range(L.shape[0]):
#         for j in range(L.shape[1]):
#             S[i, j] = L[i, j]/np.sqrt(L[i, i]*L[j, j])
            
#     q = np.diag(L)[:21].reshape((3, 7))
#     plt.imshow(np.abs( q ), 'coolwarm')
#     plt.colorbar()
#     if found: 
#         plt.title('Quality, ep: '+str(ep)+', found goal')
#     else:
#         plt.title('Quality, episode: '+str(ep))
        
#     plt.xticks(np.arange(7), ('1','2','3','4','5','6','7'))
#     plt.yticks(np.arange(3), ('1','2','3'))
    
#     plt.savefig( fname, dpi=dpi )
#     if show: plt.show()
#     plt.close()

# def plotSimilarity( V, ep, fname, found, dpi=200, show=True, plotL=False ):
#     L = np.dot( V, V.T )
#     S = np.zeros( L.shape )
#     for i in range(L.shape[0]):
#         for j in range(L.shape[1]):
#             S[i, j] = L[i, j]/np.sqrt(L[i, i]*L[j, j])
           
#     if plotL:
#         plt.imshow( L, 'coolwarm')
#     else:
#         plt.imshow( np.abs( S[:21, :21] ), 'coolwarm')
#     plt.colorbar()
    
#     pt1 = 'L, ' if plotL else 'Similarity, '
#     if found: 
#         plt.title(pt1+'ep: '+str(ep)+', found goal')
#     else:
#         plt.title(pt1+'ep: '+str(ep))
    
#     labels = [str((i, j)) for i in [1,2,3,4] for j in [1,2,3,4,5,6,7]]
#     if not plotL:
#         labels = labels[:-7]
#     plt.xticks(np.arange(len(labels)), tuple(labels))
#     plt.yticks(np.arange(len(labels)), tuple(labels))
#     plt.rc('axes', labelsize=8)
#     plt.xticks(rotation=270)
    
#     plt.savefig(fname, dpi=dpi)
#     if show: plt.show()
#     plt.close()

# def DetSARSA():
#     env = BlockerTask()
#     agent = DeterminantalSARSA( env )
    
#     rec_reward = agent.run( 40000 )
#     print('recorded reward: ')
#     print(rec_reward)

#     with open('plots/datasets/Q-det-sarsa-plot-data.json', 'w') as outfile:
#         outfile.write(json.dumps(rec_reward))
    
#     [x, y] = list(zip(*rec_reward))
#     plt.plot(x, y)
#     plt.ylim([-1.1,-0.7])
#     plt.title('Determinantal SARSA Learning Curve')
#     plt.savefig('plots/Qdet-sarsa-learning-curve', dpi=200)
#     plt.show()
    
# def detSarsaMean(repeats = 5):
#     env = BlockerTask()
    
#     rec_reward = {0:[-1 for i in range(repeats)]}
#     for i in tqdm(range(repeats)):
#         agent = DeterminantalSARSA( env )
        
#         rews = agent.run_no_rec( 40000 )
        
#         for (time, r) in rews:
#             if time not in rec_reward:
#                 rec_reward[time] = [r]
#             else:
#                 rec_reward[time].append(r)
    
#     name = 'QMean-det-sarsa'
#     with open('plots/datasets/'+name+'-plot-data.json', 'w') as outfile:
#         outfile.write(json.dumps(rec_reward))
    
#     sarsa_plot(name=name, osogami=True)
    
# def sarsa_plot(name='sarsa', osogami=False, all_runs=True, myQLearningOne=False):
#     with open('plots/datasets/'+name+'-plot-data.json', 'r') as f:
#         rec_reward = json.load(f)
    
#     rec_reward = [(int(k), v) for k, v in rec_reward.items()]
#     rec_reward = sorted(rec_reward, key=lambda x:x[0])

#     repeats = len(rec_reward[0][1])
#     rec_mean = [(time, float(np.mean(listt))) for (time, listt) in rec_reward]
#     recs = [[(time, listt[i]) for (time, listt) in rec_reward] for i in range(repeats)]
    
#     import matplotlib.pyplot as plt
#     [xm, ym] = list(zip(*rec_mean))

#     if all_runs:
#         others = [list(zip(*rec_)) for rec_ in recs]
#         for i, [x, y] in enumerate(others):
#             if i==0:
#                 plt.plot(x, y, c='k', lw=0.3, label='Runs 1-5')
#             else:
#                 plt.plot(x, y, c='k', lw=0.3)
#     plt.plot([min(xm), max(xm)], [OPTIMAL_SCORE, OPTIMAL_SCORE], c='g', label='Optimal')
#     plt.plot(xm, ym, 'r', label='Average')

#     if osogami:
#         with open('plots/datasets/Osogami-plot-data.json', 'r') as f:
#             osg = json.load(f)

#         plt.errorbar(osg['x'], osg['y'], osg['yerr'], label='Osogami\'s data')

#     if myQLearningOne:
#         with open('plots/datasets/QMean-det-sarsa-plot-data.json', 'r') as f:
#             Qrec_reward = json.load(f)

#         Qrec_reward = [(int(k), v) for k, v in Qrec_reward.items()]
#         Qrec_reward = sorted(Qrec_reward, key=lambda x:x[0])

#         Qrepeats = len(Qrec_reward[0][1])
#         Qrec_mean = [(time, float(np.mean(listt))) for (time, listt) in Qrec_reward]
#         # recs = [[(time, listt[i]) for (time, listt) in Qrec_reward] for i in range(Qrepeats)]

#         [xm, ym] = list(zip(*Qrec_mean))
#         plt.plot(xm, ym, c='y', label='My Q-Learning one')

#     plt.ylim([-1.05,-0.55])
#     plt.legend()
#     plt.title('Blocker task')
#     plt.xlabel('# actions')
#     plt.ylabel('avg. reward per action')
#     plt.savefig('plots/'+name+'-learning-curve', dpi=200)
#     plt.show()

# if __name__=="__main__":
#     env = BlockerTask()
#     d = DetSARSA_variant(env)
#     d.run_rec_hidden_state(40000)
    
# DetSARSA()
# detSarsaMean(repeats = 2)
#sarsa()
#sarsa_plot(name='QMean-det-sarsa', osogami=True)