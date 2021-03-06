from Environments.TransportationTask import TransportationTask, record, makeGIF
import numpy as np
from numpy.linalg import det, pinv, multi_dot as dot


class DeterminantalSARSA:
    def __init__(self, env, ρ=0.9, η0=0.001, α=0):
        self.env = env
        self.ρ  = ρ
        self.η0 = η0
        self.α  = α
        self.β10e4 = 10
        self.βfrac = 40000# CHANGED!10000
        self.ηstart = 40000# CHANGED!10000
        self.bin = 400
        self.λ = 1
        self.ep_len = 100
       

        self.N = 6*7+4
        K = self.N
        self.V = np.eye(self.N) + np.random.uniform(-0.01, 0.01, size=(self.N, K))
        
        
        self.found = False
    
    def indices(self, state):
        ((r1, c1), gr1, (r2, c2), gr2) = state
        inds = [r1*7 + c1, r2*7 + c2]
        if gr1==1: inds.append(self.N-4)
        if gr1==2: inds.append(self.N-3)
        if gr2==1: inds.append(self.N-2)
        if gr2==2: inds.append(self.N-1)
        return inds
            
    def encode(self, state):
        x = np.zeros(self.N)# x = [0]*self.N
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
        for a in self.env.valid_actions( state )[1:]:
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
            m = max(dets)# BAD! I've done this to avoid numerical errors
            dets /= m

        pmf = np.power( dets, β )

        ind = np.random.choice(range(len(pmf)), p=pmf/np.sum(pmf))
        return actions[ind]
        
    def ei_s(self, x):
        idxs = np.reshape( np.argwhere( x == 1 ), (1, -1))[0]
        ei = np.zeros((len(idxs), len(x)))
        for i, j in enumerate(idxs):
            ei[i, j] = 1
        return ei
    
    def V_rd(self, x):
        idxs = np.reshape( np.argwhere( x == 1 ), (1, -1))[0]
        return np.copy( self.V[idxs, :] )
    
    def V_wr(self, x, mat):
        idxs = np.reshape( np.argwhere( x == 1 ), (1, -1))[0]
        self.V[idxs, :] += mat
        
    def episode(self, tstart, rcd=False):
        self.env.reset_state()
        state  = self.env.start_state
        action = self.boltzmannPolicy(state, self.β)
        x = self.φ(state, action, ns=False)

        for t in range(1, self.ep_len+1):
            η = self.η0 * min(1, self.ηstart/(tstart + t) )
            self.β = np.power(self.β10e4, (tstart + t)/self.βfrac)
            
            # Observe reward r_{t+1} and next state s_{t+1}
            next_state, reward, done = self.env.step(state, action)
            # Choose the next action a_{t+1}
            next_action = self.boltzmannPolicy(next_state, self.β)

            next_x = self.φ(next_state, next_action, ns=False)
            
            
            
            if rcd:
            	(p1, gr1, p2, gr2) = next_state
            	xax = 'state: '+str((p1, p2))+', action: '+str(next_action)+', grasping: '+str((gr1,gr2))
            	tt = 'obj: '+str(self.env.object)+', r = '+str(reward)+', ended = '+str(done)+', : iter = '
            	record(t+tstart, tt, xax, next_state, self.env.object, self.env, up=False)

            V_x = self.V_rd( x )
            V_next_x = self.V_rd( next_x )
            
            Q_x = self.α + np.log( det(np.dot(V_x, V_x.T)) )
            Q_next_x = self.α + np.log( det(np.dot(V_next_x, V_next_x.T)) )
            
            TD = reward + self.ρ * Q_next_x - Q_x
            grad_Q = 2 * pinv( V_x ).T
            
            # V update
#            print(self.α)
#            self.λ *= 0.9999
            self.V_wr( x, η * TD * grad_Q )# - (V_x - self.ei_s(x)) * η * self.λ )
            
#            self.α += η * TD# grad_α = 1
            
            self.total_reward += reward
            if (tstart+t) % self.bin == 0:
                self.rec_reward.append( (t + tstart, self.total_reward/self.bin ) )
                self.total_reward = 0
                
            if done:
                self.found = True
                print('\n\n\n\nCompleted episode, t=',str(t),'\n\n\n')
                return t
            
            
            state = next_state
            action = next_action
            x = next_x
            
        return self.ep_len
    
    # def run(self, max_steps):
    #     self.rec_reward = []
    #     self.total_reward = 0
    #     self.β = 1
        
    #     t = 0
    #     rcd = False
    #     ep = 0
    #     while t < max_steps:
    #         ep += 1
    #         if ep%100==0: 
    #             print('time: '+str(t))
                
    #         if ep%10==0:
    #             plotQuality( self.V, ep, 'plots/temp-plots/temp-plots1/V'+str(ep), self.found, dpi=100, show=False )
    #             plotSimilarity( self.V, ep, 'plots/temp-plots/temp-plots2/V'+str(ep), self.found, dpi=100, show=False, plotL=True )
    #             plotSimilarity( self.V, ep, 'plots/temp-plots/temp-plots3/V'+str(ep), self.found, dpi=100, show=False )
            
    #         dt = self.episode(t, rcd)
            
    #         t += dt
        
    #     makeGIF('plots/temp-plots/temp-plots1', 'plots/changing-quality')
    #     makeGIF('plots/temp-plots/temp-plots2', 'plots/changing-L')
    #     makeGIF('plots/temp-plots/temp-plots3', 'plots/changing-similarity')
    #     return self.rec_reward
    
    def run_no_rec(self, max_steps, ep_length=100):
        self.rec_reward = []
        self.total_reward = 0
        self.β = 1
        self.ep_len = ep_length
        self.bin = 3000
        
        t = 0
        ep = 0
        rcd = False
        while t < max_steps:
            ep += 1

            # if t > 200:
            # 	rcd = False

            if t > max_steps - 2*self.ep_len:
            	print('recording episode...')
            	rcd = True
            print('time: '+str(t)+', β: '+str(self.β))
            dt = self.episode(t, rcd)
            
            t += dt
        print('making gif...')
        makeGIF('plots/temp-plots/temp-plots1', 'plots/DET-SARSA-ObjTransportation')
        return self.rec_reward
            




env = TransportationTask()
detsarsa = DeterminantalSARSA(env, ρ=0.95, η0=0.005, α=0)#, ρ=0.98)
rewards = detsarsa.run_no_rec(int(5e4), 500)

import matplotlib.pyplot as plt
[x,y] = list(zip(*rewards))
plt.plot(x, y)
plt.savefig('plots/DetSARSA-learning-curve-Transportation', dpi=400)
plt.show()
