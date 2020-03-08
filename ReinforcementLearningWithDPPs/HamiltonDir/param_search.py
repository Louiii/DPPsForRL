import sys
import json
import numpy as np
from numpy.linalg import det, pinv, multi_dot as dot


# environment
class BlockerTask:
    def __init__(self):
        ''' state is the coords of the agent
        blockers state is the index of the empty square in the bottom row
        '''
        self.r, self.c = 4, 7
        self.ind_actions = [(0,0),(0,1),(1,0),(0,-1),(-1,0)]
        self.onBoard = lambda r, c: r >= 0 and r < self.r and c >= 0 and c < self.c
        
        self.all_actions = [(a, b, c) for a in self.ind_actions for b in self.ind_actions for c in self.ind_actions]
        
        self.valid_actions = lambda state: self.all_actions
        
        self.state = self.reset_state()
    
    def reset_state(self):
        cols = sorted(np.random.choice(7, 3, replace=False))
        self.state = tuple([(0, c) for c in cols])
        self.blockers_state = np.random.choice([0, 3, 6])
        return self.state
    
    def goal_check(self, state):
#        ((r1,c1), (r2,c2), (r3,c3)) = state
        if (3, self.blockers_state) in list(state):
            return True
        return False
    
    def update_blockers(self):
        ((r1,c1), (r2,c2), (r3,c3)) = self.state# should be in rows 0,1,2
        positions = list(self.state)
        if self.blockers_state==0:
            if (2,0) in positions:# we must move to block the left side
                self.blockers_state = 3# moved to left, center open
                if (2,3) in positions:# if center is being attacked, move
                    self.blockers_state = 6
                    
        elif self.blockers_state==3:
            if (2,3) in positions:# we must move to block the center
                if (2,6) in positions:# if the right side is attacked, move right
                    self.blockers_state = 0
                else:# move left
                    self.blockers_state = 6
            
        elif self.blockers_state==6:
            if (2, 6) in positions:# we must move to block the right side
                self.blockers_state = 3# moved to right, center open
                if (2,3) in positions:# if center is being attacked, move
                    self.blockers_state = 0
        
            
    
    def ind_step(self, pos, ac, other_agents):
        (r, c) = pos
        (dr, dc) = ac
        nr, nc = r+dr, c+dc
        if not self.onBoard(nr, nc) or (nr, nc) in other_agents:
            return pos
        # check if its hit a blocker
        if nr==3:# top row
            if nc!=self.blockers_state:
                return pos
        return (nr, nc)
    
    def step(self, state, action, searchingPolicy=False):
        ''' Take action
            Deterministically move blockers
        '''
        prevBlkr = self.blockers_state

        (p1, p2, p3) = state
        (a1, a2, a3) = action
        # move agent 1:
        np1 = self.ind_step(p1, a1, {p2, p3})
        # move agent 2:
        np2 = self.ind_step(p2, a2, {np1, p3})
        # move agent 3:
        np3 = self.ind_step(p3, a3, {np1, np2})
        
        self.state = (np1, np2, np3)
    
        if self.goal_check(self.state):
            return self.state, 1, True
        
        self.update_blockers()
        if searchingPolicy:
            self.blockers_state = prevBlkr
        
        return self.state, -1, False

# detsarsa
class Table_model:
    def __init__(self):
        self.step_apprx = {}

    def update(self, s, a, s_dash, r):
        if (s, a) in self.step_apprx:
            (old_ns, multiple) = self.step_apprx[(s,a)]

            if multiple:
                if s_dash not in old_ns:
                    old_ns.append(s_dash)
                    self.step_apprx[(s,a)] = (old_ns, True)
            else:
                if s_dash != old_ns:
                    self.step_apprx[(s,a)] = ([old_ns, s_dash], True)
        else:
            self.step_apprx[(s,a)] = (s_dash, False)

    def predict(self, s, a):
        if (s, a) in self.step_apprx:
            (old_ns, multiple) = self.step_apprx[(s,a)]
            if multiple:
                return old_ns[np.random.randint(len(old_ns))]
            return old_ns
        return s

class DetSARSA_Model:
    def __init__(self, env, m, ρ=0.9, α=0, params=[10, 30000, 30000, 0.001]):
        self.env = env
        self.ρ  = ρ
        self.α  = α

        [bi, bf, es, ec] = params
        self.η0 = ec
        self.βinit = bi
        self.βfrac = bf
        self.ηstart = es

        self.bin = 4000
        self.λ = 1

        self.model = m
        
        self.N = 4*7
        K = self.N
        self.V = np.eye(self.N) + np.random.uniform(-0.01, 0.01, size=(self.N, K))
        
        self.max_n_ep = 40
        
        self.found = False
        self.i = 0# counter used for rendering
    
    def indices(self, state):
        return [i*7 + j for (i, j) in state]
                    
    def φ(self, state, action, ns=True):
        next_state = self.model.predict(state, action)
        return self.indices(next_state)
    
    def boltzmannPolicy(self, state, β):
        ''' compute all possible successor states
            encode all possible successor states
            compute det(L)^beta for all successor states
            sample action from pdf
        '''
        dets = []
        for a in self.env.valid_actions( state ):
            x = self.φ( state, a )
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
        return self.env.valid_actions( state )[ind]

    def updateβη(self, time):
        self.η = self.η0 * min(1, self.ηstart/time )
        self.β = np.power(self.βinit, time/self.βfrac)

    def V_rd(self, x):
        return np.copy( self.V[x, :] )
    
    def V_wr(self, x, mat):
        self.V[x, :] += mat
        
    def episode(self, tstart, rcd=False):
        state  = self.env.reset_state()
        action = self.boltzmannPolicy(state, self.β)
        x = self.φ(state, action, ns=False)
        V_x = self.V_rd( x )
        Q_x = self.α + np.log( det(np.dot(V_x, V_x.T)) )

        for t in range(1, self.max_n_ep+1):
            self.updateβη(tstart + t)
            
            # Observe reward r_{t+1} and next state s_{t+1}
            next_state, reward, done = self.env.step(state, action)

            self.model.update(state, action, next_state, reward)
            # Choose the next action a_{t+1}
            next_action = self.boltzmannPolicy(next_state, self.β)

            next_x = self.φ(next_state, next_action, ns=False)
            V_next_x = self.V_rd( next_x )
            Q_next_x = self.α + np.log( det(np.dot(V_next_x, V_next_x.T)) )
            
            TD = reward + self.ρ * Q_next_x - Q_x
            grad_Q = 2 * pinv( V_x ).T
            
            # V update
            self.V_wr( x, self.η * TD * grad_Q )
            
            self.total_reward += reward
            if (tstart+t) % self.bin == 0:
                self.rec_reward.append( (t + tstart, self.total_reward/self.bin ) )
                self.total_reward = 0
                
            if done:
                self.found = True
                return t

            # update variables
            state, action, x, V_x, Q_x = next_state, next_action, next_x, V_next_x, Q_next_x
            
        return self.max_n_ep

    def run(self, max_steps):
        self.rec_reward = []
        self.total_reward = 0
        self.β = 1
        
        t = 0
        ep = 0
        while t < max_steps:
            ep += 1
            dt = self.episode(t, False)
            t += dt
        
        return self.rec_reward


def updateRewardList(rec_reward, rews):
    for (time, r) in rews:
        if time not in rec_reward:
            rec_reward[time] = [r]
        else:
            rec_reward[time].append(r)
    return rec_reward

def computeMean(rec_reward):
    average = {}
    for time, rewards in rec_reward.items():
        average[time] = sum(rewards)/len(rewards)
    rec_reward['average'] = average
    return rec_reward

def detSarsaModelMean(path, params, repeats=5, steps=250000):
    rec_reward = {0:[-1 for i in range(repeats)]}
    for i in range(repeats):
        env = BlockerTask()
        m = Table_model()
        agent = DetSARSA_Model( env, m, params=params )
        rews = agent.run( steps )
        rec_reward = updateRewardList(rec_reward, rews)
    rec_reward = computeMean(rec_reward)

    with open(path, 'w') as outfile:
        outfile.write(json.dumps(rec_reward))


# get the parameters
i = int(sys.argv[1])-1
with open("parameter_list.json", 'r') as f:
	parameter_list = json.load(f)
params = parameter_list[i]

# run 5 times and average
detSarsaModelMean("data/d"+str(i)+".json", params, repeats=5, steps=300000)
