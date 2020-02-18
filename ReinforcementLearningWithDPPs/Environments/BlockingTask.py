#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:50:49 2019

@author: louisrobinson
"""
from PrintState import maze_record, makeGIF
import numpy as np


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
#        next_state = tuple([(i+di, j+dj) for (i, j), (di, dj) in zip(state, action)])
        
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


#    
#tests = [(( ((2,0), (2,3), (2,6)), ((0,0),(0,0),(0,1)) ), (((2, 0), (2, 3), (2, 6)), -1, False) ),
#         (( ((2,0), (2,3), (2,6)), ((0,0),(0,0),(1,0)) ), (((2, 0), (2, 3), (3, 6)), 1, True) )
#         ]
#
#env = BlockerTask()
#for q, a in tests:
#    env.blockers_state = 6
#    my_a = env.step(*q)
#    print('passed test!!') if my_a==a else print('Failed test!!\nmy result: ', str(my_a), ',\ntrue result: ', str(a))
#
########################################
# AC/gym wrapper

class BlockerEnv():
    def __init__(self):
        self.ev = BlockerTask()
        self.action_dim = len(self.ev.all_actions)# dr_i, dc_i, i in 1,2,3
        self.state_dim  = 6# r_i, c_i, i in 1,2,3
        self.state = self.reset()

        self.render_count = 0

    def reset(self):
        self.state = self.ev.reset_state()
        return self.convert_state(self.state)

    def render(self, dpi=40):
        maze_record(self.render_count, 'Blocking task', self.state, 4, 7, self.ev.blockers_state, dpi=dpi)
        self.render_count += 1

    def close(self):
        return

    def convert_state(self, s):
        ((r1,c1), (r2,c2), (r3,c3)) = s
        return np.array([r1, c1, r2, c2, r3, c3])

    def step(self, a):
        # convert action from np to tuple
        action = self.ev.all_actions[a]#((a[0],a[1]), (a[2],a[3]), (a[4],a[5]))
        next_state, reward, done = self.ev.step(self.state, action)
        self.state = next_state
        # convert state back from tuple to np
        next_state = self.convert_state(next_state)
        return next_state, reward, done, None

    def output_GIF(self):
        makeGIF('../plots/temp-plots/temp-plots1', '../plots/ActorCriticBlockerTaskGIF')
