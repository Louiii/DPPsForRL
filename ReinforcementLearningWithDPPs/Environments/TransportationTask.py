#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:50:49 2019

@author: louisrobinson
"""

import numpy as np


class TransportationTask:
	def __init__(self):
		''' 
		origin is (0,0) in the bottom left
		(5, 0) is the top left, (0, 6) is the bottom right, (1,0) is up etc.
		grasping is 0,1,2 for each agent- indicating free, grasping-left, grasping-right
		state is the coords of the agent
		'''
		self.start_state = ((0,0), 0, (0,4), 0)
		self.r, self.c = 6, 7# rows, columns
		self.onBoard = lambda r, c: r >= 0 and r < self.r and c >= 0 and c < self.c
		self.walls = [(1, i) for i in [0,1,3,4,5,6]] + [(4,3),(0,6)]
		self.ind_actions = [(0,0),(0,1),(1,0),(0,-1),(-1,0)]
		self.all_actions = [(a, b) for a in self.ind_actions for b in self.ind_actions]
		
		# just because my other environment had this function
		self.valid_actions = lambda state: self.all_actions

		self.reset_state()
	
	def reset_state(self):
		self.state = self.start_state
		self.object = (3,3)
		return self.state

	def goal_check(self, state):
		return self.object in [(5,2),(5,3),(5,4)]

	def neighbourState(self, pos1, pos2):
		(r, c) = pos1 
		neighbours1 = [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
		return pos2 in neighbours1

	# def oppositeActionRC(self, a1, a2, p1, p2):
	# 	(r1, c1), (r2, c2) = p1, p2
	# 	columnNeighbours = c1==c2+1 or c1==c2-1

	# 	(dr1, dc1), (dr2, dc2) = a1, a2

	# 	if columnNeighbours and 

	# 	return (0,0)==(dr1+dr2, dc1+dc2)

	def graspable(self, pos):
		(r, c) = self.object
		r_l = {(r, c-1):1, (r, c+1):2}
		if pos in r_l:# left and right of object
			return r_l[pos]
		else:
			return 0

	def ind_step(self, pos, gr, action):
		''' ignores the other agent '''
		(r, c)   = pos
		(dr, dc) = action
		nr, nc = r+dr, c+dc
		if not self.onBoard(nr, nc):
			return pos, gr
		if (nr, nc) in self.walls or (nr, nc)==self.object:
			return pos, gr
		if gr==0:
			gr = self.graspable((nr, nc))
		return (nr, nc), gr

	def pushable(self, pos1, pos2, action):
		(ro, co) = self.object
		(r1, c1), (r2, c2) = pos1, pos2
		(dr, dc) = action
		nr1, nc1, nr2, nc2 = r1+dr, c1+dc, r2+dr, c2+dc
		nro, nco = ro+dr, co+dc
		if self.onBoard(nr1, nc1) and self.onBoard(nr2, nc2) and (nr1, nc1) not in self.walls and (nr2, nc2) not in self.walls and (nro, nco) not in self.walls:
			return True
		return False

	def ind_step_oneAgentConnected(self, pos, gr, act, otherPos):
		# BUT MUST ACCOUNT FOR THE OTHER AGENT AS AN OBSTACLE
		(r, c)   = pos
		(dr, dc) = act
		nr, nc = r+dr, c+dc
		if not self.onBoard(nr, nc) or (nr, nc)==otherPos:
			return pos, gr
		if (nr, nc) in self.walls or (nr, nc)==self.object:
			return pos, gr
		if gr==0:
			gr = self.graspable((nr, nc))
		return (nr, nc), gr

	def step(self, state, action, searchingPolicy=False):
		prevObj = self.object

		(pos1, gr1, pos2, gr2) = state
		(act1, act2) = action

		reward = 0
		if gr1 == 0 and gr2 == 0:# neither are holding on
			npos1, gr1 = self.ind_step(pos1, gr1, act1)
			npos2, gr2 = self.ind_step(pos2, gr2, act2)

			# print('----')
			# print((pos1, gr1, act1, npos1, gr1))
			# print((pos2, gr2, act2, npos2, gr2))
			# print('======')

			if gr1 in [1, 2] or gr2 in [1, 2]:
				if gr1 in [1, 2] and gr2 in [1, 2]:
					reward = 2
				else:
					reward = 1

			# agent 1 moves, then agent 2
			# neighbour, and agent1 is moving into agent2, and agent2 is not moving
			if self.neighbourState(pos1, pos2) and npos1==pos2 and npos2==pos2:# neither can move!
				npos1, npos2 = pos1, pos2
			elif npos1==npos2:# check they are not going into the same place
				npos1, npos2 = pos1, pos2

			# # agent 1 moves, then agent 2
			# if self.neighbourState(pos1, pos2):
			# 	if npos1==pos2:
			# 		if npos2==pos2:# neither can move!
			# 			npos1, npos2 = pos1, pos2
			# 	else:# both can move

			# 	if (act1==act2 or self.oppositeActionRC(act1, act2, pos1, pos2)):
			# 	print('=>')
			# 	if act1==act2:# same move
			# 		if npos1==pos1 or npos2==pos2:# obstacles for either
			# 			npos1, npos2 = pos1, pos2
			# 	else:# oppositeMove, can't move!
			# 		npos1, npos2 = pos1, pos2
			# else:
			# 	print('->')
			# 	if npos1 != pos2 and npos2 == npos1:
			# 		npos2 = pos2# move 1 valid, move 2 is invalid
			# 	if npos1 == pos2:
			# 		npos1 = pos1# move 1 is invalid
			# 		if npos2 == pos1:
			# 			npos2 = pos2# move 2 is also invalid


			# # agent 1 moves, then agent 2
			# if self.neighbourState(pos1, pos2) and (act1==act2 or self.oppositeActionRC(act1, act2, pos1, pos2)):
			# 	print('=>')
			# 	if act1==act2:# same move
			# 		if npos1==pos1 or npos2==pos2:# obstacles for either
			# 			npos1, npos2 = pos1, pos2
			# 	else:# oppositeMove, can't move!
			# 		npos1, npos2 = pos1, pos2
			# else:
			# 	print('->')
			# 	if npos1 != pos2 and npos2 == npos1:
			# 		npos2 = pos2# move 1 valid, move 2 is invalid
			# 	if npos1 == pos2:
			# 		npos1 = pos1# move 1 is invalid
			# 		if npos2 == pos1:
			# 			npos2 = pos2# move 2 is also invalid
		elif gr1 in [1, 2] and gr2 in [1, 2]:
			if act1==act2 and self.pushable(pos1, pos2, act1):
				# if nothing is in the way both agents, and the object can move
				(ro, co) = self.object
				(dr, dc) = act1
				self.object = (ro+dr, co+dc)

				(r1, c1), (r2, c2) = pos1, pos2
				npos1, npos2 = (r1+dr, c1+dc), (r2+dr, c2+dc)
			else:
				npos1, npos2 = pos1, pos2
		elif gr1 in [1, 2]:
			npos1 = pos1 # agent1 can't move
			# agent2 may be able to
			npos2, gr2 = self.ind_step_oneAgentConnected(pos2, gr2, act2, pos1)
			if gr2 in [1, 2]:
				reward = 1
		else:# gr2
			npos2 = pos2 # agent2 can't move
			# agent1 may be able to
			npos1, gr1 = self.ind_step_oneAgentConnected(pos1, gr1, act1, pos2)
			if gr1 in [1, 2]:
				reward = 1
		self.state = (npos1, gr1, npos2, gr2)


		if self.goal_check(self.state):
			if searchingPolicy:
			# if we don't want the hidden state (of the object) to change
				self.object = prevObj

			return self.state, 10, True

		if searchingPolicy:
			self.object = prevObj
		
		return self.state, reward, False

##################################
# gym/AC Wrapper

class TransportationEnv():
    def __init__(self):
        self.ev = TransportationTask()
        self.action_dim = len(self.ev.all_actions)# dr_i, dc_i, i in 1,2,3
        self.state_dim  = 6# r_i, c_i, i in 1,2,3
        self.state = self.reset()

        self.render_count = 0

    def reset(self):
        self.state = self.ev.reset_state()
        return self.convert_state(self.state)

    def render(self, dpi=80):
        record(self.render_count, 'Transportation task', 'x', self.state, self.ev.object, self.ev, dpi=dpi)
        self.render_count += 1

    def close(self):
        return

    def convert_state(self, s):
        ((r1, c1), gr1, (r2, c2), gr2) = s
        return np.array([r1, c1, gr1, r2, c2, gr2])

    def step(self, a):
        # convert action from np to tuple
        action = self.ev.all_actions[a]#((a[0],a[1]), (a[2],a[3]), (a[4],a[5]))
        next_state, reward, done = self.ev.step(self.state, action)
        self.state = next_state
        # convert state back from tuple to np
        next_state = self.convert_state(next_state)
        return next_state, reward, done, None

    def output_GIF(self):
        makeGIF('../plots/temp-plots/temp-plots1', '../plots/ActorCriticTransportationTaskGIF')

# env = TransportationTask()
# state = ((2, 2), 0, (1, 2), 0)
# print( (state, env.object) )
# (state, reward, ended) = env.step( state, ((0,-1), (1,0)) )
# print( (state, reward, ended, env.object) )
# (state, reward, ended) = env.step( state, ((-1,0), (1,0)) )
# print( (state, reward, ended, env.object) )
# (state, reward, ended) = env.step( state, ((1,0), (0,0)) )
# print( (state, reward, ended, env.object) )
# (state, reward, ended) = env.step( state, ((-1,0), (0,0)) )
# print( (state, reward, ended, env.object) )

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FixedLocator
from pylab import fill
import imageio
import os
import shutil

def makeGIF(plotroot, filename, interval=0.1, end_pause=12):
    # Set the directory you want to start from
    rootDir = './'+plotroot+'/'
    files={}
    for dirName, subdirList, fileList in os.walk(rootDir):
        for f in fileList:
            if f[0]=='V':
                iteration = f[1:-4]
                files[int(iteration)] = f
    keys = list(files.keys())
    keys.sort()
    
    for k in keys:
        path = rootDir+files[k]
        im = imageio.imread(path)
        (h, w, _) = im.shape

    images = []
    for k in keys:
        images.append(imageio.imread(rootDir+files[k]))
    for _ in range(end_pause):
        images.append(imageio.imread(rootDir+files[keys[-1]]))
    kargs = { 'duration': interval }
    imageio.mimsave(filename+'.gif', images, **kargs)

    # delete all of the png files
    shutil.rmtree(rootDir)
    os.mkdir(plotroot)

def printObjectTransportation(agents, obj, grasping, obstacles, goals, r, c, **kwargs):
    fig, ax1 = plt.subplots()

    if 'title' in kwargs: ax1.set_title(kwargs['title'], fontsize=18)
    if 'xax' in kwargs: ax1.set_xlabel(kwargs['xax'], fontsize=15)

    # Shade and label agents
    for i, (sy, sx) in zip([1,2], agents):
        ax1.text(sx+0.3, sy+0.3, '$A_'+str(i)+'$', color="b", fontsize=20)
        fill([sx,sx+1,sx+1,sx], [sy,sy,sy+1,sy+1], 'b', alpha=0.2, edgecolor='b')

    # Shade and label goal cells
    for (gy, gx) in goals:
    	fill([gx,gx+1,gx+1,gx], [gy,gy,gy+1,gy+1], 'g', alpha=0.2, edgecolor='g')
    ax1.text(goals[1][1]-0.3, goals[1][0]+0.4, 'Home base', color="g", fontsize=15)

    # Make obstacles:
    for (y, x) in obstacles: 
        fill([x,x+1,x+1,x], [y,y,y+1,y+1], 'k', alpha=0.2, edgecolor='k')

    # make object
    [y, x] = obj
    ax1.text(x+0.05, y+0.37, 'Object', color="k", fontsize=14)
    fill([x,x+1,x+1,x], [y+0.3,y+0.3,y+0.7,y+0.7], 'r', alpha=0.5, edgecolor='k')

    # make grid lines on center
    plt.xticks([0.5+i for i in range(c)], [i for i in range(c)])
    plt.yticks([0.5+i for i in range(r)], [i for i in range(r)])
    plt.xlim(0,c)
    plt.ylim(0,r)

    # make grid
    minor_locator1 = AutoMinorLocator(2)
    minor_locator2 = FixedLocator([j for j in range(r)])
    plt.gca().xaxis.set_minor_locator(minor_locator1)
    plt.gca().yaxis.set_minor_locator(minor_locator2)
    plt.grid(which='minor')

    if 'filename' in kwargs:
        plt.savefig(kwargs['filename'], dpi=kwargs['dpi'])
    plt.close(fig)

def record(iteration, tt, xax, state, obj, env, up=True, dpi=80):
    agents, grasping = [state[0], state[2]], [state[1], state[3]]
    goals = [(5,2),(5,3),(5,4)]
    fn = "plots/temp-plots/temp-plots1/V"+str(iteration)+".png"
    if up: fn = "../"+fn
    printObjectTransportation(agents, obj, grasping, env.walls, goals, env.r, env.c, 
    	show=False, filename=fn, title=tt+str(iteration), cbarlbl="Value", xax=xax, dpi=dpi)

