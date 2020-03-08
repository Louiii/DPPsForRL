#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 17:02:27 2019

@author: louisrobinson
"""
from DeterminantalSARSA.DeterminantalSARSA import DeterminantalSARSA
from DeterminantalSARSA.DeterminantalSARSA_Model import DetSARSA_Model, Model
from Environments.BlockingTask import BlockerTask
from SARSA.SARSA import SARSA
import matplotlib.pyplot as plt
import json
import numpy as np
from tqdm import tqdm
from PrintState import makeGIF

OPTIMAL_SCORE = -0.578922

def DetSARSA():
    env = BlockerTask()
    agent = DeterminantalSARSA( env )
    
    rec_reward = agent.run( 40000 )
    print('recorded reward: ')
    print(rec_reward)
    
    [x, y] = list(zip(*rec_reward))
    plt.plot(x, y)
    plt.ylim([-1.1,-0.7])
    plt.title('Determinantal SARSA Learning Curve')
    plt.savefig('plots/det-sarsa-individual-learning-curve', dpi=120)
    plt.show()
    
#    plotQuality( agent.V, 'plots/dppQuality' )
#    plotSimilarity( agent.V, 'plots/dppSimilarity' )

def DetSARSA_Model_run(repeats = 2):
    rec_reward = {0:[-1 for i in range(repeats)]}
    for i in tqdm(range(repeats)):
        env = BlockerTask()
        m = Model()
        agent = DetSARSA_Model( env, m )
        
        rews = agent.run_no_rec( 16000 )
        
        for (time, r) in rews:
            if time not in rec_reward:
                rec_reward[time] = [r]
            else:
                rec_reward[time].append(r)
    
    name = 'det-sarsa-model'
    with open('plots/datasets/'+name+'-plot-data.json', 'w') as outfile:
        outfile.write(json.dumps(rec_reward))
    sarsa_plot(name=name, osogami=True)

def detSarsaMean(repeats = 5):
    env = BlockerTask()
    
    rec_reward = {0:[-1 for i in range(repeats)]}
    for i in tqdm(range(repeats)):
        agent = DeterminantalSARSA( env )
        
        rews = agent.run_no_rec( 40000 )
        
        for (time, r) in rews:
            if time not in rec_reward:
                rec_reward[time] = [r]
            else:
                rec_reward[time].append(r)
    
    name = 'det-sarsa'
    with open('plots/datasets/'+name+'-plot-data.json', 'w') as outfile:
        outfile.write(json.dumps(rec_reward))

#    makeGIF('plots/temp-plots', 'plots/DET-SARSA-BlockerTaskGIF')
    sarsa_plot(name=name, osogami=True)
    
def sarsa(repeats = 5):
    env = BlockerTask()
    
    rec_reward = {0:[-1 for i in range(repeats)]}
    rec_any = False
    for i in tqdm(range(repeats)):
        agent = SARSA( env, 0.1 )
        
        if i==repeats-1: 
            rec_any=True
            
        rews = agent.run( 200000, interval=1, rec_any=rec_any )
        
        for (time, r) in rews:
            if time not in rec_reward:
                rec_reward[time] = [r]
            else:
                rec_reward[time].append(r)
    
    with open('plots/datasets/sarsa-plot-data.json', 'w') as outfile:
        outfile.write(json.dumps(rec_reward))


    makeGIF('plots/temp-plots/temp-plots1', 'plots/SARSA-BlockerTaskGIF')
    sarsa_plot()
    
def sarsa_plot(name='sarsa', osogami=False, all_runs=True, myQLearningOne=False):
    with open('plots/datasets/'+name+'-plot-data.json', 'r') as f:
        rec_reward = json.load(f)
    
    rec_reward = [(int(k), v) for k, v in rec_reward.items()]
    rec_reward = sorted(rec_reward, key=lambda x:x[0])

    repeats = len(rec_reward[0][1])
    rec_mean = [(time, float(np.mean(listt))) for (time, listt) in rec_reward]
    recs = [[(time, listt[i]) for (time, listt) in rec_reward] for i in range(repeats)]
    
    import matplotlib.pyplot as plt
    [xm, ym] = list(zip(*rec_mean))

    if all_runs:
        others = [list(zip(*rec_)) for rec_ in recs]
        for i, [x, y] in enumerate(others):
            if i==0:
                plt.plot(x, y, c='k', lw=0.3, label='Runs 1-5')
            else:
                plt.plot(x, y, c='k', lw=0.3)
    plt.plot([min(xm), max(xm)], [OPTIMAL_SCORE, OPTIMAL_SCORE], c='g', label='Optimal')
    plt.plot(xm, ym, 'r', label='Average')

    if osogami:
        with open('plots/datasets/OsogamiRaymond/Osogami-plot-data.json', 'r') as f:
            osg = json.load(f)

        plt.errorbar(osg['x'], osg['y'], osg['yerr'], label='Osogami\'s data')

    if myQLearningOne:
        with open('plots/datasets/OsogamiRaymond/QMean-det-sarsa-plot-data.json', 'r') as f:
            Qrec_reward = json.load(f)

        Qrec_reward = [(int(k), v) for k, v in Qrec_reward.items()]
        Qrec_reward = sorted(Qrec_reward, key=lambda x:x[0])

        Qrepeats = len(Qrec_reward[0][1])
        Qrec_mean = [(time, float(np.mean(listt))) for (time, listt) in Qrec_reward]
        # recs = [[(time, listt[i]) for (time, listt) in Qrec_reward] for i in range(Qrepeats)]

        [xm, ym] = list(zip(*Qrec_mean))
        plt.plot(xm, ym, c='y', label='My Q-Learning one')

    plt.ylim([-1.05,-0.55])
    plt.legend()
    plt.title('Blocker task')
    plt.xlabel('# actions')
    plt.ylabel('avg. reward per action')
    plt.savefig('plots/'+name+'-learning-curve', dpi=200)
    plt.show()
    


#sarsa_plot(name='det-sarsa', osogami=True, all_runs=False, myQLearningOne=True)

# DetSARSA()
DetSARSA_Model_run()
# detSarsaMean(2)
# sarsa(2)