from DeterminantalSARSA.DeterminantalSARSA import DeterminantalSARSA
from DeterminantalSARSA.DeterminantalQlearning import DeterminantalQlearning
from DeterminantalSARSA.DetSarsaDifferentStateEnc import DetSARSA_variant
from SARSA.SARSA import SARSA
from Qlearning.Qlearning import Qlearning
import sys
sys.path.append('../')
from Environments.BlockingTask import BlockerTask
# from Environments.TransportationTask import *

import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
import json
import numpy as np
from tqdm import tqdm


OPTIMAL_SCORE = -0.578922
PALATINATE = "#72246C"
'''
fns invoking each algorithm:
    input: # of steps
    output: write average reward data to json
fn that plots each algorithms data depending on which is suplied
'''
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

def detSarsaMean(path, env="BlockerTask", repeats=5, steps=40000):
    env = BlockerTask() if env=="BlockerTask" else TransportationTask()
    
    rec_reward = {0:[-1 for i in range(repeats)]}
    for i in tqdm(range(repeats)):
        agent = DeterminantalSARSA( env )
        rews = agent.run_no_rec( steps )
        rec_reward = updateRewardList(rec_reward, rews)
    rec_reward = computeMean(rec_reward)

    with open(path, 'w') as outfile:
        outfile.write(json.dumps(rec_reward))

def sarsaMean(path, env="BlockerTask", repeats=5, steps=300000):
    env = BlockerTask() if env=="BlockerTask" else TransportationTask()
    
    rec_reward = {0:[-1 for i in range(repeats)]}
    for i in tqdm(range(repeats)):
        agent = SARSA( env, 0.1 )
        rews = agent.run( epLen=40, mxsteps=steps, rec_any=False )
        rec_reward = updateRewardList(rec_reward, rews)
    rec_reward = computeMean(rec_reward)
    
    with open(path, 'w') as outfile:
        outfile.write(json.dumps(rec_reward))

def myDetQlearningMean(path, env="BlockerTask", repeats=5, steps=40000):
    env = BlockerTask() if env=="BlockerTask" else TransportationTask()
    
    rec_reward = {0:[-1 for i in range(repeats)]}
    for i in tqdm(range(repeats)):
        agent = DeterminantalQlearning( env )
        rews = agent.run_no_rec( steps )
        rec_reward = updateRewardList(rec_reward, rews)
    rec_reward = computeMean(rec_reward)

    with open(path, 'w') as outfile:
        outfile.write(json.dumps(rec_reward))

def QlearningMean(path, env="BlockerTask", repeats=5, steps=300000):
    env = BlockerTask() if env=="BlockerTask" else TransportationTask()
    
    rec_reward = {0:[-1 for i in range(repeats)]}
    for i in tqdm(range(repeats)):
        agent = Qlearning( env, 0.1 )
        rews = agent.run( epLen=40, mxsteps=steps, rec_any=False )
        rec_reward = updateRewardList(rec_reward, rews)
    rec_reward = computeMean(rec_reward)
    
    with open(path, 'w') as outfile:
        outfile.write(json.dumps(rec_reward))

def detSarsaVariantMean(path, env="BlockerTask", repeats=5, steps=40000):
    env = BlockerTask() if env=="BlockerTask" else TransportationTask()
    
    rec_reward = {0:[-1 for i in range(repeats)]}
    for i in tqdm(range(repeats)):
        agent = DetSARSA_variant( env )
        rews = agent.run_no_rec( steps )
        rec_reward = updateRewardList(rec_reward, rews)
    rec_reward = computeMean(rec_reward)

    with open(path, 'w') as outfile:
        outfile.write(json.dumps(rec_reward))

def simulate(repeats, env, to_simulate):
    for (fn, (simulate, path, steps)) in to_simulate.items():
        if simulate:
            # globals is a fn that returns a dictionary of all fns in the file
            globals()[fn](path, env, repeats, steps)

def compute_snd_devs(rec_reward):
    std_dev = []
    for (t, r) in rec_reward.items():
        std_dev.append( (t, np.std(r)) )
    std_dev = sorted(std_dev, key=lambda x:int(x[0]))
    yerr = [sd for (_, sd) in std_dev]
    return yerr

def plotMultiple(data, mask=None, fn='all_algorithms'):
    plot_data = {}
    for algo, (path,c) in data.items():
        with open(path, 'r') as f:
            rec_reward = json.load(f)
        if 'yerr' in rec_reward:
            xs, ys, yerr = rec_reward['x'], rec_reward['y'], rec_reward['yerr']
            yerr = yerr[0]
        else:
            reward_av = rec_reward['average']
            del rec_reward['average']

            yerr = compute_snd_devs(rec_reward) 

            reward_av = [(int(k), v) for k, v in reward_av.items()]
            reward_av = sorted(reward_av, key=lambda x:x[0])
            [xs, ys] = list(zip(*reward_av))
            xs, ys, yerr = np.array(xs), np.array(ys), np.array(yerr)
            if mask is not None:
                m = [i for i in range(len(xs)) if i%3==0]
                xs, ys, yerr = xs[m], ys[m], yerr[m]
        plot_data[algo] = [np.array(xs), np.array(ys), np.array(yerr), c]

    plt.plot([0, 30], [OPTIMAL_SCORE, OPTIMAL_SCORE], c='g', lw=0.5, label='Optimal')

    # cols = [PALATINATE, 'r', 'm', 'c', 'lightgreen', 'b'] + list(mcolors.CSS4_COLORS)
    shift = 0
    for algo, [xs, ys, yerr, c] in plot_data.items():
        mask = xs<320000
        xs, ys, yerr = xs[mask], ys[mask], yerr[mask]
        plt.errorbar(xs/10000+shift, ys, yerr=yerr, c=c, lw=0.5, label=algo)
        shift -= 0.02

    plt.ylim([-1.05,-0.55])
    plt.legend(loc='lower right')
    plt.title('Algorithm comparison on the blocker task')
    plt.xlabel('Number of actions taken $\\times$ 10000')
    plt.ylabel('Average reward per action')
    plt.savefig('plots/'+fn, dpi=700)
    plt.show()

def zoomedPlot(data, osogami=True, all_runs=True, errBar=True, fname="zoomedLearningCurves"):
    plot_data = {}
    cols = ['r', PALATINATE, 'b', 'm', 'c', 'k', 'y']
    for algo, path in data.items():
        with open(path, 'r') as f:
            rec_reward = json.load(f)

        reward_av = rec_reward['average']
        del rec_reward['average']
        yerr = compute_snd_devs(rec_reward) 
        reward_av = [(int(k), v) for k, v in reward_av.items()]
        reward_av = sorted(reward_av, key=lambda x:x[0])
        [xms, yms] = list(zip(*reward_av))
        plot_data[algo+' average'] = [xms, yms]

    
        rec_reward = [(int(k), v) for k, v in rec_reward.items()]
        rec_reward = sorted(rec_reward, key=lambda x:x[0])

        repeats = len(rec_reward[0][1])
        recs = [[(time, listt[i]) for (time, listt) in rec_reward] for i in range(repeats)]

        if all_runs:
            others = [list(zip(*rec_)) for rec_ in recs]
            col = cols.pop(0)
            for i, [x, y] in enumerate(others):
                x = np.array(x)/1000
                if i==0:
                    plt.plot(x, y, c=col, lw=0.3, label=str(algo)+' runs 1-'+str(len(others)))
                else:
                    plt.plot(x, y, c=col, lw=0.3)
        xms, yms, yerr = np.array(xms)/1000, np.array(yms), np.array(yerr)
        if errBar:
            plt.errorbar(xms, yms, yerr=yerr, c=cols.pop(0), lw=0.5, label=algo)
        # else:
        #     plt.plot(xms, yms, yerr, cols.pop(0), lw=0.5, label=algo+' average')

    plt.plot([0, 40], [OPTIMAL_SCORE, OPTIMAL_SCORE], c='g', lw=0.5, label='Optimal')

    if osogami:
        with open('plots/datasets/OsogamiRaymond/Osogami-plot-data.json', 'r') as f:
            osg = json.load(f)
        x, y, yerr = np.array(osg['x']), np.array(osg['y']), np.array(osg['yerr'])
        plt.errorbar(x/1000, y, yerr, c='k', lw=0.5, label='Determinantal SARSA\nOsogami & Raymond')

    plt.ylim([-1.05,-0.55])
    plt.xlim([0, 40])
    plt.legend(loc='upper left')
    # plt.title('Determinantal algorithms on the blocker task')
    plt.xlabel('Number of actions taken $\\times$ 1000')
    plt.ylabel('Average reward per action')
    plt.savefig('plots/'+fname, dpi=400)
    plt.show()


if __name__ == '__main__':
    repeats = 10
    p = 'plots/datasets/'
    path1 = p+'det-sarsa-data-'+str(repeats)+'runs.json'
    path2 = p+'sarsa-data-'+str(repeats)+'runs.json'
    path3 = p+'my-det-Q-learning-data-'+str(repeats)+'runs.json'
    path4 = p+'my-Q-learning-data-'+str(repeats)+'runs.json'
    path5 = p+'my-detSarsaVariatMean-data-'+str(repeats)+'runs.json'

    env = "BlockerTask"

    to_simulate = {'detSarsaMean':        (False, path1, 150000), 
                   'sarsaMean':           (False, path2, 300000),
                   'myDetQlearningMean':  (False, path3, 150000),
                   'QlearningMean':       (False, path4, 300000),
                   'detSarsaVariantMean': (False, path5, 40000),}

    simulate(repeats, env, to_simulate)

    # data = {}
    data = {'DetSARSA':(path1, PALATINATE), 'SARSA':(path2, 'c'),
            'DetQ-learning':(path3, 'r'), 'Q-learning':(path4, 'b')}
    plotMultiple(data, fn='algorithm_comparison')

    # data = {'DetSARSA':path1, 'DetQ-learning':path3}#, 'DetSARSA-variant':path5}
    # zoomedPlot(data, all_runs=False)
    data = {'DetSARSA-Alternate':path5}
    zoomedPlot(data, all_runs=True, errBar=False, fname='AlternateForm')
    data = {'DetSARSA-Mine':path1}
    zoomedPlot(data, all_runs=True, errBar=False, fname='MyForm')



