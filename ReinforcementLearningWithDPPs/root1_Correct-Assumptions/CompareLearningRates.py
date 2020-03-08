from DeterminantalSARSA.DeterminantalSARSA_Model import DetSARSA_Model
from Models import Table, NN
from SARSA.SARSA_noBoltzStep import SARSA_noBoltzStep
from ActorCritic.ActorCriticClass import *
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

def detSarsaModelMean(path, env="BlockerTask", repeats=5, steps=40000):
    env = BlockerTask() if env=="BlockerTask" else TransportationTask()
    
    rec_reward = {0:[-1 for i in range(repeats)]}
    for i in tqdm(range(repeats)):
        env = BlockerTask()
        m = Table_model()
        agent = DetSARSA_Model( env, m )
        rews = agent.run_no_rec( steps )
        rec_reward = updateRewardList(rec_reward, rews)
    rec_reward = computeMean(rec_reward)

    with open(path, 'w') as outfile:
        outfile.write(json.dumps(rec_reward))

def detSarsaNNModelMean(path, env="BlockerTask", repeats=5, steps=40000):
    env = BlockerTask() if env=="BlockerTask" else TransportationTask()
    
    rec_reward = {0:[-1 for i in range(repeats)]}
    for i in tqdm(range(repeats)):
        env = BlockerTask()
        m = NN_model()
        agent = DetSARSA_Model( env, m )
        rews = agent.run_no_rec( steps )
        rec_reward = updateRewardList(rec_reward, rews)
    rec_reward = computeMean(rec_reward)

    with open(path, 'w') as outfile:
        outfile.write(json.dumps(rec_reward))

def sarsaNoBoltzStepMean(path, env="BlockerTask", repeats=5, steps=300000):
    env = BlockerTask() if env=="BlockerTask" else TransportationTask()
    
    rec_reward = {0:[-1 for i in range(repeats)]}
    for i in tqdm(range(repeats)):
        agent = SARSA_noBoltzStep( env, α=0.5, γ=0.9, ε=0.3 )
        rews = agent.run( epLen=40, mxsteps=steps, rec_any=False )
        rec_reward = updateRewardList(rec_reward, rews)
    rec_reward = computeMean(rec_reward)
    
    with open(path, 'w') as outfile:
        outfile.write(json.dumps(rec_reward))

def actorCriticMean(path, env="BlockerTask", repeats=5, steps=300000):
    env = BlockerEnv() if env=="BlockerTask" else TransportationEnv()
    
    rec_reward = {0:[-1 for i in range(repeats)]}
    for i in tqdm(range(repeats)):
        agent = AC( env, "Blkr", initAC=True )
        rews = agent.run( steps, 2000 )
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
    plt.clf()

if __name__ == '__main__':
    repeats = 10
    p = 'plots/datasets/'
    path1 = p+'actor-critic-data-'+str(repeats)+'runs.json'
    path2 = p+'my-detSarsaModelMean-data-'+str(repeats)+'runs.json'
    path3 = p+'my-detSarsaModelMean2-data-'+str(repeats)+'runs.json'
    repeats = 1
    path4 = p+'my-detSarsaNNModelMean-data-'+str(repeats)+'runs.json'
    path5 = p+'my-sarsa-noBoltzStep-data-'+str(repeats)+'runs.json'

    env = "BlockerTask"

    to_simulate = {'actorCriticMean':     (False, path1, 300000),
                   'detSarsaModelMean':   (False, path2, 250000),
                   'detSarsaModelMean':   (False, path3, 250000),
                   'detSarsaNNModelMean': (False, path4, 250000),
                   'sarsaNoBoltzStepMean':(False, path5, 300000)}

    repeats = 10
    simulate(repeats, env, to_simulate)

    # TESTING MODEL BASED:
    data = {'Actor-Critic':(path1, 'm'),'detSarsaModelMean':(path2, PALATINATE), 'sarsaNoBoltzStepMean':(path5, 'r')}
    plotMultiple(data, fn='model_comparison')


    SilverHessTeh = {'ENATDAC':(p+'HessSilverTeh/ENATDAC-plot-data.json', 'c'),
                     'EQNAC':(p+'HessSilverTeh/EQNAC-plot-data.json', 'b'),
                     'NN':(p+'HessSilverTeh/NN-plot-data.json', 'm'),
                     'ESARSA':(p+'HessSilverTeh/ESARSA-plot-data.json', 'r')}
    data = {}
    data.update({'DetSARSA-Model1':(path2, PALATINATE)})#, 'detSarsaNNModelMean':path4})
    data.update(SilverHessTeh)
    plotMultiple(data, fn='hess_comparison')



