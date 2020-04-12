import numpy as np
import sys
sys.path.append('../../')
from Environments.BlockingTask import *
import json
from tqdm import tqdm



env = BlockerTask()

ind_to_state = lambda k: (k//7, k%7)
ind_states = list(range(21))
def gen_sublists(thelist, k):
    n = len(thelist)
    if k == 0:
        yield list(), 0
    elif k == n:
        yield list(thelist), n
    elif k < n:
        for sublist, idx in gen_sublists(thelist[:-1], k - 1):
            for i, item in enumerate(thelist[idx:], idx):
                yield sublist + [item], i + 1

def choose_sets1(thelist, k):
    if not 0 <= k <= len(thelist):
        raise ValueError((k, len(thelist)))
    combinations = [ L for L, i in gen_sublists(thelist, k) ]
    unordered = [(ind_to_state(a), ind_to_state(b), ind_to_state(c)) for (a,b,c) in combinations]
    allorders = unordered[::]
    for i, j, k in [(0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]:
        for u in unordered:
            allorders.append( (u[i], u[j], u[k]) )
    return allorders

states = choose_sets1(ind_states, 3)

data = []
for s in tqdm(states):
	for a in env.all_actions:
		# this is not a complete dataset!! doesn't include 
		# every possible combination of blocker states
		env.blockers_state = np.random.choice([0, 3, 6])

		ns, r, _ = env.step(s, a)
		data.append([s, a, ns, r])

with open('env_dataset', 'w') as outfile:
    outfile.write(json.dumps(data))



