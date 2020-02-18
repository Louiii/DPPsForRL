import sys
sys.path.append('../')
from Environments.TransportationTask import TransportationTask, makeGIF, record
import numpy as np

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