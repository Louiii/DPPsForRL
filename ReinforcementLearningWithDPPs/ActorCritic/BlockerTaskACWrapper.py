import sys
sys.path.append('../')
from Environments.BlockingTask import BlockerTask
from PrintState import maze_record, makeGIF
import numpy as np
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