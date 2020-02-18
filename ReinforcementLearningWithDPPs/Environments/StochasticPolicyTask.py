import numpy as np

def per(n):
    list_ = []
    for i in range(1, 2**n):
        s=bin(i)[2:]
        s='0'*(n-len(s))+s
        list_.append( list(map(int,list(s))) )
    return list_

class SPT:
    def __init__(self, N, n_correct):
        self.N = N
        self.n_correct = n_correct
        self.states = [None]# nothing is observed (None, None)
        self.hidden_states = per(N)
        self.actions = self.hidden_states
        self.valid_actions = lambda _:self.actions

        # alternate between two hidden reward states.
        idxs = np.random.choice(range(len(self.hidden_states)), n_correct, replace=False)
        self.correct_states = tuple([self.hidden_states[idxs[i]] for i in range(n_correct)])
        # self.correct_states = (self.hidden_states[10], self.hidden_states[6])
        self.current = 0

        self.toString = lambda l:''.join([str(a) for a in l])

        self.retTuple = False

    def reset_state(self):
        return self.states[0]

    # def step(self, state, action, searchingPolicy=False):
    #     if self.retTuple: action = tuple(action)
    #     if searchingPolicy: return action, 0, False

    #     if self.toString(action)==self.toString(self.correct_states[self.current]):
    #         self.current = (self.current + 1)%self.n_correct
    #         # print('\n\nFOUND\n\n')
    #         return action, 10, False
    #     return action, 0, False

    def step(self, state, action, searchingPolicy=False):
        # if self.retTuple: action = tuple(action)
        if searchingPolicy: return self.states[0], 0, False

        if self.toString(action)==self.toString(self.correct_states[self.current]):
            self.current = (self.current + 1)%self.n_correct
            # print('\n\nFOUND\n\n')
            return self.states[0], 10, False
        return self.states[0], 0, False

# env = SPT(5, 2)
# print(env.step(None, [0,0,0,0,0]))
# print(env.step(None, [0,0,0,0,1]))
# cs = env.correct_states
# print(env.step(None, cs[0]))
# print(env.step(None, cs[0]))
# print(env.step(None, cs[0]))
# print(env.step(None, cs[0]))
# print(env.step(None, cs[1]))
# print(env.step(None, cs[0]))
# print(env.step(None, cs[1]))
# print(env.step(None, cs[0]))
# print(env.step(None, cs[1]))
# print(env.step(None, cs[0]))
