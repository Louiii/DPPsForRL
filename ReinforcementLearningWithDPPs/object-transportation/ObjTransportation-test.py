from Environments.TransportationTask import *


class SARSA:
    def __init__(self, env, α=0.5, γ=1, bs=10000):
        self.env = env
        self.α = α
        self.γ = γ
        self.Q = dict()

        self.bin_size = bs
        
    # def convert(self, o):
    #     if isinstance(o, np.int64): return int(o) 
        
    def f(self, state):
        x = [0]*(6*7+2)
        ((r1, c1), gr1, (r2, c2), gr2) = state
        x[r1*7 + c1] = 1
        x[r2*7 + c2] = 1
        x[-2] = gr1
        x[-1] = gr2
        return tuple(x)
            
    def q(self, x):
        # x = self.f(s)
        # x = (s, a)
        if x in self.Q:
            return self.Q[x]
        else:
            return 0
        
    def policy(self, state, ε=0.3):
        actions = self.env.valid_actions( state )
        if ε < np.random.rand():
            qa = []
            for a in actions:
                ns, r, d = self.env.step(state, a, searchingPolicy=True)
                nx = self.f( ns )
                qa.append( (a, self.q( nx ) ) )
            mx = max(qa, key=lambda x:x[1])
            # if mx[1]!=0:
            return mx[0]
        return actions[np.random.randint(len(actions))]
    
    def episode(self, tstart, epi_max, rec=False):
        self.env.reset_state()
        state  = self.env.state
        action = self.policy( state )
        x = self.f(state)

        for t in range(1, epi_max+1):
            next_state, reward, done = self.env.step(state, action)
            
            if rec:
            	title = 'r = '+str(reward)+', ended = '+str(done)+', : iter = '
            	xaxis = 'ns: '+str(next_state)
            	record(t+tstart, title, xaxis, state, self.env.object, self.env, up=False)
            
            # next_action = self.policy(next_state)# either this line or the one above 'if done'
            
            # state, next_state = tuple(state), tuple(next_state)
            next_x = self.f(next_state)
            update = self.α*(reward + self.γ*self.q(next_x) - self.q(x))
            # x = self.f(state)
            # x = (state, action)
            # print(x)
            if x in self.Q:
                self.Q[x] += update
            else:
                self.Q[x] = update
            
            
            
            self.total_reward += reward
            if (tstart+t) % self.bin_size == 0:
                print('average_reward ; ', str(self.total_reward/self.bin_size))
                self.rec_reward.append( (tstart+t, self.total_reward/self.bin_size ) )
                self.total_reward = 0
            
            
            state = next_state
            action = self.policy(state)
            x = next_x
            
            if done: return t
        return epi_max
        
    
    def run(self, mxsteps=1000, interval=10, rec_any=True):
        """ on-policy TD control for estimating Q """
        self.rec_reward = []
        self.total_reward = 0
        
        t = 0
        rec = False
        ep = 0
        while t <= mxsteps :
            ep += 1
            if ep%200==0:
                print(str(t)+', lenQ '+str(len(self.Q)))
            
            # if t > mxsteps - 200 and rec_any: rec= True
            
            dt = self.episode(t, 500, rec)
            print(dt)
            t += dt
            
            
            
        return self.rec_reward



env = TransportationTask()
# state = ((3, 2), True, (3, 4), True)

# (state, reward, ended) = env.step( state, ((1,0), (1,0)) )
# record(0, 'r = '+str(reward)+', ended = '+str(ended)+', : iter = ', state, env.object, env)

# for i in range(1, 400):
# 	actions = env.all_actions
# 	action = actions[np.random.randint(len(actions))]
# 	(state, reward, ended) = env.step( state, action )
# 	record(i, 'r = '+str(reward)+', ended = '+str(ended)+', : iter = ', state, env.object, env)

# makeGIF('plots/temp-plots', 'plots/ObjTransportation')

## next step, how to stop env.object moving during policy evaluation

sarsa = SARSA(env, α=0.1, γ=0.95, bs=30000)
rewards = sarsa.run(1.5e6)
# makeGIF('plots/temp-plots/temp-plots1', 'plots/SARSA-ObjTransportation')

[x,y] = list(zip(*rewards))
plt.plot(x, y)
plt.savefig('plots/SARSA-learning-curve-Transportation', dpi=400)
plt.show()



# class SARSA:
#     def __init__(self, env, α=0.5, γ=1):
#         self.env = env
#         self.α = α
#         self.γ = γ
#         self.Q = dict()
        
#     def convert(self, o):
#         if isinstance(o, np.int64): return int(o) 
    
# #    def saveQ(self):
# #        with open('plots/Qtable.json', 'w') as outfile:
# #            outfile.write(json.dumps(list(self.Q.items()), default=self.convert))
# #
# #    def loadQ(self):
# #        try:
# #            print('Loading Q-Table...')
# #            with open("plots/Qtable.json","r") as f:
# #                self.Q = dict()
# #                for [[s, a], q] in json.loads(f.read()):
# #                    s = tuple([(np.int64(x[0]), np.int64(x[1])) for x in s])
# #                    a = tuple([(np.int64(x[0]), np.int64(x[1])) for x in a])
# #                    self.Q[(s, a)] = q
# #        except IOError:
# #            print("No previous Q-Table found, making a new one...")
# #            self.Q = dict()
        
#     def f(self, state):
#         # x = np.zeros(4*7)
#         # for (i, j) in state:
#         #     ind = i*7 + j
#         #     x[ind] = 1
#         return state#tuple(x)
            
#     def q(self, s, a):
#         # x = self.f(s)
#         x = (s, a)
#         if x in self.Q:
#             return self.Q[x]
#         else:
#             return 0
        
#     def policy(self, state, ε=0.1):
#         actions = self.env.valid_actions( state )
#         if ε < np.random.rand():
#             qa = []
#             for a in actions:
#                 ns, r, d = self.env.step(state, a, searchingPolicy=True)
#                 qa.append( (a, self.q( ns, a ) ) )
#             mx = max(qa, key=lambda x:x[1])
#             if mx[1]!=0:
#             	return mx[0]
#         return actions[np.random.randint(len(actions))]
    
#     def episode(self, tstart, epi_max, rec=False):
#         self.env.reset_state()
#         state  = self.env.state
#         action = self.policy( state )
        

#         for t in range(1, epi_max+1):
#             next_state, reward, done = self.env.step(state, action)
            
#             if rec:
#             	record(t, 'r = '+str(reward)+', ended = '+str(done)+', : iter = ', state, env.object, env)
            
#             next_action = self.policy(next_state)# either this line or the one above 'if done'
            
#             # state, next_state = tuple(state), tuple(next_state)
#             update = self.α*(reward + self.γ*self.q(next_state, next_action) - self.q(state, action))
#             # x = self.f(state)
#             x = (state, action)
#             if x in self.Q:
#                 self.Q[x] += update
#             else:
#                 self.Q[x] = update
            
            
            
#             self.total_reward += reward
#             if (tstart+t) % 40000 == 0:
#                 print('average_reward ; ', str(self.total_reward/40000))
#                 self.rec_reward.append( (tstart+t, self.total_reward/40000 ) )
#                 self.total_reward = 0
            
            
#             state = next_state
#             action = next_action#self.policy(state)
            
#             if done: return t
#         return epi_max
        
    
#     def run(self, mxsteps=1000, interval=10, rec_any=True):
#         """ on-policy TD control for estimating Q """
#         self.rec_reward = []
#         self.total_reward = 0
        
#         t = 0
#         rec = False
#         ep = 0
#         while t <= mxsteps :
#             ep += 1
#             if ep%200==0:print(t)
            
#             if t > mxsteps - 200 and rec_any: rec= True
            
#             dt = self.episode(t, 100, rec)
#             t += dt
            
            
            
#         return self.rec_reward
