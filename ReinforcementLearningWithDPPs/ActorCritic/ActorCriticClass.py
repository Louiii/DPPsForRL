import os#, gym
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

# from BlockerTaskACWrapper import BlockerEnv
# from TransportationTaskACWrapper import TransportationEnv
import sys
sys.path.append('../')
from Environments.BlockingTask import BlockerEnv
from Environments.TransportationTask import TransportationEnv

'''

REFERENCES:


https://github.com/yc930401/Actor-Critic-pytorch

https://github.com/higgsfield/RL-Adventure-2/blob/master/1.actor-critic.ipynb

https://arxiv.org/pdf/1509.02971.pdf

'''


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.lin1 = nn.Linear(state_dim, 128)
        self.lin2 = nn.Linear(128, 256)
        self.lin3 = nn.Linear(256, action_dim)

    def forward(self, state):
        out1 = F.relu(self.lin1(state))
        out2 = F.relu(self.lin2(out1))
        out3 = self.lin3(out2)
        return Categorical(F.softmax(out3, dim=-1))# pmf over possible actions


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.lin1 = nn.Linear(state_dim, 128)
        self.lin2 = nn.Linear(128, 256)
        self.lin3 = nn.Linear(256, 1)

    def forward(self, state):
        out1 = F.relu(self.lin1(state))
        out2 = F.relu(self.lin2(out1))
        return self.lin3(out2)# Approximated Q-value

class AC:
    def __init__(self, env, envType, initAC=False):
        self.env, self.envType = env, envType
        self.state_dim, self.action_dim = env.state_dim, env.action_dim

        if initAC:
            self.init_ac()
        else:
            self.load()

        self.bin = 40000

    def init_ac(self):
        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim, self.action_dim)

    def load(self):
        if os.path.exists('model/actor'+self.envType+'.pkl'):
            self.actor = torch.load('model/actor'+self.envType+'.pkl')
            print('Actor loaded')
        else:
            self.actor = Actor(self.state_dim, self.action_dim)
        if os.path.exists('model/critic'+self.envType+'.pkl'):
            self.critic = torch.load('model/critic'+self.envType+'.pkl')
            print('Critic loaded')
        else:
            self.critic = Critic(self.state_dim, self.action_dim)

    def compute_returns(self, next_value, rewards, masks, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns

    def trainIters(self, total_iters, render=True, max_iters=10000, save=True, dpi=40):
        optimiserA = optim.Adam(self.actor.parameters())
        optimiserC = optim.Adam(self.critic.parameters())

        total_time, total_reward, all_rewards = 0, 0, []

        ep_lengths = []
        # for iter in tqdm(range(n_iters)):
        it = 0
        while total_iters > it:

            state = self.env.reset()
            log_probs = []
            values = []
            rewards = []
            masks = []
            # entropy = 0
            # env.reset()

            # run episode:
            for i in count():
                total_time += 1
                if render: self.env.render(dpi=dpi)

                state = torch.FloatTensor(state)
                dist, value = self.actor(state), self.critic(state)

                action = dist.sample()
                next_state, reward, done, _ = self.env.step(action.cpu().numpy())

                log_prob = dist.log_prob(action).unsqueeze(0)
                # entropy += dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.tensor([reward], dtype=torch.float))
                masks.append(torch.tensor([1-done], dtype=torch.float))

                state = next_state

                #####################
                total_reward += reward
                if total_time%self.bin==0:
                    print(str(total_time)+", "+str(total_reward/self.bin))
                    all_rewards.append( (total_time, total_reward/self.bin ) )
                    total_reward = 0
                #####################
                if done or max_iters < i:
                    it += i
                    ep_lengths.append(i)
                    # print('Iteration: {}, Ep length: {}'.format(it, i))
                    break

            if render: self.env.render(dpi=dpi)


            next_value = self.critic( torch.FloatTensor(next_state) )
            returns = self.compute_returns(next_value, rewards, masks)

            log_probs = torch.cat(log_probs)
            returns = torch.cat(returns).detach()
            values = torch.cat(values)

            advantage = returns - values

            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()


            optimiserA.zero_grad()
            optimiserC.zero_grad()

            actor_loss.backward()
            critic_loss.backward()

            optimiserA.step()
            optimiserC.step()

        if save:
            torch.save(self.actor, 'model/actor'+self.envType+'.pkl')
            torch.save(self.critic, 'model/critic'+self.envType+'.pkl')
        self.env.close()
        return ep_lengths, all_rewards

    def learning_curve(self, n=20000):
        ep_lengths, all_rewards = run(n, 1000)
        
        import matplotlib.pyplot as plt
        import json

        with open('../plots/datasets/actor-critic-'+envType+'-plot-data.json', 'w') as outfile:
            outfile.write(json.dumps(all_rewards))

        with open('../plots/datasets/actor-critic-'+envType+'-plot-data.json', 'r') as f:
            all_rewards = json.load(f)

        [x, y] = list(zip(*all_rewards))
        plt.plot(x, y)
        plt.savefig('../plots/AC'+envType+'learning_rate', dpi=600)
        plt.show()

    def render_episodes(self, n=10):
        self.load()
        ep_lengths = self.trainIters(n_iters=n, render=True, dpi=150)
        env.output_GIF()

    def run(self, n, maxEpLen):
        _, all_rewards = self.trainIters(n, render=False, max_iters=maxEpLen, save=False)
        return all_rewards

if __name__ == '__main__':
    
    env = BlockerEnv()#gym.make("CartPole-v0").unwrapped
    # env = TransportationEnv()
    envType = 'Blkr'
    # envType = 'Tspt'
    ac = AC(env, envType)
    ac.render_episodes(n=5)


