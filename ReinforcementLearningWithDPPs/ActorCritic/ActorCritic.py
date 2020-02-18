import os#, gym
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

from BlockerTaskACWrapper import BlockerEnv
from TransportationTaskACWrapper import TransportationEnv

'''

REFERENCES:


https://github.com/yc930401/Actor-Critic-pytorch

https://github.com/higgsfield/RL-Adventure-2/blob/master/1.actor-critic.ipynb

https://arxiv.org/pdf/1509.02971.pdf

'''


env = BlockerEnv()#gym.make("CartPole-v0").unwrapped
# env = TransportationEnv()
envType = 'Blkr'
# envType = 'Tspt'


state_dim = env.state_dim#env.observation_space.shape[0]
action_dim = env.action_dim#env.action_space.n
# lr = 0.0001

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


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def trainIters(actor, critic, n_iters, render=True, max_iters=100000, save=True, dpi=40):
    optimiserA = optim.Adam(actor.parameters())
    optimiserC = optim.Adam(critic.parameters())

    total_time, total_reward, all_rewards = 0, 0, []

    ep_lengths = []
    for iter in tqdm(range(n_iters)):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        masks = []
        # entropy = 0
        # env.reset()

        # run episode:
        for i in count():
            total_time += 1
            if render: env.render(dpi=dpi)

            state = torch.FloatTensor(state)
            dist, value = actor(state), critic(state)

            action = dist.sample()
            next_state, reward, done, _ = env.step(action.cpu().numpy())

            log_prob = dist.log_prob(action).unsqueeze(0)
            # entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float))
            masks.append(torch.tensor([1-done], dtype=torch.float))

            state = next_state

            #####################
            total_reward += reward
            if total_time%3000==0:
                all_rewards.append( (total_time, total_reward/3000 ) )
                total_reward = 0
            #####################
            if done or max_iters < i:
                ep_lengths.append(i)
                print('Iteration: {}, Ep length: {}'.format(iter, i))
                break

        if render: env.render(dpi=dpi)


        next_value = critic( torch.FloatTensor(next_state) )
        returns = compute_returns(next_value, rewards, masks)

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
        torch.save(actor, 'model/actor'+envType+'.pkl')
        torch.save(critic, 'model/critic'+envType+'.pkl')
    env.close()
    return ep_lengths, all_rewards

def load():
    if os.path.exists('model/actor'+envType+'.pkl'):
        actor = torch.load('model/actor'+envType+'.pkl')
        print('Actor loaded')
    else:
        actor = Actor(state_dim, action_dim)
    if os.path.exists('model/critic'+envType+'.pkl'):
        critic = torch.load('model/critic'+envType+'.pkl')
        print('Critic loaded')
    else:
        critic = Critic(state_dim, action_dim)
    return actor, critic

def init_ac():
    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim, action_dim)
    return actor, critic
        
def run(n, maxEpLen):
    ep_lengths, all_rewards = trainIters(actor, critic, n_iters=n, render=False, max_iters=maxEpLen, save=False)
    return all_rewards

def learning_curve(n=20000):
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

def render_episodes(n=10):
    actor, critic = load()
    ep_lengths = trainIters(actor, critic, n_iters=n, render=True, dpi=150)
    env.output_GIF()


if __name__ == '__main__':
    # a, c = load()
    # a, c = init_ac()
    # ep_lengths, all_rewards = trainIters(a, c, n_iters=1000, render=False, max_iters=1000)

    render_episodes(n=10)
    # learning_curve(n=500)